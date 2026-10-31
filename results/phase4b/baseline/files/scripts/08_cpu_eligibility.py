"""Measure KoNViD raw-video-to-score CPU time for all frozen candidates."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import predict_ridge
from src.cache import atomic_json
from src.features import extract_features, load_backbone, sha256
from src.heads import build_head
from src.phase3 import load_phase3_context, require_cpu_winner, resolve_cpu_winner
from src.preprocess import iter_samples, spatial_tensor
from src.training import config_hash


def _load_candidates(context):
    result = {}
    for candidate in context['candidates']:
        feature, head = candidate['feature_set'], candidate['head']
        cv = candidate['refit']['cv_selected_alphas'][0] if head == 'ridge' else candidate['refit']['cv_epochs'][0]
        run_path = Path(cv['run_path'])
        checkpoint = run_path.parent / ('model.npz' if head == 'ridge' else 'best.pt')
        if sha256(run_path) != cv['run_sha256'] or sha256(checkpoint) != cv['checkpoint_sha256']:
            raise ValueError(f'changed CV candidate: {candidate["candidate_id"]}')
        if head == 'ridge':
            with np.load(checkpoint, allow_pickle=False) as data:
                model = {k: data[k] for k in data.files}
        else:
            model = build_head(head, 1536 if feature == 'dino' else 2048, candidate['config']).to('cpu').eval()
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['state_dict'])
        result[candidate['candidate_id']] = (model, cv['checkpoint_sha256'])
    return result


@torch.inference_mode()
def _head_time(model, head, tensor):
    start = time.perf_counter()
    if head == 'ridge':
        predict_ridge(model, tensor.numpy().mean(0, keepdims=True, dtype=np.float64))
    else:
        model(tensor.unsqueeze(0), torch.tensor([len(tensor)]))
    return time.perf_counter() - start


def measure(context):
    config = context['config']
    output = Path(config['results_root']) / 'cpu_eligibility.json'
    if output.exists():
        return require_cpu_winner(context)
    ids = Path(config['cpu_sample_ids']).read_text().splitlines()
    if len(ids) != 20 or len(set(ids)) != 20 or set(ids) - set(context['ids']['konvid']):
        raise ValueError('invalid fixed CPU sample')
    torch.set_num_threads(4)
    features = yaml.safe_load(Path(config['feature_config']).read_text())
    source_video = importlib.import_module('scripts.02_extract_spatial').source_video
    backbones = {name: load_backbone(name, 'cpu', features) for name in ('dino', 'resnet50')}
    heads = _load_candidates(context)
    sample = {r['video_id']: r for r in context['inventories']['konvid']}
    # Warm model kernels outside the measured raw-video path.
    for name, backbone in backbones.items():
        extract_features(backbone, torch.zeros(1, 3, 448, 448), name, 'cpu')
    for candidate in context['candidates']:
        dim = 1536 if candidate['feature_set'] == 'dino' else 2048
        _head_time(heads[candidate['candidate_id']][0], candidate['head'], torch.zeros(2, dim))
    rows = []
    for key in ids:
        source = sample[key]
        duration = float(source['duration_s'])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f'{key}: invalid duration')
        for repeat in range(config['cpu_repeats']):
            for feature, backbone in backbones.items():
                with source_video(source, features['datasets']['konvid']) as path:
                    start = time.perf_counter()
                    preproc = inference = 0.
                    vectors = []
                    for _, frame, _ in iter_samples(path, features):
                        tick = time.perf_counter()
                        tensor = spatial_tensor(frame).unsqueeze(0)
                        preproc += time.perf_counter() - tick
                        tick = time.perf_counter()
                        values = extract_features(backbone, tensor, feature, 'cpu')
                        inference += time.perf_counter() - tick
                        vectors.append(torch.cat((values['cls'], values['patchmean_L12']), dim=1) if feature == 'dino' else values['pooled'])
                    elapsed = time.perf_counter() - start
                decoded = max(0., elapsed - preproc - inference)
                tensor = torch.cat(vectors, dim=0)
                for candidate in (c for c in context['candidates'] if c['feature_set'] == feature):
                    head_seconds = _head_time(heads[candidate['candidate_id']][0], candidate['head'], tensor)
                    total = decoded + preproc + inference + head_seconds
                    rows.append({'candidate_id': candidate['candidate_id'], 'video_id': key, 'dataset': 'konvid',
                                 'repeat': repeat, 'duration_s': duration, 'decode_seconds': decoded,
                                 'preprocess_seconds': preproc, 'backbone_seconds': inference,
                                 'head_seconds': head_seconds, 'total_seconds': total,
                                 'normalized_total_seconds': 10 * total / duration,
                                 'source_sha256': source['source_sha256'],
                                 'config_sha256': config_hash(candidate['config']),
                                 'checkpoint_sha256': heads[candidate['candidate_id']][1]})
        print(f'CPU benchmark {key}: {len(rows)}/600 candidate repeats', flush=True)
    evidence = {'cpu_threads': 4, 'sample_ids': ids, 'repeats': 3, 'rows': rows,
                'phase2_gate_sha256': context['phase2_gate_sha256'],
                'freeze_sha256': context['freeze_sha256'], 'code_sha256': context['code_sha256']}
    evidence['winner'] = resolve_cpu_winner(context['freeze'], evidence)
    atomic_json(output, evidence)
    return require_cpu_winner(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(load_phase3_context(args.config))), flush=True)


if __name__ == '__main__':
    main()
