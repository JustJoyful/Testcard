"""Write raw LIVE-VQC predictions from all frozen full-KoNViD artifacts."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import predict_ridge
from src.cache import atomic_json
from src.data import InventoryFeatureDataset
from src.features import sha256
from src.heads import build_head
from src.phase3 import load_phase3_context, require_prediction_inputs, validate_live_rows
from src.training import predict


def predict_phase3(context):
    gate = require_prediction_inputs(context)
    config = context['config']
    candidates = {c['candidate_id']: c for c in context['candidates']}
    for feature in ('resnet50', 'dino'):
        dataset = InventoryFeatureDataset(Path(config['cache_root']), Path(config['live_inventory']), 'livevqc',
                                          feature, include_labels=False)
        for head in ('mean_pool', 'bigru', 'transformer', 'mamba', 'ridge'):
            for seed in ([None] if head == 'ridge' else range(5)):
                entry = gate['runs'][(feature, head, seed)]
                run, directory = entry['run'], entry['directory']
                predictions = directory / 'predictions.json'
                metadata = directory / 'predictions.meta.json'
                if predictions.exists() or metadata.exists():
                    if not predictions.exists() or not metadata.exists():
                        raise ValueError(f'incomplete immutable prediction artifact: {directory}')
                    meta = json.loads(metadata.read_text())
                    rows = json.loads(predictions.read_text())
                    if (meta.get('predictions_sha256') != sha256(predictions) or
                            meta.get('run_sha256') != entry['run_sha256'] or
                            meta.get('checkpoint_sha256') != run['checkpoint_sha256'] or
                            meta.get('phase2_gate_sha256') != context['phase2_gate_sha256'] or
                            meta.get('code_sha256') != context.get('code_sha256') or
                            meta.get('freeze_sha256') != context['freeze_sha256']):
                        raise ValueError(f'stale LIVE prediction artifact: {directory}')
                    validate_live_rows(context, rows, feature, head, seed, run)
                    continue
                if head == 'ridge':
                    with np.load(entry['checkpoint'], allow_pickle=False) as data:
                        artifact = {name: data[name] for name in data.files}
                    pooled = np.array([dataset[i]['features'].numpy().mean(0, dtype=np.float64)
                                       for i in range(len(dataset))])
                    values = predict_ridge(artifact, pooled)
                    raw = [{'video_id': dataset[i]['video_id'], 'prediction': float(value)}
                           for i, value in enumerate(values)]
                else:
                    frozen = context['freeze']['refit_by_feature'][feature][head]['config']
                    model = build_head(head, 1536 if feature == 'dino' else 2048, frozen)
                    saved = torch.load(entry['checkpoint'], map_location=frozen['device'], weights_only=True)
                    model.load_state_dict(saved['state_dict'])
                    model.to(frozen['device'])
                    raw = predict(model, dataset, run['target_stats'], frozen['device'])
                rows = [{**row, 'dataset': 'livevqc', 'feature_set': feature, 'head': head,
                         'seed': seed,
                         'checkpoint_sha256': run['checkpoint_sha256'], 'run_identity': run['identity']}
                        for row in raw]
                validate_live_rows(context, rows, feature, head, seed, run)
                atomic_json(predictions, rows)
                atomic_json(metadata, {'status': 'complete', 'predictions_sha256': sha256(predictions),
                                       'run_sha256': entry['run_sha256'],
                                       'checkpoint_sha256': run['checkpoint_sha256'],
                                       'phase2_gate_sha256': context['phase2_gate_sha256'],
                                       'code_sha256': context.get('code_sha256'),
                                       'freeze_sha256': context['freeze_sha256']})
                print(f'LIVE prediction {feature}/{head}/seed{seed}: {len(rows)} videos', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    predict_phase3(load_phase3_context(args.config, include_live_labels=False))


if __name__ == '__main__':
    main()
