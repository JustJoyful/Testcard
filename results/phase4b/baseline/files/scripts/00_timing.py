"""Measure the fixed 20-video extraction pipeline on CUDA fp16 and CPU fp32."""
import argparse
import csv
import gc
import importlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import extract_features, load_backbone, sha256
from src.preprocess import iter_samples, motion_tensors, probe_video, spatial_tensor

FIELDS = ['video_id', 'backbone', 'device', 'precision', 'repeat', 'duration_s', 'T',
          'decode_s', 'preprocess_s', 'inference_s', 'total_s', 'peak_memory_bytes', 'status', 'failure_reason']


def input_tensor(frame, window, name, config):
    if name in ('dino', 'resnet50'):
        return spatial_tensor(frame).unsqueeze(0)
    tensor = motion_tensors(window, config)
    return [x.unsqueeze(0) for x in tensor] if isinstance(tensor, list) else tensor.unsqueeze(0)


def measure_video(path, model, name, device, config, clock=time.perf_counter):
    start = clock()
    meta = probe_video(path, timestamps=False)
    row = {'duration_s': meta['duration_s'], 'T': 0, 'decode_s': 0., 'preprocess_s': 0., 'inference_s': 0.}
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    iterator = iter_samples(path, config)
    try:
        while True:
            before = clock()
            try:
                _, frame, window = next(iterator)
            except StopIteration:
                row['decode_s'] += clock() - before
                break
            row['decode_s'] += clock() - before
            before = clock()
            samples = input_tensor(frame, window, name, config)
            row['preprocess_s'] += clock() - before
            if device == 'cuda':
                torch.cuda.synchronize()
            before = clock()
            result = extract_features(model, samples, name, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            row['inference_s'] += clock() - before
            expected = {'cls': 768, **{f'patch{stat}_L{n}': 768 for stat in ['mean', 'std'] for n in [6, 9, 12]}} if name == 'dino' else {'pooled': {'resnet50': 2048, 'slowfast_r50': 2304, 'r3d18': 512}[name]}
            if set(result) != set(expected) or any(tuple(result[key].shape) != (1, dimension) for key, dimension in expected.items()):
                raise ValueError(f'{path.name}: unexpected feature shapes')
            row['T'] += 1
    finally:
        if hasattr(iterator, 'close'):
            iterator.close()
    row['total_s'] = clock() - start
    if row['T'] != len(np.arange(0, meta['duration_s'], .25)):
        raise ValueError(f'{path.name}: sample count mismatch')
    row['peak_memory_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else None
    row['status'], row['failure_reason'] = 'ok', ''
    return row


def coverage_complete(config, rows):
    names = ['dino', 'resnet50', config['backbones']['motion']['candidate']]
    expected = {(key, name, device, repeat) for key in config['samples']['konvid'] for name in names
                for device in ['cuda', 'cpu'] for repeat in range(config['timing']['repeats'])}
    actual = [(r['video_id'], r['backbone'], r['device'], int(r['repeat'])) for r in rows]
    return len(config['samples']['konvid']) == 20 and len(rows) == len(expected) and set(actual) == expected and all(
        r['status'] == 'ok' and all(math.isfinite(float(r[k])) and float(r[k]) > 0 for k in ['decode_s', 'preprocess_s', 'inference_s', 'total_s']) for r in rows)


def run_benchmark(config, paths, save_row=None):
    ids = config['samples']['konvid']
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError('timing requires exactly 20 unique fixed IDs')
    torch.set_num_threads(config['timing']['cpu_threads'])
    rows, loads = [], []
    for name in ['dino', 'resnet50', config['backbones']['motion']['candidate']]:
        for device in ['cuda', 'cpu']:
            model, error = None, ''
            load_start = time.perf_counter()
            try:
                model = load_backbone(name, device, config)
                if device == 'cuda' and not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable')
                load_s = time.perf_counter() - load_start
                iterator = iter_samples(paths[ids[0]], config)
                try:
                    _, frame, window = next(iterator)
                    samples = input_tensor(frame, window, name, config)
                    for _ in range(config['timing']['warmup']):
                        extract_features(model, samples, name, device)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                finally:
                    if hasattr(iterator, 'close'):
                        iterator.close()
                loads.append({'backbone': name, 'device': device, 'load_s': load_s,
                              'warmup_s': time.perf_counter()-load_start-load_s, 'status': 'ok'})
            except Exception as exc:
                error = f'{name}/{device}: {type(exc).__name__}: {exc}'
                loads.append({'backbone': name, 'device': device, 'status': 'failed', 'failure_reason': error})
            for key in ids:
                for repeat in range(config['timing']['repeats']):
                    row = dict.fromkeys(FIELDS, '')
                    row.update(video_id=key, backbone=name, device=device,
                               precision='fp16' if device == 'cuda' else 'fp32', repeat=repeat)
                    try:
                        if error:
                            raise RuntimeError(error)
                        row.update(measure_video(paths[key], model, name, device, config))
                    except Exception as exc:
                        row.update(status='failed', failure_reason=f'{key}: {type(exc).__name__}: {exc}')
                    rows.append(row)
                    if save_row:
                        save_row(row)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, loads


def code_hash():
    import hashlib
    h = hashlib.sha256()
    for path in sorted([*Path('src').glob('*.py'), *Path('scripts').glob('00_*.py')]):
        h.update(str(path).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def evidence_hashes(config_path, config):
    import hashlib
    runtime = {key: config.get(key) for key in ['seed', 'samples', 'datasets', 'cache_version']}
    runtime['decoder'] = {k: v for k, v in config.get('decoder', {}).items() if k != 'confirmed'}
    runtime['preprocessing'] = {k: v for k, v in config.get('preprocessing', {}).items() if k != 'frozen'}
    runtime['backbones'] = {name: {k: v for k, v in settings.items() if k != 'confirmed'} for name, settings in config['backbones'].items()}
    runtime['timing'] = {key: config['timing'].get(key) for key in ['warmup', 'repeats', 'cpu_threads']}
    execution_hash = hashlib.sha256(json.dumps(runtime, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'config_sha256': sha256(config_path), 'execution_config_sha256': execution_hash,
            'lock_sha256': sha256('env/requirements.lock'), 'code_sha256': code_hash(),
            'manifests': {str(path): sha256(path) for path in sorted(Path('data/manifests').glob('*.csv'))},
            'weights': {key: value.get('weights_sha256') for key, value in config['backbones'].items()}}


def hashes_match(recorded, current):
    if not recorded or not current or not current.get('execution_config_sha256'):
        return False
    return {k: v for k, v in recorded.items() if k != 'config_sha256'} == {k: v for k, v in current.items() if k != 'config_sha256'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not all(v.get('confirmed') for v in config['backbones'].values()):
        parser.error('run 00_backbone_check.py successfully before timing')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    paths = importlib.import_module('scripts.00_inventory').prepare_sample_paths(config, 'konvid')
    output = Path('results/phase0')
    run = {'hashes': evidence_hashes(args.config, config), 'seed': config['seed'], 'fold': None,
           'head_checkpoint': None, 'code_commit': None, 'code_commit_note': 'Agent cannot write Git metadata; code content hash is recorded.', 'cache_version': config['cache_version'], 'cpu_threads': config['timing']['cpu_threads'],
           'warmup': config['timing']['warmup'], 'repeats': config['timing']['repeats'],
           'hardware': subprocess.run(['nvidia-smi'], text=True, capture_output=True).stdout,
           'decoder': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
           'cpu_memory_note': 'GPU allocator peak only; CPU peak_memory_bytes is null, not an estimate.',
           'status': 'running'}
    (output/'run.json').write_text(json.dumps(run, indent=2)+'\n')
    start = time.perf_counter()
    with (output/'timings.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader()
        def save(row):
            writer.writerow(row); f.flush()
            print(row['video_id'], row['backbone'], row['device'], row['repeat'], row['status'], flush=True)
        rows, loads = run_benchmark(config, paths, save)
    complete = coverage_complete(config, rows)
    run.update(status='complete' if complete else 'failed', model_loading=loads,
               benchmark_wall_s=time.perf_counter()-start)
    (output/'run.json').write_text(json.dumps(run, indent=2)+'\n')
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
