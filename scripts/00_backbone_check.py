"""Download pinned pretrained weights and verify one real sample on CPU and CUDA.

Run from the project root in Kitty before 00_timing.py. No training or prediction.
"""
import argparse
import gc
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import extract_features, load_backbone, sha256
from src.preprocess import iter_samples


def check_models(config, frame, window):
    timing = importlib.import_module('scripts.00_timing')
    torch.set_num_threads(config['timing']['cpu_threads'])
    evidence = {'checks': [], 'status': 'failed'}
    for name in ['dino', 'resnet50', config['backbones']['motion']['candidate']]:
        checks = []
        for device in ['cuda', 'cpu']:
            row = {'backbone': name, 'device': device, 'precision': 'fp16' if device == 'cuda' else 'fp32'}
            model = None
            start = time.perf_counter()
            try:
                model = load_backbone(name, device, config)
                row['load_s'] = time.perf_counter()-start
                samples = timing.input_tensor(frame, window, name, config)
                result = extract_features(model, samples, name, device)
                row.update(status='ok', feature_dimensions={key: list(value.shape) for key, value in result.items()},
                           finite=all(torch.isfinite(value).all().item() for value in result.values()),
                           weights_sha256=sha256(model._phase0_checkpoint), weights_bytes=model._phase0_checkpoint.stat().st_size)
                expected = {'cls': 768, **{f'patch{stat}_L{n}': 768 for stat in ['mean', 'std'] for n in [6, 9, 12]}} if name == 'dino' else {'pooled': {'resnet50': 2048, 'slowfast_r50': 2304, 'r3d18': 512}[name]}
                if {key: value[-1] for key, value in row['feature_dimensions'].items()} != expected:
                    raise ValueError('unexpected feature dimensions')
            except Exception as exc:
                row.update(status='failed', failure_reason=f'{type(exc).__name__}: {exc}')
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(name, device, row['status'], row.get('failure_reason', ''), flush=True)
            checks.append(row); evidence['checks'].append(row)
        settings = config['backbones']['motion' if name in ('slowfast_r50', 'r3d18') else name]
        settings['confirmed'] = all(row['status'] == 'ok' for row in checks)
        if settings['confirmed']:
            if checks[0]['weights_sha256'] != checks[1]['weights_sha256']:
                settings['confirmed'] = False
                raise ValueError('device weight identities differ')
            settings['weights_sha256'] = checks[0]['weights_sha256']
            settings['weights_bytes'] = checks[0]['weights_bytes']
            settings['feature_dimensions'] = {key: value[-1] for key, value in checks[0]['feature_dimensions'].items()}
    if all(value['confirmed'] for value in config['backbones'].values()):
        evidence['status'] = 'passed'
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = importlib.import_module('scripts.00_inventory').prepare_sample_paths(config, 'konvid')
    iterator = iter_samples(paths[config['samples']['konvid'][0]], config)
    try:
        timestamp, frame, window = next(iterator)
    finally:
        iterator.close()
    evidence = check_models(config, frame, window)
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    evidence.update(video_id=config['samples']['konvid'][0], timestamp=timestamp,
                    hashes=importlib.import_module('scripts.00_timing').evidence_hashes(args.config, config))
    Path('results/phase0/backbones.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return 0 if evidence['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
