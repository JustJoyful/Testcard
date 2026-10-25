import importlib
from pathlib import Path

import numpy as np
import torch


def timing():
    return importlib.import_module('scripts.00_timing')


def test_complete_coverage_keeps_failures_on_both_devices(monkeypatch):
    module = timing()
    ids = [str(i) for i in range(20)]
    config = {'samples': {'konvid': ids}, 'backbones': {'motion': {'candidate': 'slowfast_r50'}},
              'timing': {'warmup': 1, 'repeats': 3, 'cpu_threads': 1}}
    def unavailable(*args):
        raise RuntimeError('no pretrained weights')
    monkeypatch.setattr(module, 'load_backbone', unavailable)
    rows, loads = module.run_benchmark(config, {key: Path(key+'.mp4') for key in ids})
    assert len(rows) == 360
    assert {(r['video_id'], r['backbone'], r['device'], r['repeat']) for r in rows} == {
        (key, name, device, repeat) for key in ids for name in ['dino', 'resnet50', 'slowfast_r50']
        for device in ['cuda', 'cpu'] for repeat in range(3)}
    assert all(r['status'] == 'failed' and 'weights' in r['failure_reason'] for r in rows)
    assert not module.coverage_complete(config, rows)


def test_cpu_only_cannot_claim_cuda_rows(monkeypatch):
    module = timing()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def failure(*args):
        return torch.nn.Identity()
    monkeypatch.setattr(module, 'load_backbone', failure)
    config = {'samples': {'konvid': [str(i) for i in range(20)]}, 'backbones': {'motion': {'candidate': 'slowfast_r50'}},
              'timing': {'warmup': 1, 'repeats': 3, 'cpu_threads': 1}}
    rows, _ = module.run_benchmark(config, {key: Path(key) for key in config['samples']['konvid']})
    assert all(r['status'] == 'failed' for r in rows if r['device'] == 'cuda')
    assert all('CUDA' in r['failure_reason'] for r in rows if r['device'] == 'cuda')


def test_inference_synchronizes_gpu_and_excludes_load_warmup(monkeypatch):
    module = timing()
    events = []
    clock = iter([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(module, 'probe_video', lambda *a, **kw: {'duration_s': .25})
    monkeypatch.setattr(module, 'iter_samples', lambda *args: iter([(0., frame, None)]))
    monkeypatch.setattr(module, 'spatial_tensor', lambda x: torch.zeros(3, 448, 448))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: events.append('sync'))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 100)
    def extract(*args):
        events.append('inference')
        return {'pooled': torch.zeros(1, 2048)}
    monkeypatch.setattr(module, 'extract_features', extract)
    row = module.measure_video(Path('one.mp4'), torch.nn.Identity(), 'resnet50', 'cuda', {}, clock=lambda: next(clock))
    assert events == ['sync', 'inference', 'sync']
    assert row['inference_s'] == 1.
    assert row['T'] == 1
    assert row['total_s'] >= row['inference_s']


def test_execution_fingerprint_ignores_decisions_but_tracks_runtime_inputs(tmp_path, monkeypatch):
    import copy
    import yaml
    monkeypatch.chdir(tmp_path)
    (tmp_path/'env').mkdir(); (tmp_path/'env/requirements.lock').write_text('torch==2.14.1\n')
    config = {'seed': 0, 'samples': {'konvid': ['a']}, 'cache_version': 'v1',
              'datasets': {'konvid': {'archive': 'videos.zip'}}, 'preprocessing': {'spatial_crop': 448},
              'decoder': {'name': 'ffmpeg', 'confirmed': False},
              'backbones': {'dino': {'hub_commit': 'a'*40, 'weights_sha256': 'w'}},
              'timing': {'cpu_threads': 4, 'warmup': 1, 'repeats': 3, 'cpu_budget_seconds_per_10s': None},
              'feasibility': {'weekly_hours': None}, 'ugc': {'status': 'skipped'}}
    path = tmp_path/'config.yaml'
    def fingerprint(value):
        path.write_text(yaml.safe_dump(value))
        return timing().evidence_hashes(path, value)
    original = fingerprint(config)
    changed = copy.deepcopy(config)
    changed['timing']['cpu_budget_seconds_per_10s'] = 100
    changed['feasibility']['weekly_hours'] = 10
    changed['decoder']['confirmed'] = True
    updated = fingerprint(changed)
    assert original['config_sha256'] != updated['config_sha256']
    assert original['execution_config_sha256'] == updated['execution_config_sha256']
    for section, key, value in [('preprocessing', 'spatial_crop', 224), ('timing', 'cpu_threads', 8), ('samples', 'konvid', ['b'])]:
        altered = copy.deepcopy(config); altered[section][key] = value
        assert fingerprint(altered)['execution_config_sha256'] != original['execution_config_sha256']
    altered = copy.deepcopy(config); altered['backbones']['dino']['weights_sha256'] = 'other'
    assert fingerprint(altered)['execution_config_sha256'] != original['execution_config_sha256']
