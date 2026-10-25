import importlib

import pytest
import torch


def features():
    return importlib.import_module('src.features')


class DinoDouble(torch.nn.Module):
    num_register_tokens = 0

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def get_intermediate_layers(self, x, n, return_class_token, norm):
        assert n == [5, 8, 11] and norm and return_class_token
        assert not self.training and not torch.is_grad_enabled()
        patches = torch.tensor([[[1., 3.], [3., 7.]]]) * self.scale
        cls = torch.tensor([[99., 101.]]) * self.scale
        return [(patches, cls)] * 3


def test_loader_freezes_eval_and_dino_statistics_exclude_cls(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, 'load', lambda *args, **kwargs: DinoDouble())
    model = features().load_backbone('dino', 'cpu', {'backbones': {'dino': {'hub_commit': 'a'*40}}, 'weights_dir': str(tmp_path)})
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    result = features().extract_features(model, torch.zeros(1, 3, 2, 2), 'dino', 'cpu')
    torch.testing.assert_close(result['cls'], torch.tensor([[99., 101.]]))
    torch.testing.assert_close(result['patchmean_L6'], torch.tensor([[2., 5.]]))
    torch.testing.assert_close(result['patchstd_L12'], torch.tensor([[1., 2.]]))
    assert len(result) == 7
    assert all(v.dtype == torch.float32 and not v.requires_grad for v in result.values())


def test_missing_weights_never_substitute_random_model(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError('weights unavailable')
    monkeypatch.setattr(torch.hub, 'load', fail)
    with pytest.raises(RuntimeError, match='weights unavailable'):
        features().load_backbone('dino', 'cpu', {'backbones': {'dino': {'hub_commit': 'a'*40}}, 'weights_dir': str(tmp_path)})


def test_unpinned_hub_rejected_before_loading():
    with pytest.raises(ValueError, match='commit'):
        features().load_backbone('dino', 'cpu', {'backbones': {'dino': {'hub_commit': 'main'}}})


def test_cpu_output_is_finite_detached_fp32_and_bad_features_fail():
    model = torch.nn.Linear(2, 3).eval().requires_grad_(False)
    result = features().extract_features(model, torch.ones(1, 2, dtype=torch.float64), 'resnet50', 'cpu')
    assert result['pooled'].shape == (1, 3)
    assert result['pooled'].dtype == torch.float32
    assert not result['pooled'].requires_grad
    with pytest.raises(ValueError, match='nonfinite'):
        features().extract_features(model, torch.full((1, 2), float('nan')), 'resnet50', 'cpu')


def test_cuda_unavailable_is_identified(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        features().load_backbone('resnet50', 'cuda', {})


def test_backbone_smoke_never_confirms_missing_cuda(monkeypatch, tmp_path):
    import yaml
    module = importlib.import_module('scripts.00_backbone_check')
    monkeypatch.setattr(module, 'load_backbone', lambda *a: (_ for _ in ()).throw(RuntimeError('CUDA unavailable')))
    config = {'backbones': {'dino': {'confirmed': False}, 'resnet50': {'confirmed': False}, 'motion': {'candidate': 'slowfast_r50', 'confirmed': False}}, 'timing': {'cpu_threads': 1}}
    frame = __import__('numpy').zeros((2, 2, 3), dtype='uint8')
    evidence = module.check_models(config, frame, __import__('numpy').stack([frame]*32))
    assert evidence['status'] == 'failed'
    assert len(evidence['checks']) == 6
    assert all(r['status'] == 'failed' for r in evidence['checks'])
    assert not any(s['confirmed'] for s in config['backbones'].values())
