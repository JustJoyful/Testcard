import importlib
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('backbone,branches', [('dino',7), ('resnet50',1)])
def test_spatial_extraction_caches_all_seven_dino_branches(backbone, branches, monkeypatch):
    script = importlib.import_module('scripts.02_extract_spatial')
    frame = np.ones((2,2,3), np.uint8)
    monkeypatch.setattr(script, 'iter_samples', lambda p,c: iter([(0,frame,None),(0.25,frame,None)]))
    monkeypatch.setattr(script, 'spatial_tensor', lambda f: torch.ones(3,2,2))
    def features(m, x, name, device):
        assert x.shape == (1,3,2,2)
        return {k: torch.ones(1,768 if name == 'dino' else 2048) for k in
                (['cls','patchmean_L6','patchstd_L6','patchmean_L9','patchstd_L9','patchmean_L12','patchstd_L12'] if name=='dino' else ['pooled'])}
    monkeypatch.setattr(script, 'extract_features', features)
    actual = script.extract_video(None, {}, None, backbone, 'cpu')
    assert len(actual) == branches and all(x.shape[0] == 2 and x.dtype == np.float16 for x in actual.values())
