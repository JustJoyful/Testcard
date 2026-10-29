import copy

import numpy as np
import pytest
import torch


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, features, lengths):
        return self.layer(features.mean(1)).squeeze(-1)


def tiny_config():
    return {'device': 'cpu', 'cpu_threads': 1, 'learning_rate': .01,
            'weight_decay': .001, 'batch_size': 2, 'max_epochs': 200,
            'warmup_epochs': 5, 'rank_weight': 0., 'gradient_clip_norm': 1.,
            'patience': 15, 'feature_set': 'resnet50'}


def tiny_data():
    return [{'video_id': f'k{i}', 'features': torch.tensor([[float(i), float(i*i)]], dtype=torch.float32),
             'mos': float(i + 1)} for i in range(4)]


def test_fixed_epoch_fit_uses_final_epoch_and_resume_identity(tmp_path):
    from src.training import fit_fixed_epochs
    model = TinyHead()
    data = tiny_data()
    config = tiny_config()
    output = tmp_path / 'run'
    result = fit_fixed_epochs(model, data, config, output, 0, {'fold': None}, 2)
    assert result['status'] == 'complete'
    assert result['completed_epochs'] == result['best_epoch'] == 2
    assert result['selection'] == 'fixed_final_epoch'
    assert result['target_stats']['mean'] == 2.5
    assert result['history'][0]['learning_rate'] == pytest.approx(.002)
    assert result['history'][1]['learning_rate'] == pytest.approx(.004)
    assert fit_fixed_epochs(TinyHead(), data, config, output, 0, {'fold': None}, 2) == result
    with pytest.raises(ValueError):
        fit_fixed_epochs(TinyHead(), data, config, output, 0, {'fold': None}, 3)
    changed = copy.deepcopy(config)
    changed['learning_rate'] = .02
    with pytest.raises(ValueError):
        fit_fixed_epochs(TinyHead(), data, changed, output, 0, {'fold': None}, 2)


def test_fixed_first_epoch_matches_cv_update(tmp_path):
    from src.training import fit_fixed_epochs, fit_one
    torch.manual_seed(7)
    cv_model = TinyHead()
    fixed_model = TinyHead()
    fixed_model.load_state_dict(copy.deepcopy(cv_model.state_dict()))
    config = {**tiny_config(), 'max_epochs': 1}
    data = tiny_data()
    fit_one(cv_model, data, data, config, tmp_path / 'cv', 0, {'fold': 0})
    fit_fixed_epochs(fixed_model, data, config, tmp_path / 'fixed', 0, {'fold': None}, 1)
    for name, value in cv_model.state_dict().items():
        torch.testing.assert_close(value, fixed_model.state_dict()[name], rtol=0, atol=0)


def test_run_refits_uses_each_frozen_epoch_for_all_forty_neural_runs(tmp_path, monkeypatch):
    import importlib
    script = importlib.import_module('scripts.09_phase3_refit')
    calls = []
    monkeypatch.setattr(script, 'InventoryFeatureDataset', lambda *args: tiny_data())
    monkeypatch.setattr(script, 'build_head', lambda *args: TinyHead())
    monkeypatch.setattr(script, '_validate_refit_sources', lambda spec: ['a' * 64] * 25)
    def record(model, data, cfg, out, seed, provenance, epochs, **kw):
        calls.append((cfg['feature_set'], provenance['head'], seed, epochs, len(data)))
        return {'status': 'complete'}
    monkeypatch.setattr(script, 'fit_fixed_epochs', record)
    freeze = {'refit_by_feature': {feature: {head: {'config': {**tiny_config(), 'feature_set': feature},
                            'median_best_epoch': n, 'cv_epochs': [{'run_sha256': 'a' * 64}] * 25}
                            for n, head in enumerate(('mean_pool', 'bigru', 'transformer', 'mamba'), 2)}
                            for feature in ('resnet50', 'dino')}, 'seeds': list(range(5))}
    context = {'freeze': freeze, 'freeze_sha256': 'b' * 64,
               'phase2_gate_sha256': 'c' * 64, 'provenance': {},
               'config': {'cache_root': tmp_path, 'konvid_inventory': tmp_path / 'konvid.csv',
                          'results_root': tmp_path / 'results'}}
    (tmp_path / 'konvid.csv').write_text('video_id\nk0\nk1\nk2\nk3\n')
    script.run_refits(context, neural_only=True)
    assert len(calls) == 40
    assert {item[3] for item in calls if item[1] == 'mamba'} == {5}
    assert {item[0] for item in calls} == {'dino', 'resnet50'}


def test_fixed_alpha_matches_single_alpha_cv_solver():
    from src.baselines import fit_ridge, fit_ridge_fixed_alpha
    x = np.array([[0., 1.], [1., 2.], [2., 2.], [3., 4.]])
    y = np.array([1., 2., 4., 3.])
    fixed = fit_ridge_fixed_alpha(x, y, 10.)
    cv = fit_ridge(x, y, x, y, [10.])
    for key in ('coefficient', 'feature_mean', 'feature_scale', 'target_mean', 'target_std', 'alpha'):
        np.testing.assert_allclose(fixed[key], cv[key])


def test_prediction_accepts_feature_only_items():
    from src.training import predict
    model = TinyHead()
    items = [{k: v for k, v in item.items() if k != 'mos'} for item in tiny_data()]
    assert len(predict(model, items, {'mean': 2., 'std': 1.}, 'cpu')) == len(items)


def test_ridge_refits_use_separate_frozen_alphas_and_reject_tampering(tmp_path, monkeypatch):
    import importlib
    import json
    script = importlib.import_module('scripts.09_phase3_refit')
    monkeypatch.setattr(script, 'InventoryFeatureDataset', lambda *args: tiny_data())
    monkeypatch.setattr(script, '_validate_refit_sources', lambda spec: ['a' * 64] * 5)
    specs = {'resnet50': {'ridge': {'phase3_alpha': 1000., 'cv_selected_alphas': [{}] * 5}},
             'dino': {'ridge': {'phase3_alpha': 10., 'cv_selected_alphas': [{}] * 5}}}
    candidates = [{'candidate_id': f'{feature}/ridge', 'feature_set': feature, 'head': 'ridge',
                   'config': tiny_config()} for feature in specs]
    context = {'freeze': {'refit_by_feature': specs, 'deployment': {'candidates': candidates}},
               'freeze_sha256': 'b' * 64, 'phase2_gate_sha256': 'c' * 64, 'provenance': {},
               'config': {'cache_root': tmp_path, 'konvid_inventory': tmp_path / 'konvid.csv',
                          'results_root': tmp_path / 'results'}}
    (tmp_path / 'konvid.csv').write_text('video_id\nk0\nk1\nk2\nk3\n')
    script.run_ridge_refits(context)
    root = tmp_path / 'results' / 'refit'
    assert json.loads((root / 'resnet50/ridge/seedNone/run.json').read_text())['alpha'] == 1000.
    assert json.loads((root / 'dino/ridge/seedNone/run.json').read_text())['alpha'] == 10.
    script.run_ridge_refits(context)
    (root / 'dino/ridge/seedNone/model.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        script.run_ridge_refits(context)
