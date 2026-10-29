import copy

import pytest


def test_live_rows_require_exact_order_identity_and_finite_values():
    from src.phase3 import validate_live_rows
    context = {'ids': {'livevqc': ['l1', 'l2']}}
    run = {'checkpoint_sha256': 'a' * 64, 'identity': {'seed': 0}}
    rows = [{'video_id': key, 'dataset': 'livevqc', 'feature_set': 'dino', 'head': 'mean_pool',
             'seed': 0, 'prediction': pred, 'checkpoint_sha256': 'a' * 64,
             'run_identity': run['identity']} for key, pred in [('l1', 1.), ('l2', 2.)]]
    validate_live_rows(context, rows, 'dino', 'mean_pool', 0, run)
    for changed in [rows[:1], rows + [rows[0]], list(reversed(rows))]:
        with pytest.raises(ValueError):
            validate_live_rows(context, changed, 'dino', 'mean_pool', 0, run)
    for field, value in [('prediction', float('nan')), ('mos', 11.), ('feature_set', 'resnet50'),
                         ('checkpoint_sha256', 'b' * 64)]:
        changed = copy.deepcopy(rows)
        changed[0][field] = value
        with pytest.raises(ValueError):
            validate_live_rows(context, changed, 'dino', 'mean_pool', 0, run)


def test_prediction_checks_gate_before_dataset_or_inference(monkeypatch):
    import importlib
    script = importlib.import_module('scripts.10_phase3_predict')
    monkeypatch.setattr(script, 'require_prediction_inputs', lambda context: (_ for _ in ()).throw(ValueError('blocked')))
    monkeypatch.setattr(script, 'InventoryFeatureDataset', lambda *args: (_ for _ in ()).throw(AssertionError('inference started')))
    with pytest.raises(ValueError, match='blocked'):
        script.predict_phase3({'config': {}})
