import importlib
import json

import numpy as np
import pytest


def test_headline_uses_raw_rank_scores_and_target_mapped_plcc():
    report = importlib.import_module('scripts.11_phase3_report')
    y = np.linspace(0, 100, 30)
    by_seed = {seed: (y / 25 + np.sin(np.arange(30) + seed) * .2) for seed in range(5)}
    metrics, headline = report.summarize_live('resnet50', 'mean_pool', y, by_seed)
    assert len(metrics) == 5
    assert all(row['mapping_status'] == 'ok' and len(row['mapping_coefficients']) == 4 for row in metrics)
    assert headline['srcc_seed0'] == metrics[0]['srcc']
    assert headline['srcc_mean'] == pytest.approx(np.mean([row['srcc'] for row in metrics]))
    assert headline['srcc_sd'] == pytest.approx(np.std([row['srcc'] for row in metrics], ddof=1))
    assert 'plcc_ci95' not in headline
    with pytest.raises(ValueError):
        report.summarize_live('dino', 'mean_pool', y, {0: np.ones(30)})


def test_comparison_roster_has_only_six_primary_srcc_pairs():
    report = importlib.import_module('scripts.11_phase3_report')
    roster = report.comparison_roster()
    assert len(roster) == 15
    assert sum(row['family'] == 'primary' for row in roster) == 6
    assert sum(row['comparison_id'].startswith('ridge/') for row in roster) == 4
    assert sum(row['comparison_id'].startswith('feature/') for row in roster) == 5


def test_failed_report_keeps_failed_gate(tmp_path, monkeypatch):
    report = importlib.import_module('scripts.11_phase3_report')
    monkeypatch.setattr(report, 'load_phase3_context', lambda path: {'config': {'results_root': str(tmp_path)}})
    monkeypatch.setattr(report, 'require_prediction_inputs', lambda context: (_ for _ in ()).throw(ValueError('missing DINO')))
    with pytest.raises(ValueError, match='missing DINO'):
        report.phase3_report(tmp_path / 'phase3.yaml')
    assert json.loads((tmp_path / 'phase_gate.json').read_text())['status'] == 'failed'


def test_direct_report_invalidates_prior_pass_on_source_failure(tmp_path, monkeypatch):
    import yaml
    report = importlib.import_module('scripts.11_phase3_report')
    config = tmp_path / 'phase3.yaml'
    config.write_text(yaml.safe_dump({'results_root': str(tmp_path)}))
    (tmp_path / 'phase_gate.json').write_text('{"status":"passed","outputs_sha256":{}}')
    monkeypatch.setattr(report, 'load_phase3_context', lambda path: (_ for _ in ()).throw(ValueError('stale Phase 2')))
    with pytest.raises(ValueError, match='stale Phase 2'):
        report.phase3_report(config)
    assert json.loads((tmp_path / 'phase_gate.json').read_text())['status'] == 'failed'
