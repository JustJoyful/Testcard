import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

import pytest

from src.cache import atomic_json
from src.features import sha256


def test_fixed_control_rejects_nonfrozen_epoch(tmp_path, monkeypatch):
    from src.phase3 import require_fixed_control
    import src.metrics as metrics
    import src.tuning as tuning
    monkeypatch.setattr(tuning, 'require_control', lambda *args: None)
    score = {'status': 'ok', 'srcc': .01, 'ci95': [-.1, .1]}
    monkeypatch.setattr(metrics, 'bootstrap_srcc', lambda *args: score)
    split = tmp_path / 'splits.csv'
    split.write_text('video_id,outer_fold,role\nk1,0,test\nk2,0,test\n')
    root = tmp_path / 'control_fixed_epoch'
    root.mkdir()
    (root / 'permutations.json').write_text('{}')
    items = []
    for seed in range(5):
        directory = root / f'seed{seed}'
        directory.mkdir()
        (directory / 'best.pt').write_bytes(b'checkpoint')
        run = {'selection': 'fixed_final_epoch', 'seed': seed, 'completed_epochs': 3, 'best_epoch': 2,
               'checkpoint_sha256': sha256(directory / 'best.pt')}
        atomic_json(directory / 'run.json', run)
        atomic_json(directory / 'predictions.json', [
            {'video_id': key, 'mos': mos, 'prediction': pred,
             'checkpoint_sha256': run['checkpoint_sha256']}
            for key, mos, pred in [('k1', 10., 1.), ('k2', 90., 2.)]])
        items.append({'seed': seed, 'status': 'passed', 'bootstrap': score,
                      'run_sha256': sha256(directory / 'run.json'),
                      'checkpoint_sha256': run['checkpoint_sha256'],
                      'predictions_sha256': sha256(directory / 'predictions.json')})
    atomic_json(root / 'diagnostic.json', {'status': 'passed', 'feature_set': 'resnet50',
        'freeze_sha256': 'f', 'phase2_gate_sha256': 'g', 'code_sha256': 'd',
        'permutations_sha256': sha256(root / 'permutations.json'), 'seeds': items})
    context = {'config': {'results_root': str(tmp_path)}, 'selected_feature_set': 'resnet50',
               'freeze_sha256': 'f', 'phase2_gate_sha256': 'g', 'code_sha256': 'd',
               'provenance': {}, 'freeze': {'configs': {'mean_pool': {'splits': str(split)}},
               'refit_by_feature': {'resnet50': {'mean_pool': {'median_best_epoch': 2}}}},
               'inventories': {'konvid': [{'video_id': 'k1', 'mos': '10'},
                                          {'video_id': 'k2', 'mos': '90'}]}}
    with pytest.raises(ValueError):
        require_fixed_control(context)
    for seed, item in enumerate(items):
        path = root / f'seed{seed}' / 'run.json'
        run = json.loads(path.read_text())
        run['completed_epochs'] = 2
        atomic_json(path, run)
        item['run_sha256'] = sha256(path)
    atomic_json(root / 'diagnostic.json', {**json.loads((root / 'diagnostic.json').read_text()), 'seeds': items})
    assert require_fixed_control(context)['status'] == 'passed'


def test_qualified_control_preserves_failed_evidence_and_decision(tmp_path, monkeypatch):
    from src.phase3 import require_qualified_control
    import src.metrics as metrics
    import src.tuning as tuning

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tuning, 'require_control', lambda *args: None)
    score = lambda p: {'status': 'ok', 'srcc': -.2 if p[0] in (4., 5.) else 0.,
                       'ci95': [-.3, -.1] if p[0] in (4., 5.) else [-.1, .1]}
    monkeypatch.setattr(metrics, 'bootstrap_srcc', lambda y, p: score(p))
    decision = tmp_path / 'DECISIONS.md'
    decision.write_text('## 2026-10-28: Phase 3 qualified continuation after fixed-epoch control failure\n')
    split = tmp_path / 'splits.csv'
    split.write_text('video_id,outer_fold,role\nt0,0,train\nt1,0,train\nv0,0,val\nv1,0,val\nk0,0,test\nk1,0,test\nk2,0,test\n')
    root = tmp_path / 'control_fixed_epoch'
    root.mkdir()
    permutations = {}
    for role, ids in [('train', ['t0', 't1']), ('val', ['v0', 'v1'])]:
        order = np.random.default_rng(0).permutation(len(ids))
        permutations[role] = [{'video_id': key, 'label_source_id': ids[int(j)]}
                              for key, j in zip(ids, order)]
    atomic_json(root / 'permutations.json', permutations)
    config = {'splits': str(split)}
    context = {'config': {'results_root': str(tmp_path)}, 'selected_feature_set': 'resnet50',
               'freeze_sha256': 'f', 'phase2_gate_sha256': 'g', 'provenance': {},
               'freeze': {'refit_by_feature': {'resnet50': {'mean_pool':
                          {'config': config, 'median_best_epoch': 3}}}},
               'inventories': {'konvid': [{'video_id': key, 'mos': str(mos)}
                                for key, mos in [('k0', 1), ('k1', 2), ('k2', 3)]]}}
    items, details, initial_seeds, trained, untrained = [], [], [], [], []
    for seed in range(5):
        directory = root / f'seed{seed}'
        directory.mkdir()
        (directory / 'best.pt').write_bytes(f'checkpoint {seed}'.encode())
        run = {'status': 'complete', 'config': config, 'selection': 'fixed_final_epoch',
               'seed': seed, 'completed_epochs': 3, 'best_epoch': 3,
               'checkpoint_sha256': sha256(directory / 'best.pt'),
               'provenance': {'code_sha256': 'old', 'label_mode': 'shuffled',
                              'control_permutations_sha256': sha256(root / 'permutations.json')}}
        atomic_json(directory / 'run.json', run)
        p = np.array([float(seed + 1), float(seed + 2), float(seed + 4)])
        q = np.array([float(seed + 10), float(seed + 13), float(seed + 12)])
        y = [1., 2., 3.]
        rows = [{'video_id': f'k{i}', 'mos': y[i], 'prediction': float(p[i])} for i in range(3)]
        initial = [{'video_id': f'k{i}', 'mos': y[i], 'prediction': float(q[i])} for i in range(3)]
        atomic_json(directory / 'predictions.json', rows)
        item = {'seed': seed, 'status': 'failed' if seed in (3, 4) else 'passed',
                'bootstrap': score(p), 'run_sha256': sha256(directory / 'run.json'),
                'checkpoint_sha256': run['checkpoint_sha256'],
                'predictions_sha256': sha256(directory / 'predictions.json')}
        items.append(item)
        details.append({'trained': score(p), 'untrained': score(q),
                        'status_under_unchanged_rule': item['status'],
                        'trained_vs_untrained_pearson_r': float(pearsonr(p, q).statistic),
                        'trained_vs_untrained_spearman_r': float(spearmanr(p, q).statistic)})
        initial_seeds.append({'rows': initial, 'bootstrap': score(q)})
        trained.append(p)
        untrained.append(q)
    atomic_json(root / 'diagnostic.json', {'status': 'failed', 'seeds': items[:4],
                'freeze_sha256': 'f', 'phase2_gate_sha256': 'g', 'code_sha256': 'old'})
    atomic_json(root / 'seed4_result.json', items[4])
    atomic_json(root / 'untrained_baseline.json', {'seeds': initial_seeds,
                'code_sha256': 'old', 'freeze_sha256': 'f'})
    atomic_json(root / 'investigation.json', {'original_control_status': 'failed',
                'original_diagnostic_sha256': sha256(root / 'diagnostic.json'),
                'seed4_result_sha256': sha256(root / 'seed4_result.json'),
                'untrained_baseline_sha256': sha256(root / 'untrained_baseline.json'),
                'permutations_sha256': sha256(root / 'permutations.json'),
                'code_sha256_at_control': 'old', 'seeds': details,
                'phase1_phase2_comparable_trained_ensemble': score(np.mean(trained, axis=0)),
                'untrained_ensemble': score(np.mean(untrained, axis=0))})
    atomic_json(root / 'qualification.json', {'status': 'approved_qualified_continuation',
                'original_diagnostic_sha256': sha256(root / 'diagnostic.json'),
                'investigation_sha256': sha256(root / 'investigation.json'),
                'decision_sha256': sha256(decision)})
    assert require_qualified_control(context)['status'] == 'passed with qualification'
    original_decision = decision.read_text()
    decision.write_text(original_decision + 'changed\n')
    with pytest.raises(ValueError, match='decision'):
        require_qualified_control(context)
    decision.write_text(original_decision)
    predictions = root / 'seed3' / 'predictions.json'
    original_rows = predictions.read_text()
    predictions.write_text(original_rows.replace('4.0', '40.0', 1))
    with pytest.raises(ValueError, match='seed 3'):
        require_qualified_control(context)
    predictions.write_text(original_rows)
    permutation_path = root / 'permutations.json'
    original_permutations = permutation_path.read_text()
    altered_permutations = json.loads(original_permutations)
    altered_permutations['train'][0]['label_source_id'] = 'v0'
    atomic_json(permutation_path, altered_permutations)
    investigation = json.loads((root / 'investigation.json').read_text())
    investigation['permutations_sha256'] = sha256(permutation_path)
    atomic_json(root / 'investigation.json', investigation)
    qualification = json.loads((root / 'qualification.json').read_text())
    qualification['investigation_sha256'] = sha256(root / 'investigation.json')
    atomic_json(root / 'qualification.json', qualification)
    with pytest.raises(ValueError, match='train control permutation'):
        require_qualified_control(context)
    permutation_path.write_text(original_permutations)
    investigation['permutations_sha256'] = sha256(permutation_path)
    atomic_json(root / 'investigation.json', investigation)
    qualification['investigation_sha256'] = sha256(root / 'investigation.json')
    atomic_json(root / 'qualification.json', qualification)
    altered = json.loads((root / 'diagnostic.json').read_text())
    altered['seeds'][3]['status'] = 'passed'
    atomic_json(root / 'diagnostic.json', altered)
    investigation = json.loads((root / 'investigation.json').read_text())
    investigation['original_diagnostic_sha256'] = sha256(root / 'diagnostic.json')
    atomic_json(root / 'investigation.json', investigation)
    qualification = json.loads((root / 'qualification.json').read_text())
    qualification['original_diagnostic_sha256'] = sha256(root / 'diagnostic.json')
    qualification['investigation_sha256'] = sha256(root / 'investigation.json')
    atomic_json(root / 'qualification.json', qualification)
    with pytest.raises(ValueError, match='decision or failure evidence'):
        require_qualified_control(context)
