import copy
import json

import pytest


def fixture():
    candidates = [{'candidate_id': f'f/h{i}', 'headline_cv_srcc': 0.7 + i / 100,
                   'feature_set': 'f', 'head': f'h{i}'} for i in range(10)]
    ids = [f'k{i}' for i in range(20)]
    rows = [{'candidate_id': c['candidate_id'], 'video_id': key, 'dataset': 'konvid',
             'repeat': repeat, 'duration_s': 10., 'decode_seconds': 1.,
             'preprocess_seconds': 1., 'backbone_seconds': 1., 'head_seconds': 1.,
             'total_seconds': 4., 'normalized_total_seconds': 4.,
             'source_sha256': 'a' * 64, 'config_sha256': 'b' * 64,
             'checkpoint_sha256': 'c' * 64}
            for c in candidates for key in ids for repeat in range(3)]
    return {'deployment': {'candidates': candidates, 'cpu_budget_seconds_per_10s': 120}}, {
        'cpu_threads': 4, 'sample_ids': ids, 'repeats': 3, 'rows': rows}


def test_cpu_winner_requires_complete_measured_grid():
    from src.phase3 import resolve_cpu_winner
    freeze, evidence = fixture()
    assert resolve_cpu_winner(freeze, evidence)['candidate_id'] == 'f/h9'
    for edit in [lambda e: e['rows'].pop(),
                 lambda e: e['rows'][0].pop('decode_seconds'),
                 lambda e: e['rows'][0].update(dataset='livevqc'),
                 lambda e: e['rows'][0].update(duration_s=-1),
                 lambda e: e['rows'][0].update(head_seconds=float('nan')),
                 lambda e: e['rows'][0].pop('checkpoint_sha256'),
                 lambda e: e.update(cpu_threads=2)]:
        changed = copy.deepcopy(evidence)
        edit(changed)
        with pytest.raises(ValueError):
            resolve_cpu_winner(freeze, changed)


def test_cpu_score_then_latency_then_id_and_budget_boundary():
    from src.phase3 import resolve_cpu_winner
    freeze, evidence = fixture()
    for row in evidence['rows']:
        if row['candidate_id'] == 'f/h9':
            row['head_seconds'] = 117.
            row['total_seconds'] = row['normalized_total_seconds'] = 120.
    assert resolve_cpu_winner(freeze, evidence)['candidate_id'] == 'f/h9'
    for row in evidence['rows']:
        if row['candidate_id'] == 'f/h9':
            row['head_seconds'] = 118.
            row['total_seconds'] = row['normalized_total_seconds'] = 121.
    assert resolve_cpu_winner(freeze, evidence)['candidate_id'] == 'f/h8'
    freeze['deployment']['candidates'][7]['headline_cv_srcc'] = freeze['deployment']['candidates'][8]['headline_cv_srcc']
    for row in evidence['rows']:
        if row['candidate_id'] == 'f/h8':
            row['head_seconds'] = 2.
            row['total_seconds'] = row['normalized_total_seconds'] = 5.
    assert resolve_cpu_winner(freeze, evidence)['candidate_id'] == 'f/h7'
    for row in evidence['rows']:
        row['head_seconds'] = 118.
        row['total_seconds'] = row['normalized_total_seconds'] = 121.
    with pytest.raises(ValueError):
        resolve_cpu_winner(freeze, evidence)


def test_saved_cpu_duration_must_match_konvid_source(tmp_path, monkeypatch):
    from src.phase3 import require_cpu_winner, resolve_cpu_winner
    from src.training import config_hash
    import src.phase3 as phase3
    freeze, evidence = fixture()
    for candidate in freeze['deployment']['candidates']:
        candidate['config'] = {'candidate': candidate['candidate_id']}
        candidate['refit'] = {'cv_epochs': [{'run_path': 'unused/run.json',
                                            'run_sha256': 'c' * 64,
                                            'checkpoint_sha256': 'c' * 64}]}
    for row in evidence['rows']:
        row['config_sha256'] = config_hash(next(c['config'] for c in freeze['deployment']['candidates']
                                                if c['candidate_id'] == row['candidate_id']))
    evidence.update(phase2_gate_sha256='g' * 64, freeze_sha256='f' * 64, code_sha256='d' * 64)
    sample = tmp_path / 'ids.txt'
    sample.write_text('\n'.join(evidence['sample_ids']) + '\n')
    result = tmp_path / 'cpu_eligibility.json'
    context = {'config': {'results_root': str(tmp_path), 'cpu_sample_ids': str(sample)},
               'phase2_gate_sha256': 'g' * 64, 'freeze_sha256': 'f' * 64,
               'code_sha256': 'd' * 64, 'freeze': freeze,
               'candidates': freeze['deployment']['candidates'],
               'inventories': {'konvid': [{'video_id': key, 'source_sha256': 'a' * 64,
                                          'duration_s': '10'} for key in evidence['sample_ids']]}}
    monkeypatch.setattr(phase3, 'sha256', lambda path: 'c' * 64)
    evidence['winner'] = resolve_cpu_winner(freeze, evidence)
    result.write_text(json.dumps(evidence))
    require_cpu_winner(context)
    evidence['rows'][0]['duration_s'] = 1000.
    evidence['rows'][0]['normalized_total_seconds'] = .04
    evidence['winner'] = resolve_cpu_winner(freeze, evidence)
    result.write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        require_cpu_winner(context)
