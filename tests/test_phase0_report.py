import copy
import importlib

import pytest


def fixture():
    config = {'samples': {'konvid': [str(i) for i in range(20)], 'livevqc_portrait': ['a', 'b', 'c']},
              'backbones': {k: {'confirmed': True, 'weights_sha256': 'weight', 'feature_dimensions': {'pooled': 2}} for k in ['dino', 'resnet50', 'motion']},
              'timing': {'repeats': 3, 'cpu_budget_seconds_per_10s': 100},
              'feasibility': {'deadline': '2026-10-31', 'weekly_hours': 10, 'available_compute_hours': 100, 'reserve_disk_bytes': 100},
              'ugc': {'status': 'skipped', 'cloud_status': 'deferred_phase3'}, 'decoder': {'confirmed': True}}
    config['backbones']['motion']['candidate'] = 'slowfast_r50'
    rows = [dict(video_id=key, backbone=name, device=device, repeat=repeat, status='ok', duration_s=10, T=40,
                 decode_s=1., preprocess_s=1., inference_s=1., total_s=5.)
            for key in config['samples']['konvid'] for name in ['dino', 'resnet50', 'slowfast_r50']
            for device in ['cuda', 'cpu'] for repeat in range(3)]
    inventory = {'konvid': {'video_count': 20, 'successful_probes': 20, 'duration_total_s': 200, 'timesteps': 800},
                 'livevqc': {'video_count': 10, 'successful_probes': 10, 'duration_total_s': 100, 'timesteps': 400},
                 'disk': {'free_bytes': 1_000_000, 'largest_temporary_video_bytes': 1000}}
    hashes = {'config_sha256': 'c', 'execution_config_sha256': 'e', 'lock_sha256': 'l', 'code_sha256': 'code', 'weights': {}}
    evidence = {'current_hashes': hashes, 'run': {'status': 'complete', 'hashes': hashes},
                'backbones': {'status': 'passed', 'hashes': hashes},
                'rotation': {'status': 'passed', 'records': [{'video_id': i, 'upright': True, 'reviewer': 'user'} for i in ['a', 'b', 'c']], 'preprocess_sha256': 'p'},
                'preprocess_sha256': 'p', 'active_labor_hours_estimate': 4., 'days_remaining': 6.}
    for name in ['konvid', 'livevqc']:
        inventory[name]['manifest_sha256'] = name+'-complete'
    evidence['manifest_sha256'] = {name: name+'-complete' for name in ['konvid', 'livevqc']}
    return config, inventory, rows, evidence


def report(*args):
    return importlib.import_module('scripts.00_report').build_report(*args)


def test_projection_from_measured_totals_and_twofold_allowance():
    result = report(*fixture())
    assert result['status'] == 'go'
    assert result['projections']['dino']['konvid_seconds'] == 100.
    assert result['projections']['dino']['livevqc_seconds'] == 50.
    assert result['projections']['dino']['combined_planning_seconds'] == 300.
    assert result['predicted_feature_bytes'] == 14_400


@pytest.mark.parametrize('field', ['deadline', 'weekly_hours', 'available_compute_hours', 'reserve_disk_bytes'])
def test_missing_user_constraint_blocks_completion(field):
    args = list(fixture()); args[0]['feasibility'][field] = None
    assert report(*args)['status'] == 'no-go'


@pytest.mark.parametrize('failure', ['cpu_budget', 'rotation', 'coverage', 'backbone', 'stale', 'disk', 'compute', 'labor'])
def test_missing_or_failed_gate_blocks_completion(failure):
    c, inv, rows, evidence = fixture()
    if failure == 'cpu_budget': c['timing']['cpu_budget_seconds_per_10s'] = None
    if failure == 'rotation': evidence['rotation'] = None
    if failure == 'coverage': rows.pop()
    if failure == 'backbone': c['backbones']['motion']['confirmed'] = False
    if failure == 'stale': evidence['current_hashes'] = {'changed': True}
    if failure == 'disk': inv['disk']['free_bytes'] = 1
    if failure == 'compute': c['feasibility']['available_compute_hours'] = .0001
    if failure == 'labor': c['feasibility']['weekly_hours'] = .01
    assert report(c, inv, rows, evidence)['status'] == 'no-go'


def test_active_ugc_requires_current_parity_and_smoke_but_deferred_cloud_does_not():
    c, inv, rows, evidence = fixture()
    c['ugc']['status'] = 'active'
    assert report(c, inv, rows, evidence)['status'] == 'no-go'


def test_partial_manifest_with_stale_success_summary_is_no_go():
    c, inv, rows, evidence = fixture()
    inv['konvid']['manifest_sha256'] = 'complete-konvid'
    inv['livevqc']['manifest_sha256'] = 'complete-live'
    evidence['manifest_sha256'] = {'konvid': 'partial-konvid', 'livevqc': 'complete-live'}
    inv['konvid']['timesteps'] = 1
    inv['livevqc']['timesteps'] = 0
    assert report(c, inv, rows, evidence)['status'] == 'no-go'


def test_decision_only_config_hash_change_keeps_existing_measurements():
    c, inv, rows, evidence = fixture()
    evidence['current_hashes'] = {**evidence['current_hashes'], 'config_sha256': 'budget-updated'}
    assert report(c, inv, rows, evidence)['status'] == 'go'
