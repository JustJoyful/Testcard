"""Generate measurement-derived Phase 0 gates; exit nonzero for missing evidence."""
import argparse
import csv
import datetime
import importlib
import json
import math
import statistics
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import sha256


def build_report(config: dict, inventory: dict, timings: list[dict], evidence: dict) -> dict:
    timing = importlib.import_module('scripts.00_timing')
    missing = []
    def gate(name, passed):
        if not passed:
            missing.append(name)
    limits = config['feasibility']
    def positive(value):
        return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    for key in ['deadline', 'weekly_hours', 'available_compute_hours', 'reserve_disk_bytes']:
        gate(key, bool(limits.get(key)) if key == 'deadline' else positive(limits.get(key)))
    gate('decoder_confirmation', config['decoder'].get('confirmed') is True)
    gate('complete_20_video_timings', timing.coverage_complete(config, timings))
    gate('pretrained_backbones', all(v.get('confirmed') and v.get('weights_sha256') for v in config['backbones'].values()) and evidence.get('backbones', {}).get('status') == 'passed')
    current = evidence.get('current_hashes')
    gate('current_timing_hashes', bool(current) and timing.hashes_match(evidence.get('run', {}).get('hashes'), current) and evidence.get('run', {}).get('status') == 'complete')
    gate('current_backbone_hashes', bool(current) and timing.hashes_match(evidence.get('backbones', {}).get('hashes'), current))
    rotation = evidence.get('rotation') or {}
    reviewed = {r['video_id'] for r in rotation.get('records', []) if r.get('upright') is True and r.get('reviewer') == 'user'}
    gate('human_portrait_review', rotation.get('status') == 'passed' and len(reviewed) >= 3 and set(config['samples']['livevqc_portrait']) <= reviewed and rotation.get('preprocess_sha256') == evidence.get('preprocess_sha256'))
    for name in ['konvid', 'livevqc']:
        data = inventory.get(name, {})
        gate(name+'_inventory', data.get('video_count', 0) > 0 and data.get('video_count') == data.get('successful_probes'))
        gate(name+'_manifest_integrity', bool(data.get('manifest_sha256')) and data.get('manifest_sha256') == evidence.get('manifest_sha256', {}).get(name) and data.get('timesteps', 0) >= data.get('video_count', 1))
    projections = {}
    for name in ['dino', 'resnet50', config['backbones']['motion']['candidate']]:
        rows = [r for r in timings if r['backbone'] == name and r['device'] == 'cuda' and r['status'] == 'ok']
        if not rows:
            continue
        ratios = [float(r['total_s'])/float(r['duration_s']) for r in rows if float(r['duration_s']) > 0]
        if not ratios:
            continue
        rate = statistics.mean(ratios)
        konvid = rate*inventory.get('konvid', {}).get('duration_total_s', 0)
        live = rate*inventory.get('livevqc', {}).get('duration_total_s', 0)
        projections[name] = {'seconds_per_decoded_second': rate, 'repeat_video_ratio_std': statistics.pstdev(ratios),
                             'konvid_seconds': konvid, 'livevqc_seconds': live,
                             'combined_planning_seconds': 2*(konvid+live),
                             'note': 'LIVE-VQC extrapolated from KoNViD; includes 2x planning allowance; training unmeasured.'}
    projected_hours = sum(v['combined_planning_seconds'] for v in projections.values())/3600 if len(projections) == 3 else None
    gate('compute_capacity', projected_hours is not None and positive(limits.get('available_compute_hours')) and projected_hours <= limits['available_compute_hours'])
    labor = evidence.get('active_labor_hours_estimate')
    days = evidence.get('days_remaining')
    gate('labor_capacity', positive(labor) and positive(days) and positive(limits.get('weekly_hours')) and labor <= days/7*limits['weekly_hours'])
    cpu_rows = [r for r in timings if r['device'] == 'cpu' and r['backbone'] == 'dino' and r['status'] == 'ok' and float(r['duration_s']) > 0]
    cpu_estimate = max((10*float(r['total_s'])/float(r['duration_s']) for r in cpu_rows), default=None)
    budget = config['timing'].get('cpu_budget_seconds_per_10s')
    gate('accepted_cpu_budget', positive(budget) and cpu_estimate is not None and 1.25*cpu_estimate <= budget)
    dimension = sum(sum(v.get('feature_dimensions', {}).values()) for v in config['backbones'].values())
    steps = sum(inventory.get(name, {}).get('timesteps', 0) for name in ['konvid', 'livevqc'])
    feature_bytes = dimension*steps*2 if dimension and steps else None
    disk = inventory.get('disk', {})
    reserve = limits.get('reserve_disk_bytes')
    required = 2*feature_bytes + disk.get('largest_temporary_video_bytes', 0) + (reserve or 0) if feature_bytes else None
    gate('disk_capacity', required is not None and positive(reserve) and disk.get('free_bytes', 0) >= required)
    ugc = config['ugc']
    if ugc['status'] == 'active':
        cloud = evidence.get('cloud', {})
        gate('ugc_parity_and_smoke', cloud.get('status') == 'passed' and timing.hashes_match(cloud.get('hashes'), current) and cloud.get('smoke') == 'passed')
    else:
        gate('ugc_decision', ugc['status'] in ('skipped', 'metadata_only'))
        if ugc['status'] == 'metadata_only':
            gate('ugc_subset_locked', evidence.get('ugc_subset_verified') is True)
    return {'status': 'no-go' if missing else 'go', 'missing_or_failed_gates': missing,
            'projections': projections, 'projected_gpu_planning_hours': projected_hours,
            'predicted_feature_bytes': feature_bytes, 'additional_disk_required_with_reserve_bytes': required,
            'disk_accounting': 'Current free bytes already exclude retained archives, installed environments, weights and retained sample clips; additional capacity includes two full feature copies for atomic-write headroom, a transient raw video and reserve.',
            'cpu_dino_seconds_per_10s_worst_sample': cpu_estimate,
            'cpu_budget_note': 'DINO-only extraction estimate; head inference and API overhead unmeasured; 25% extraction headroom required.',
            'evidence_hashes': current, 'preprocessing_frozen': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    directory = Path('results/phase0')
    def read_json(name):
        path = directory/name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            return {'status': 'invalid_json', 'failure_reason': str(exc)}
    inventory = read_json('inventory.json')
    # Derived sample counts, never fitted to MOS or predictions.
    for name in ['konvid', 'livevqc']:
        path = Path(f'data/manifests/{name}.csv')
        if path.exists() and name in inventory and sha256(path) == inventory[name].get('manifest_sha256'):
            with path.open() as f:
                inventory[name]['timesteps'] = sum(len(np.arange(0, float(r['duration_s']), .25)) for r in csv.DictReader(f) if r['status'] == 'ok')
    inventory.setdefault('disk', {})['largest_temporary_video_bytes'] = max((inventory.get(name, {}).get('largest_temporary_video_bytes', 0) for name in ['konvid', 'livevqc']), default=0)
    inventory['disk']['free_bytes'] = shutil.disk_usage('.').free
    rows = []
    if (directory/'timings.csv').exists():
        with (directory/'timings.csv').open() as f:
            rows = list(csv.DictReader(f))
    evidence = {key: read_json(name) for key, name in [('run', 'run.json'), ('backbones', 'backbones.json'), ('rotation', 'rotation_review.json'), ('cloud', 'cloud.json')]}
    evidence.update(current_hashes=importlib.import_module('scripts.00_timing').evidence_hashes(args.config, config),
                    manifest_sha256={name: sha256(f'data/manifests/{name}.csv') for name in ['konvid', 'livevqc'] if Path(f'data/manifests/{name}.csv').exists()},
                    preprocess_sha256=sha256('src/preprocess.py'),
                    active_labor_hours_estimate=config['feasibility'].get('remaining_phase0_labor_hours'))
    try:
        deadline = datetime.date.fromisoformat(config['feasibility']['deadline'])
        evidence['days_remaining'] = (deadline-datetime.datetime.now(ZoneInfo('Asia/Kolkata')).date()).days
    except (TypeError, ValueError):
        evidence['days_remaining'] = None
    report = build_report(config, inventory, rows, evidence)
    report['source_evidence_sha256'] = {str(p): sha256(p) for p in directory.glob('*.json') if p.name not in ('feasibility.json', 'preflight_status.json')}
    (directory/'feasibility.json').write_text(json.dumps(report, indent=2)+'\n')
    text = ['# Phase 0 feasibility', '', 'Status: **'+report['status']+'**', '', 'Missing or failed gates:', '']
    text.extend('- '+gate for gate in report['missing_or_failed_gates'])
    text += ['', 'Measured projections:', '', '```json', json.dumps(report['projections'], indent=2), '```', '', report['cpu_budget_note'], '', report['disk_accounting'], '', 'Hardware checks are separate from offline unit tests. Preprocessing is not frozen until all gates pass. No training or LIVE-VQC predictions have been run.']
    (directory/'report.md').write_text('\n'.join(text)+'\n')
    print(report['status']+': '+', '.join(report['missing_or_failed_gates']))
    return 0 if report['status'] == 'go' else 1


if __name__ == '__main__':
    raise SystemExit(main())
