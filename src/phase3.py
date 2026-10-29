"""Phase 3 source and cache gates."""
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

from src.cache import DINO_BRANCHES, reusable
from src.features import sha256
from src.training import config_hash


def source_code_sha256(root: Path = Path(__file__).resolve().parents[1]) -> str:
    digest = hashlib.sha256()
    for folder in ('src', 'scripts'):
        for path in sorted((Path(root) / folder).rglob('*')):
            if path.suffix not in ('.py', '.sh') or not path.is_file():
                continue
            digest.update(str(path.relative_to(root)).encode() + b'\0' + sha256(path).encode() + b'\n')
    return digest.hexdigest()


def _inventory(path, dataset, *, include_labels=True):
    rows = list(csv.DictReader(Path(path).open()))
    ids = [r['video_id'] for r in rows]
    if not rows or len(ids) != len(set(ids)) or any(r['dataset'] != dataset or
            (include_labels and not math.isfinite(float(r['mos']))) for r in rows):
        raise ValueError(f'invalid {dataset} inventory')
    if not include_labels:
        rows = [{key: value for key, value in row.items() if key != 'mos'} for row in rows]
    return rows


def load_phase3_context(config_path: Path, *, include_live_labels=True) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())
    gate_path = Path(config['phase2_gate'])
    gate = json.loads(gate_path.read_text())
    if gate.get('status') != 'passed' or not gate.get('outputs_sha256'):
        raise ValueError('Phase 2 gate did not pass')
    for name, expected in gate['outputs_sha256'].items():
        if sha256(gate_path.parent / name) != expected:
            raise ValueError(f'changed Phase 2 output: {name}')
    freeze_path = Path(config['phase2_freeze'])
    if (gate['outputs_sha256'].get('konvid_freeze.json') is None or
            sha256(freeze_path) != gate['outputs_sha256']['konvid_freeze.json']):
        raise ValueError('configured freeze is not the gate-pinned Phase 2 freeze')
    freeze = json.loads(freeze_path.read_text())
    selection_path = gate_path.parent / 'selection.json'
    selection = json.loads(selection_path.read_text())
    if (freeze.get('status') != 'research_configuration_frozen' or
            config_hash(selection) != freeze.get('selection_sha256') or
            selection.get('selected_feature_set') != freeze.get('selected_feature_set') or
            len(freeze['deployment']['candidates']) != 10):
        raise ValueError('Phase 2 freeze/selection mismatch')
    inventories = {name: _inventory(config[f'{name if name == "konvid" else "live"}_inventory'], name,
                                  include_labels=name != 'livevqc' or include_live_labels)
                   for name in ('konvid', 'livevqc')}
    return {'config': config, 'phase2_gate': gate, 'phase2_gate_sha256': sha256(gate_path),
            'freeze': freeze, 'freeze_sha256': sha256(freeze_path),
            'code_sha256': source_code_sha256(),
            'selected_feature_set': freeze['selected_feature_set'],
            'candidates': freeze['deployment']['candidates'],
            'provenance': freeze.get('provenance'), 'inventories': inventories,
            'ids': {key: [r['video_id'] for r in rows] for key, rows in inventories.items()}}


def live_cache_gaps(context: dict) -> dict[str, list[str]]:
    root = Path(context['config']['cache_root'])
    rows = context['inventories']['livevqc']
    return {feature: [r['video_id'] for r in rows if not reusable(root, 'livevqc', r['video_id'], branches, r)]
            for feature, branches in (('dino', DINO_BRANCHES), ('resnet50', ['resnet50']))}


def resolve_cpu_winner(freeze: dict, measurements: dict) -> dict:
    candidates = freeze['deployment']['candidates']
    ids = measurements.get('sample_ids', [])
    repeats = measurements.get('repeats')
    if (measurements.get('cpu_threads') != 4 or repeats != 3 or len(ids) != 20 or
            len(set(ids)) != 20 or len(candidates) != 10):
        raise ValueError('invalid CPU sample/repeat/thread policy')
    candidate_ids = {c['candidate_id'] for c in candidates}
    expected = {(c, key, repeat) for c in candidate_ids for key in ids for repeat in range(3)}
    seen = set()
    latency = {c: 0. for c in candidate_ids}
    for row in measurements.get('rows', []):
        key = (row.get('candidate_id'), row.get('video_id'), row.get('repeat'))
        if key not in expected or key in seen or row.get('dataset') != 'konvid' or row.get('status', 'ok') != 'ok':
            raise ValueError('invalid/duplicate CPU row')
        seen.add(key)
        try:
            numbers = [float(row[name]) for name in ('duration_s', 'decode_seconds',
                'preprocess_seconds', 'backbone_seconds', 'head_seconds', 'total_seconds',
                'normalized_total_seconds')]
            if (numbers[0] <= 0 or any(value < 0 or not math.isfinite(value) for value in numbers) or
                    not math.isclose(numbers[5], sum(numbers[1:5]), rel_tol=1e-7, abs_tol=1e-7) or
                    not math.isclose(numbers[6], 10 * numbers[5] / numbers[0], rel_tol=1e-7, abs_tol=1e-7) or
                    any(len(row[name]) != 64 for name in ('source_sha256', 'config_sha256', 'checkpoint_sha256'))):
                raise ValueError('invalid CPU stage/time/hash')
        except (KeyError, TypeError) as exc:
            raise ValueError('missing CPU stage/hash') from exc
        latency[key[0]] = max(latency[key[0]], numbers[6])
    if seen != expected:
        raise ValueError('incomplete CPU measurement grid')
    budget = freeze['deployment']['cpu_budget_seconds_per_10s']
    eligible = [c for c in candidates if latency[c['candidate_id']] <= budget]
    if not eligible:
        raise ValueError('no CPU-eligible candidate')
    winner = min(eligible, key=lambda c: (-c['headline_cv_srcc'], latency[c['candidate_id']], c['candidate_id']))
    return {'candidate_id': winner['candidate_id'], 'feature_set': winner['feature_set'],
            'head': winner['head'], 'headline_cv_srcc': winner['headline_cv_srcc'],
            'cpu_latency_seconds_per_10s': latency[winner['candidate_id']],
            'candidate_latency': latency, 'cpu_threads': 4}


def require_cpu_winner(context: dict) -> dict:
    config = context['config']
    evidence_path = Path(config['results_root']) / 'cpu_eligibility.json'
    evidence = json.loads(evidence_path.read_text())
    if (evidence.get('phase2_gate_sha256') != context['phase2_gate_sha256'] or
            evidence.get('freeze_sha256') != context['freeze_sha256'] or
            evidence.get('code_sha256') != context.get('code_sha256') or
            evidence.get('sample_ids') != Path(config['cpu_sample_ids']).read_text().splitlines()):
        raise ValueError('stale CPU source/sample identity')
    inventory = {r['video_id']: r for r in context['inventories']['konvid']}
    candidates = {c['candidate_id']: c for c in context['candidates']}
    checked = set()
    for row in evidence['rows']:
        candidate = candidates.get(row.get('candidate_id'))
        source = inventory.get(row.get('video_id'))
        if (candidate is None or source is None or row['source_sha256'] != source['source_sha256'] or
                row['config_sha256'] != config_hash(candidate['config']) or
                float(row['duration_s']) != float(source['duration_s'])):
            raise ValueError('CPU row source/config mismatch')
        cv = candidate['refit']['cv_selected_alphas'][0] if candidate['head'] == 'ridge' else candidate['refit']['cv_epochs'][0]
        if row['checkpoint_sha256'] != cv['checkpoint_sha256']:
            raise ValueError('CPU checkpoint/source mismatch')
        if candidate['candidate_id'] not in checked:
            checkpoint = Path(cv['run_path']).parent / ('model.npz' if candidate['head'] == 'ridge' else 'best.pt')
            if sha256(checkpoint) != cv['checkpoint_sha256'] or sha256(cv['run_path']) != cv['run_sha256']:
                raise ValueError('CPU checkpoint/source mismatch')
            checked.add(candidate['candidate_id'])
    winner = resolve_cpu_winner(context['freeze'], evidence)
    if evidence.get('winner') != winner:
        raise ValueError('saved CPU winner changed')
    return winner


def require_fixed_control(context: dict) -> dict:
    from src.metrics import bootstrap_srcc
    from src.tuning import control_directory, require_control
    for phase_root in ('results/phase1', 'results/phase2/resnet50'):
        for final in (False, True):
            require_control(control_directory(phase_root, final), context['provenance'])
    root = Path(context['config']['results_root']) / 'control_fixed_epoch'
    record = json.loads((root / 'diagnostic.json').read_text())
    if (record.get('status') != 'passed' or record.get('feature_set') != context['selected_feature_set'] or
            record.get('freeze_sha256') != context['freeze_sha256'] or
            record.get('code_sha256') != context.get('code_sha256') or
            record.get('phase2_gate_sha256') != context['phase2_gate_sha256'] or
            sha256(root / 'permutations.json') != record.get('permutations_sha256') or
            len(record.get('seeds', [])) != 5):
        raise ValueError('failed/stale fixed-epoch control')
    import numpy as np
    split = context['freeze']['configs']['mean_pool']['splits']
    test_ids = [r['video_id'] for r in csv.DictReader(Path(split).open())
                if int(r['outer_fold']) == 0 and r['role'] == 'test']
    labels = {r['video_id']: float(r['mos']) for r in context['inventories']['konvid']}
    frozen_epoch = context['freeze']['refit_by_feature'][context['selected_feature_set']]['mean_pool']['median_best_epoch']
    for seed, item in enumerate(record['seeds']):
        directory = root / f'seed{seed}'
        run = json.loads((directory / 'run.json').read_text())
        rows = json.loads((directory / 'predictions.json').read_text())
        if (item.get('seed') != seed or item.get('status') != 'passed' or
                sha256(directory / 'run.json') != item.get('run_sha256') or
                sha256(directory / 'best.pt') != item.get('checkpoint_sha256') or
                sha256(directory / 'predictions.json') != item.get('predictions_sha256') or
                run.get('selection') != 'fixed_final_epoch' or run.get('seed') != seed or
                run.get('completed_epochs') != frozen_epoch or run.get('best_epoch') != frozen_epoch or
                run.get('checkpoint_sha256') != item.get('checkpoint_sha256')):
            raise ValueError('incomplete fixed-epoch control seed')
        ids = [row['video_id'] for row in rows]
        if (ids != test_ids or len(ids) < 2 or
                any(row.get('checkpoint_sha256') != item['checkpoint_sha256'] or
                    row.get('mos') != labels[row['video_id']] for row in rows)):
            raise ValueError('invalid fixed-epoch control predictions')
        score = bootstrap_srcc(np.array([row['mos'] for row in rows]),
                               np.array([row['prediction'] for row in rows]))
        if (score != item['bootstrap'] or score['status'] != 'ok' or
                not score['ci95'][0] <= 0 <= score['ci95'][1] or abs(score['srcc']) >= .15):
            raise ValueError('failed fixed-epoch control score')
    return record


def require_qualified_control(context: dict) -> dict:
    """Validate the preserved failure and the separately approved continuation."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from src.metrics import bootstrap_srcc
    from src.tuning import control_directory, require_control

    for phase_root in ('results/phase1', 'results/phase2/resnet50'):
        for final in (False, True):
            require_control(control_directory(phase_root, final), context['provenance'])
    root = Path(context['config']['results_root']) / 'control_fixed_epoch'
    diagnostic_path = root / 'diagnostic.json'
    record = json.loads(diagnostic_path.read_text())
    last_path = root / 'seed4_result.json'
    last = json.loads(last_path.read_text())
    investigation_path = root / 'investigation.json'
    investigation = json.loads(investigation_path.read_text())
    baseline_path = root / 'untrained_baseline.json'
    baseline = json.loads(baseline_path.read_text())
    qualification = json.loads((root / 'qualification.json').read_text())
    decision_path = Path('DECISIONS.md')
    heading = '## 2026-10-28: Phase 3 qualified continuation after fixed-epoch control failure'
    if (record.get('status') != 'failed' or len(record.get('seeds', [])) != 4 or
            record['seeds'][3].get('seed') != 3 or record['seeds'][3].get('status') != 'failed' or
            last.get('seed') != 4 or
            record.get('freeze_sha256') != context['freeze_sha256'] or
            record.get('phase2_gate_sha256') != context['phase2_gate_sha256'] or
            investigation.get('original_control_status') != 'failed' or
            investigation.get('original_diagnostic_sha256') != sha256(diagnostic_path) or
            investigation.get('seed4_result_sha256') != sha256(last_path) or
            investigation.get('untrained_baseline_sha256') != sha256(baseline_path) or
            investigation.get('permutations_sha256') != sha256(root / 'permutations.json') or
            investigation.get('code_sha256_at_control') != record.get('code_sha256') or
            baseline.get('code_sha256') != record.get('code_sha256') or
            baseline.get('freeze_sha256') != context['freeze_sha256'] or
            qualification.get('status') != 'approved_qualified_continuation' or
            qualification.get('original_diagnostic_sha256') != sha256(diagnostic_path) or
            qualification.get('investigation_sha256') != sha256(investigation_path) or
            qualification.get('decision_sha256') != sha256(decision_path) or
            heading not in decision_path.read_text()):
        raise ValueError('missing/stale qualified control decision or failure evidence')

    feature = context['selected_feature_set']
    spec = context['freeze']['refit_by_feature'][feature]['mean_pool']
    split = spec['config']['splits']
    roles = list(csv.DictReader(Path(split).open()))
    test_ids = [r['video_id'] for r in roles if int(r['outer_fold']) == 0 and r['role'] == 'test']
    permutations = json.loads((root / 'permutations.json').read_text())
    for role in ('train', 'val'):
        ids = [r['video_id'] for r in roles if int(r['outer_fold']) == 0 and r['role'] == role]
        order = np.random.default_rng(0).permutation(len(ids))
        expected = [{'video_id': key, 'label_source_id': ids[int(j)]} for key, j in zip(ids, order)]
        if permutations.get(role) != expected:
            raise ValueError(f'changed {role} control permutation')
    labels = {r['video_id']: float(r['mos']) for r in context['inventories']['konvid']}
    items = record['seeds'] + [last]
    trained, untrained = [], []
    for seed, item in enumerate(items):
        directory = root / f'seed{seed}'
        run = json.loads((directory / 'run.json').read_text())
        rows = json.loads((directory / 'predictions.json').read_text())
        initial = baseline['seeds'][seed]['rows']
        if (item.get('seed') != seed or
                sha256(directory / 'run.json') != item.get('run_sha256') or
                sha256(directory / 'best.pt') != item.get('checkpoint_sha256') or
                sha256(directory / 'predictions.json') != item.get('predictions_sha256') or
                run.get('status') != 'complete' or run.get('config') != spec['config'] or
                run.get('selection') != 'fixed_final_epoch' or run.get('seed') != seed or
                run.get('completed_epochs') != spec['median_best_epoch'] or
                run.get('best_epoch') != spec['median_best_epoch'] or
                run.get('checkpoint_sha256') != item['checkpoint_sha256'] or
                run.get('provenance', {}).get('code_sha256') != record['code_sha256'] or
                run.get('provenance', {}).get('label_mode') != 'shuffled' or
                run.get('provenance', {}).get('control_permutations_sha256') != sha256(root / 'permutations.json') or
                [r['video_id'] for r in rows] != test_ids or
                [r['video_id'] for r in initial] != test_ids or
                any(r['mos'] != labels[r['video_id']] for r in rows + initial)):
            raise ValueError(f'incompatible qualified control seed {seed}')
        y = np.array([r['mos'] for r in rows])
        p = np.array([r['prediction'] for r in rows])
        q = np.array([r['prediction'] for r in initial])
        score = bootstrap_srcc(y, p)
        null_score = bootstrap_srcc(y, q)
        passed = score['status'] == 'ok' and score['ci95'][0] <= 0 <= score['ci95'][1] and abs(score['srcc']) < .15
        detail = investigation['seeds'][seed]
        if (score != item['bootstrap'] or item['status'] != ('passed' if passed else 'failed') or
                null_score != baseline['seeds'][seed]['bootstrap'] or
                detail['trained'] != score or detail['untrained'] != null_score or
                detail['status_under_unchanged_rule'] != item['status'] or
                not math.isclose(detail['trained_vs_untrained_pearson_r'], float(pearsonr(p, q).statistic), abs_tol=1e-12) or
                not math.isclose(detail['trained_vs_untrained_spearman_r'], float(spearmanr(p, q).statistic), abs_tol=1e-12)):
            raise ValueError(f'changed qualified control statistic: seed {seed}')
        trained.append(p)
        untrained.append(q)
    if (bootstrap_srcc(y, np.mean(trained, axis=0)) != investigation['phase1_phase2_comparable_trained_ensemble'] or
            bootstrap_srcc(y, np.mean(untrained, axis=0)) != investigation['untrained_ensemble']):
        raise ValueError('changed qualified control ensemble')
    return {'status': 'passed with qualification', 'original_status': 'failed',
            'decision_sha256': sha256(decision_path), 'investigation_sha256': sha256(investigation_path)}


def required_refits(context: dict) -> dict:
    """Read all 42 frozen full-data artifacts before allowing LIVE inference."""
    from src.training import config_hash
    result = {}
    root = Path(context['config']['results_root']) / 'refit'
    candidates = {c['candidate_id']: c for c in context['candidates']}
    for feature in ('resnet50', 'dino'):
        for head in ('mean_pool', 'bigru', 'transformer', 'mamba', 'ridge'):
            spec = context['freeze']['refit_by_feature'][feature][head]
            sources = spec['cv_selected_alphas'] if head == 'ridge' else spec['cv_epochs']
            if len(sources) != (5 if head == 'ridge' else 25):
                raise ValueError('incomplete frozen CV sources')
            for source in sources:
                path = Path(source['run_path'])
                artifact = path.parent / ('model.npz' if head == 'ridge' else 'best.pt')
                if sha256(path) != source['run_sha256'] or sha256(artifact) != source['checkpoint_sha256']:
                    raise ValueError(f'changed frozen CV source: {path}')
            source_hashes = [source['run_sha256'] for source in sources]
            config = candidates[f'{feature}/{head}']['config'] if head == 'ridge' else spec['config']
            seeds = [None] if head == 'ridge' else range(5)
            for seed in seeds:
                directory = root / feature / head / f'seed{seed}'
                run = json.loads((directory / 'run.json').read_text())
                checkpoint = directory / ('model.npz' if head == 'ridge' else 'best.pt')
                provenance = run.get('provenance', {})
                if (run.get('status') != 'complete' or run.get('seed') != seed or
                        run.get('checkpoint_sha256') != sha256(checkpoint) or
                        provenance.get('phase2_gate_sha256') != context['phase2_gate_sha256'] or
                        provenance.get('freeze_sha256') != context['freeze_sha256'] or
                        provenance.get('code_sha256') != context.get('code_sha256') or
                        provenance.get('cv_run_sha256') != source_hashes or
                        provenance.get('head') != head or provenance.get('feature_set') != feature or
                        provenance.get('fold') is not None or
                        provenance.get('inventory_sha256') != sha256(context['config']['konvid_inventory']) or
                        any(provenance.get(k) != v for k, v in context['provenance'].items() if k != 'code_commit')):
                    raise ValueError(f'incompatible full-data refit: {directory}')
                if head == 'ridge':
                    if (run.get('alpha') != spec['phase3_alpha'] or run.get('selection') != 'fixed_alpha' or
                            run['identity'].get('config_sha256') != config_hash(config) or
                            run['identity'].get('provenance') != provenance or
                            run['identity'].get('inventory_sha256') != sha256(context['config']['konvid_inventory'])):
                        raise ValueError(f'incompatible ridge refit: {directory}')
                elif (run.get('selection') != 'fixed_final_epoch' or
                        run.get('completed_epochs') != spec['median_best_epoch'] or
                        run.get('config') != config or
                        run['identity'].get('epochs') != spec['median_best_epoch'] or
                        run['identity'].get('config_sha256') != config_hash(config) or
                        run['identity'].get('provenance') != {k: v for k, v in provenance.items() if k != 'code_commit'} or
                        run['identity'].get('seed') != seed):
                    raise ValueError(f'incompatible fixed-epoch refit: {directory}')
                if head != 'ridge':
                    import torch
                    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                    if (saved.get('selection') != 'fixed_final_epoch' or
                            saved.get('epoch') != spec['median_best_epoch'] or
                            saved.get('config') != config or saved.get('target_stats') != run.get('target_stats')):
                        raise ValueError(f'incompatible fixed-final checkpoint: {directory}')
                result[(feature, head, seed)] = {'run': run, 'directory': directory,
                                                'run_sha256': sha256(directory / 'run.json'),
                                                'checkpoint': checkpoint}
    if len(result) != 42:
        raise ValueError('missing full-data refits')
    return result


def require_prediction_inputs(context: dict) -> dict:
    from src.cache import validate_cache
    if any(live_cache_gaps(context).values()):
        raise ValueError('LIVE feature cache incomplete')
    validate_cache(context['config']['cache_root'], context['config']['feature_config'])
    winner = require_cpu_winner(context)
    control = require_qualified_control(context)
    runs = required_refits(context)
    return {'winner': winner, 'control': control, 'runs': runs}


def validate_live_rows(context, rows, feature, head, seed, run):
    expected = context['ids']['livevqc']
    if not isinstance(rows, list) or [r.get('video_id') for r in rows] != expected:
        raise ValueError('missing, duplicate or reordered LIVE predictions')
    for row in rows:
        key = row['video_id']
        if (row.get('dataset') != 'livevqc' or row.get('feature_set') != feature or
                row.get('head') != head or row.get('seed') != seed or
                row.get('checkpoint_sha256') != run['checkpoint_sha256'] or
                row.get('run_identity') != run['identity'] or
                'mos' in row or
                not math.isfinite(row['prediction'])):
            raise ValueError(f'invalid LIVE prediction row: {key}')
