"""Generate Phase 3 LIVE-VQC metrics and the source-bound exit gate."""
import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cache import atomic_json, atomic_write, csv_bytes, read_manifest
from src.features import sha256
from src.metrics import (bootstrap_indices_hash, classify_difference, correlations,
                         fit_logistic_plcc, holm_adjust, make_bootstrap_indices, paired_bootstrap)
from src.phase3 import load_phase3_context, require_prediction_inputs, validate_live_rows
from src.training import config_hash

HEADS = ('mean_pool', 'bigru', 'transformer', 'mamba')
METRICS = ('srcc', 'krcc', 'plcc_mapped')


def summarize_live(feature, head, y, by_seed):
    seeds = [None] if head == 'ridge' else list(range(5))
    if set(by_seed) != set(seeds):
        raise ValueError('incomplete seed coverage')
    rows = []
    for seed in seeds:
        score = correlations(y, by_seed[seed])
        mapping = fit_logistic_plcc(y, by_seed[seed])
        if score['status'] != 'ok' or mapping['status'] != 'ok':
            raise ValueError(f'undefined LIVE metrics/mapping: {feature}/{head}/seed{seed}')
        rows.append({'feature_set': feature, 'head': head, 'seed': seed,
                     'analysis': 'primary' if feature == 'resnet50' and head in HEADS else 'secondary',
                     'srcc': score['srcc'], 'krcc': score['krcc'], 'plcc_raw': score['plcc_raw'],
                     'plcc_mapped': mapping['plcc_mapped'], 'mapping_status': mapping['status'],
                     'mapping_coefficients': mapping['coefficients'],
                     'mapping_convergence': mapping.get('convergence'),
                     'mapping_domain': 'LIVE-VQC target labels'})
    headline = {'feature_set': feature, 'head': head, 'analysis': rows[0]['analysis'],
                'n_seeds': len(seeds), 'quantity': 'mean of per-seed LIVE metrics' if head != 'ridge' else 'deterministic LIVE metrics'}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        headline[f'{metric}_mean'] = float(np.mean(values))
        headline[f'{metric}_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        headline[f'{metric}_seed0'] = rows[0][metric] if head != 'ridge' else None
    return rows, headline


def comparison_roster():
    roster = [{'comparison_id': f'{a}/{b}', 'family': 'primary',
               'a': ('resnet50', a), 'b': ('resnet50', b)} for a, b in combinations(HEADS, 2)]
    roster.extend({'comparison_id': f'ridge/{head}', 'family': 'secondary',
                   'a': ('resnet50', 'ridge'), 'b': ('resnet50', head)} for head in HEADS)
    roster.extend({'comparison_id': f'feature/{head}', 'family': 'secondary',
                   'a': ('dino', head), 'b': ('resnet50', head)} for head in (*HEADS, 'ridge'))
    return roster


def _load_predictions(context, runs):
    values = {}
    expected_ids = context['ids']['livevqc']
    if len(expected_ids) != 585:
        raise ValueError('LIVE-VQC canonical roster must contain exactly 585 videos')
    for (feature, head, seed), entry in runs.items():
        directory = entry['directory']
        predictions = directory / 'predictions.json'
        metadata = directory / 'predictions.meta.json'
        rows = json.loads(predictions.read_text())
        meta = json.loads(metadata.read_text())
        if (meta.get('status') != 'complete' or meta.get('predictions_sha256') != sha256(predictions) or
                meta.get('run_sha256') != entry['run_sha256'] or
                meta.get('checkpoint_sha256') != entry['run']['checkpoint_sha256'] or
                meta.get('code_sha256') != context.get('code_sha256') or
                meta.get('freeze_sha256') != context['freeze_sha256'] or
                meta.get('phase2_gate_sha256') != context['phase2_gate_sha256']):
            raise ValueError(f'stale LIVE predictions: {directory}')
        validate_live_rows(context, rows, feature, head, seed, entry['run'])
        values[(feature, head, seed)] = np.array([row['prediction'] for row in rows], dtype=float)
    if len(values) != 42:
        raise ValueError('incomplete LIVE prediction roster')
    return values


def _lengths(context):
    manifest = read_manifest(context['config']['cache_root'])
    result = {}
    for dataset in ('konvid', 'livevqc'):
        rows = [row for row in manifest if row['dataset'] == dataset and row['feature_set'] == 'dino_cls']
        if len(rows) != len(context['ids'][dataset]) or set(row['video_id'] for row in rows) != set(context['ids'][dataset]):
            raise ValueError(f'incomplete sequence length manifest: {dataset}')
        values = np.array([int(row['T']) for row in rows])
        result[dataset] = {'count': len(values), 'minimum': int(values.min()),
                           'median': float(np.median(values)), 'maximum': int(values.max()),
                           'p10': float(np.percentile(values, 10)),
                           'p90': float(np.percentile(values, 90))}
    return result


def _gate_outputs(context, generated):
    root = Path(context['config']['results_root'])
    files = [root / name for name in generated]
    files += [root / 'cpu_eligibility.json', root / 'control_fixed_epoch/diagnostic.json',
              root / 'control_fixed_epoch/permutations.json',
              root / 'control_fixed_epoch/seed4_result.json',
              root / 'control_fixed_epoch/untrained_baseline.json',
              root / 'control_fixed_epoch/investigation.json',
              root / 'control_fixed_epoch/qualification.json']
    for seed in range(5):
        directory = root / 'control_fixed_epoch' / f'seed{seed}'
        files += [directory / name for name in ('run.json', 'best.pt', 'predictions.json')]
    for feature in ('resnet50', 'dino'):
        for head in (*HEADS, 'ridge'):
            for seed in ([None] if head == 'ridge' else range(5)):
                directory = root / 'refit' / feature / head / f'seed{seed}'
                files += [directory / name for name in ('run.json', 'model.npz' if head == 'ridge' else 'best.pt',
                                                        'predictions.json', 'predictions.meta.json')]
    return {str(path.relative_to(root)): sha256(path) for path in files}


def _build_phase3_report(config_path: Path) -> dict:
    context = load_phase3_context(config_path)
    root = Path(context['config']['results_root'])
    root.mkdir(parents=True, exist_ok=True)
    gate_path = root / 'phase_gate.json'
    if gate_path.exists():
        previous = json.loads(gate_path.read_text())
        if previous.get('status') == 'passed with qualification':
            if (previous.get('phase2_gate_sha256') != context['phase2_gate_sha256'] or
                    previous.get('freeze_sha256') != context['freeze_sha256'] or
                    previous.get('code_sha256') != context.get('code_sha256') or
                    any(sha256(root / name) != digest for name, digest in previous['outputs_sha256'].items())):
                atomic_json(gate_path, {'status': 'failed', 'reason': 'stale passed report'})
                raise ValueError('stale passed Phase 3 report')
            require_prediction_inputs(context)
            return json.loads((root / 'summary.json').read_text())
    atomic_json(gate_path, {'status': 'failed', 'reason': 'report incomplete'})
    gate = require_prediction_inputs(context)
    predictions = _load_predictions(context, gate['runs'])
    ids = context['ids']['livevqc']
    labels = {row['video_id']: float(row['mos']) for row in context['inventories']['livevqc']}
    y = np.array([labels[key] for key in ids])
    metrics, headline, ensemble = [], [], {}
    for feature in ('resnet50', 'dino'):
        for head in (*HEADS, 'ridge'):
            by_seed = {seed: predictions[(feature, head, seed)] for seed in ([None] if head == 'ridge' else range(5))}
            rows, summary = summarize_live(feature, head, y, by_seed)
            metrics.extend(rows)
            headline.append(summary)
            ensemble[(feature, head)] = by_seed[None] if head == 'ridge' else np.mean(list(by_seed.values()), axis=0)
    atomic_write(root / 'metrics.csv', csv_bytes(metrics, list(metrics[0])))
    atomic_write(root / 'headline.csv', csv_bytes(headline, list(headline[0])))
    indices = make_bootstrap_indices(585, seed=0, draws=10000)
    bank_hash = bootstrap_indices_hash(indices)
    comparisons = []
    for item in comparison_roster():
        scores = {}
        for metric in ('srcc', 'krcc'):
            result = paired_bootstrap(y, ensemble[item['a']], ensemble[item['b']], metric,
                                      seed=0, draws=10000, resample_indices=indices)
            if result['status'] != 'ok':
                raise ValueError(f'failed paired bootstrap: {item["comparison_id"]}/{metric}')
            scores[metric] = {**result, 'classification': classify_difference(result['ci95'], margin=.01),
                              'adjustment': 'Holm primary SRCC' if item['family'] == 'primary' and metric == 'srcc' else 'unadjusted'}
        comparisons.append({**item, **scores})
    adjusted = holm_adjust({item['comparison_id']: item['srcc']['p_raw'] for item in comparisons if item['family'] == 'primary'})
    for item in comparisons:
        item['srcc']['p_holm'] = adjusted[item['comparison_id']] if item['family'] == 'primary' else None
        item['krcc']['p_holm'] = None
    atomic_json(root / 'comparisons.json', comparisons)
    comparison_rows = []
    for item in comparisons:
        for metric in ('srcc', 'krcc'):
            score = item[metric]
            comparison_rows.append({'comparison_id': item['comparison_id'], 'family': item['family'],
                                    'metric': metric, 'a': '/'.join(item['a']), 'b': '/'.join(item['b']),
                                    'delta': score['delta'], 'ci95_low': score['ci95'][0],
                                    'ci95_high': score['ci95'][1], 'p_raw': score['p_raw'],
                                    'p_holm': score['p_holm'], 'classification': score['classification'],
                                    'adjustment': score['adjustment']})
    atomic_write(root / 'comparisons.csv', csv_bytes(comparison_rows, list(comparison_rows[0])))
    atomic_json(root / 'comparisons.meta.json', {'draws': 10000, 'seed': 0, 'video_ids_sha256': config_hash(ids),
                                                 'resample_indices_sha256': bank_hash,
                                                 'primary_srcc_pairs': 6, 'secondary_pairs': 9,
                                                 'pvalue_method': 'centered_bootstrap_add_one',
                                                 'ci_scope': 'seed-averaged prediction, LIVE video resampling',
                                                 'margin': .01, 'min_finite_fraction': .99})
    atomic_json(root / 'lengths.json', _lengths(context))
    winner = gate['winner']
    selected = (winner['feature_set'], winner['head'])
    seed = None if winner['head'] == 'ridge' else 0
    run = gate['runs'][(*selected, seed)]['run']
    chosen = next(row for row in headline if (row['feature_set'], row['head']) == selected)
    deployment = {'candidate_id': winner['candidate_id'], 'feature_set': winner['feature_set'],
                  'head': winner['head'], 'seed': seed, 'checkpoint_sha256': run['checkpoint_sha256'],
                  'checkpoint_path': str(gate['runs'][(*selected, seed)]['checkpoint']),
                  'live_seed0': next(row for row in metrics if (row['feature_set'], row['head'], row['seed']) == (*selected, seed)),
                  'live_five_seed': chosen if seed is not None else None,
                  'cpu_eligibility_sha256': sha256(root / 'cpu_eligibility.json'), 'cpu_threads': 4,
                  'serving': 'single FastAPI process; set torch.set_num_threads(4) before model work; remeasure end-to-end latency'}
    atomic_json(root / 'deployment.json', deployment)
    summary = {'status': 'passed with qualification', 'dataset': 'LIVE-VQC', 'videos': 585,
               'selected_feature_set': context['selected_feature_set'], 'cpu_winner': winner,
               'control_status': 'failed',
               'control_investigation': 'control_fixed_epoch/investigation.json',
               'decision': 'DECISIONS.md: 2026-10-28 Phase 3 qualified continuation after fixed-epoch control failure',
               'headline_rows': len(headline), 'prediction_runs': len(predictions),
               'primary_pairs': 6, 'secondary_pairs': 9,
               'method': 'Raw predictions for SRCC/KRCC; four-parameter PLCC mapping fitted on LIVE-VQC target labels. '
                         'Neural headline mean/sample SD are across five seeds. Paired intervals use seed-averaged predictions '
                         'and one shared 10,000-draw LIVE-video index bank, capturing test sampling noise only. '
                         'Only six ResNet neural SRCC pairs receive Holm adjustment; other p-values are secondary and unadjusted. '
                         '95% CI classification checks equivalence within ±0.01 before directional difference.',
               'limitations': ['KoNViD feature selection used inner-validation scores from videos also present in head comparisons.',
                               'Fixed-epoch shuffled-label control failed at seeds 3 and 4; this qualified continuation does not reclassify that result.',
                               'Five-seed trained ensemble also failed the original rule; untrained baseline failed it, while trained predictions were mostly unlike initialization.',
                               'Full-data median CV epochs approximate a different steps-per-epoch schedule.',
                               'KoNViD and LIVE-VQC differ in domain and sequence lengths.',
                               'Center crop loses edge content, especially for portrait videos.',
                               'UGC and separate length-effect claims were cut from scope.']}
    atomic_json(root / 'summary.json', summary)
    generated = ('metrics.csv', 'headline.csv', 'comparisons.json', 'comparisons.csv',
                 'comparisons.meta.json', 'lengths.json', 'deployment.json', 'summary.json')
    outputs = _gate_outputs(context, generated)
    atomic_json(gate_path, {'status': 'passed with qualification',
                            'control_status': 'failed',
                            'qualification_decision_sha256': gate['control']['decision_sha256'],
                            'phase2_gate_sha256': context['phase2_gate_sha256'],
                            'freeze_sha256': context['freeze_sha256'], 'code_sha256': context.get('code_sha256'),
                            'outputs_sha256': outputs,
                            'prediction_runs': 42, 'live_videos': 585,
                            'deployment_checkpoint_sha256': run['checkpoint_sha256']})
    return summary


def phase3_report(config_path: Path) -> dict:
    try:
        return _build_phase3_report(config_path)
    except BaseException:
        try:
            config = yaml.safe_load(Path(config_path).read_text())
            gate_path = Path(config['results_root']) / 'phase_gate.json'
            atomic_json(gate_path, {'status': 'failed', 'reason': 'report validation or generation failed'})
        except (OSError, KeyError, TypeError, ValueError):
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(phase3_report(args.config), indent=2))


if __name__ == '__main__':
    main()
