"""Matched post-hoc neural null; never changes the original control or gate."""
import argparse
import csv
import fcntl
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.cache import atomic_json, atomic_write
from src.data import CachedFeatureDataset
from src.features import sha256
from src.heads import build_head
from src.phase3 import source_code_sha256
from src.training import config_hash, fit_fixed_epochs, predict, seed_everything

ROOT = Path('results/phase3')
HERE = ROOT / 'posthoc_permutation'
OUTPUT = HERE / 'neural'
PERMUTATION_SEEDS = tuple(range(1, 201))
MODEL_SEEDS = tuple(range(5))


def shuffled_role(items, seed):
    order = np.random.default_rng(seed).permutation(len(items))
    mapping = [{'video_id': item['video_id'], 'label_source_id': items[int(j)]['video_id']}
               for item, j in zip(items, order, strict=True)]
    return [{**item, 'mos': items[int(j)]['mos']} for item, j in zip(items, order, strict=True)], mapping


def ensemble_score(predictions, labels):
    matrix = np.asarray(predictions, dtype=float)
    if matrix.shape != (5, len(labels)) or not np.isfinite(matrix).all():
        raise ValueError('five finite, aligned prediction vectors required')
    values = matrix.mean(axis=0)
    score = float(spearmanr(labels, values).statistic)
    if not np.isfinite(score):
        raise ValueError('undefined ensemble SRCC; do not discard permutation')
    return values, score


def interpret(scores, historical):
    if len(scores) != 200 or not np.isfinite(scores).all():
        raise ValueError('interpretation requires all 200 finite ensemble scores')
    low, high = np.quantile(scores, [.025, .975], method='linear')
    verdict = ('consistent with chance under the control’s own procedure' if low <= historical <= high
               else 'not explained by chance; investigate')
    return [float(low), float(high)], verdict


def protected_hashes():
    gate = json.loads((ROOT / 'phase_gate.json').read_text())
    if gate['status'] != 'passed with qualification' or gate['control_status'] != 'failed':
        raise ValueError('original gate classification changed')
    for name, expected in gate['outputs_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'original evidence changed: {name}')
    if source_code_sha256() != gate['code_sha256']:
        raise ValueError('frozen Phase 3 source changed')
    if sha256(Path('DECISIONS.md')) != gate['qualification_decision_sha256']:
        raise ValueError('pinned decision changed')
    if sha256(Path('results/phase2/konvid_freeze.json')) != gate['freeze_sha256']:
        raise ValueError('frozen configuration changed')
    paths = [ROOT / 'phase_gate.json', Path('DECISIONS.md')]
    paths += [ROOT / name for name in gate['outputs_sha256']]
    return {str(path): sha256(path) for path in paths}


def run(check_only=False):
    protected = protected_hashes()
    freeze_path = Path('results/phase2/konvid_freeze.json')
    freeze = json.loads(freeze_path.read_text())
    spec = freeze['refit_by_feature']['resnet50']['mean_pool']
    config, epochs = spec['config'], spec['median_best_epoch']
    historical_run = json.loads((ROOT / 'control_fixed_epoch/seed0/run.json').read_text())
    if epochs != 3 or config != historical_run['config'] or historical_run['completed_epochs'] != 3:
        raise ValueError('procedure differs from historical control')
    input_paths = [freeze_path, Path(config['inventory']), Path(config['splits']),
                   Path(config['cache_root']) / 'meta.json', Path('env/requirements.lock'),
                   Path('configs/control_v2.yaml'), Path(__file__), HERE / 'NEURAL_PROTOCOL.md']
    for path, key in [(Path(config['inventory']), 'inventory_sha256'),
                      (Path(config['splits']), 'split_sha256'),
                      (Path(config['cache_root']) / 'meta.json', 'cache_meta_sha256'),
                      (Path('env/requirements.lock'), 'lock_sha256')]:
        if sha256(path) != historical_run['provenance'][key]:
            raise ValueError(f'historical input changed: {path}')
    identity = {'inputs_sha256': {str(p): sha256(p) for p in input_paths},
                'protected_sha256': protected, 'config_sha256': config_hash(config),
                'permutation_seeds': list(PERMUTATION_SEEDS), 'model_seeds': list(MODEL_SEEDS), 'epochs': epochs}
    datasets = {role: CachedFeatureDataset(Path(config['cache_root']), Path(config['inventory']),
                Path(config['splits']), 0, role, 'resnet50') for role in ('train', 'val', 'test')}
    items = {role: data.items for role, data in datasets.items()}
    ids = {role: [item['video_id'] for item in rows] for role, rows in items.items()}
    if any(set(ids[a]) & set(ids[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('overlapping roles')
    mapping0 = json.loads((ROOT / 'control_fixed_epoch/permutations.json').read_text())
    if any(shuffled_role(items[role], 0)[1] != mapping0[role] for role in ('train', 'val')):
        raise ValueError('historical permutation/row ordering mismatch')
    labels = np.array([item['mos'] for item in items['test']])
    historical = json.loads((ROOT / 'control_fixed_epoch/investigation.json').read_text())['phase1_phase2_comparable_trained_ensemble']['srcc']
    if check_only:
        print('Preflight passed: original hashes intact, historical config/order matched; no training started.', flush=True)
        return
    execution_path = OUTPUT / 'execution.json'
    execution = {'identity': identity, 'historical_ensemble_srcc': historical, 'role_ids': ids}
    if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
        raise ValueError('resume identity differs; do not mix diagnostic runs')
    atomic_json(execution_path, execution)
    results = []
    for permutation in PERMUTATION_SEEDS:
        directory = OUTPUT / f'permutation{permutation:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        data, mappings = [], {}
        for role in ('train', 'val'):
            shuffled, mappings[role] = shuffled_role(items[role], permutation)
            data.extend(shuffled)
        atomic_json(directory / 'permutations.json', mappings)
        provenance = {'diagnostic': 'post_hoc_matched_neural_null', 'head': 'mean_pool',
                      'permutation_seed': permutation, 'execution_sha256': sha256(execution_path),
                      'control_permutations_sha256': sha256(directory / 'permutations.json')}
        predictions, run_hashes = [], {}
        for seed in MODEL_SEEDS:
            seed_everything(seed, config['cpu_threads'])
            model = build_head('mean_pool', 2048, config)
            seed_dir = directory / f'seed{seed}'
            record = fit_fixed_epochs(model, data, config, seed_dir, seed, provenance, epochs)
            rows = predict(model, datasets['test'], record['target_stats'], config['device'])
            if [row['video_id'] for row in rows] != ids['test']:
                raise ValueError('test prediction order mismatch')
            atomic_json(seed_dir / 'predictions.json', rows)
            predictions.append([row['prediction'] for row in rows])
            run_hashes[str(seed)] = {name: sha256(seed_dir / name) for name in ('run.json', 'best.pt', 'predictions.json')}
            print(f'Permutation {permutation}/200, model seed {seed}: complete', flush=True)
            del model
        values, score = ensemble_score(predictions, labels)
        atomic_json(directory / 'ensemble_predictions.json',
                    [{'video_id': key, 'prediction': float(p)} for key, p in zip(ids['test'], values, strict=True)])
        result = {'permutation_seed': permutation, 'ensemble_test_srcc': score,
                  'per_seed_test_srcc': [float(spearmanr(labels, p).statistic) for p in predictions],
                  'runs_sha256': run_hashes, 'ensemble_predictions_sha256': sha256(directory / 'ensemble_predictions.json')}
        atomic_json(directory / 'result.json', result)
        results.append(result)
        atomic_json(OUTPUT / 'progress.json', {'completed_permutations': len(results), 'total': 200,
                                              'last_permutation_seed': permutation, 'status': 'running'})
    interval, verdict = interpret([r['ensemble_test_srcc'] for r in results], historical)
    if protected_hashes() != protected:
        raise ValueError('protected identity changed during diagnostic')
    if any(sha256(Path(path)) != value for path, value in identity['inputs_sha256'].items()):
        raise ValueError('diagnostic input changed during execution')
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['permutation_seed', 'ensemble_test_srcc'])
    writer.writerows((r['permutation_seed'], r['ensemble_test_srcc']) for r in results)
    atomic_write(OUTPUT / 'per_permutation.csv', buffer.getvalue().encode())
    summary = {'status': 'complete_post_hoc_diagnostic_only', 'permutations': 200, 'models_per_permutation': 5,
               'historical_ensemble_srcc': historical, 'central_95_percent_null_interval': interval,
               'interpretation': verdict, 'original_control_status': 'failed', 'phase3_gate_status': 'passed with qualification',
               'execution_sha256': sha256(execution_path), 'results': results,
               'per_permutation_csv_sha256': sha256(OUTPUT / 'per_permutation.csv')}
    atomic_json(OUTPUT / 'summary.json', summary)
    report = f'''# Post-hoc matched neural permutation null

All 200 predetermined permutation seeds (1–200) completed with five model seeds (0–4) each. Each model used the historical frozen ResNet mean-pool config and three fixed epochs on fold-0 train plus validation rows, with within-role shuffled labels. Test MOS was used only for evaluation.

The historical five-seed ensemble SRCC is {historical:.6f}. The central 95% null interval over per-permutation ensemble SRCCs is [{interval[0]:.6f}, {interval[1]:.6f}] (linear 2.5th and 97.5th percentiles).

Predeclared interpretation: **{verdict}**.

This is a finite post-hoc Monte Carlo diagnostic, not proof of the absence or presence of leakage. The interval is a permutation interval, not an accuracy confidence interval. The original control remains **failed**, and the Phase 3 gate remains **passed with qualification**. All original output, source and pinned decision hashes were verified unchanged.

See [protocol](../NEURAL_PROTOCOL.md), [per-permutation results](per_permutation.csv), and [summary and hashes](summary.json). Per-model checkpoints, run records and label-free test predictions are retained in each permutation directory.
'''
    atomic_write(OUTPUT / 'report.md', report.encode())
    atomic_json(OUTPUT / 'progress.json', {'completed_permutations': 200, 'total': 200, 'status': 'complete'})
    print(verdict, interval, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='validate inputs without starting training')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.check_only)
