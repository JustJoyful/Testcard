"""Post-hoc diagnostics; run from the repository root with .venv/bin/python."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.baselines import fit_ridge_fixed_alpha, predict_ridge
from src.cache import atomic_json
from src.data import CachedFeatureDataset
from src.features import sha256
from src.phase3 import source_code_sha256
from src.tuning import permute_control_role


def main():
    root = Path('results/phase3')
    output = root / 'posthoc_permutation'
    gate_path = root / 'phase_gate.json'
    gate = json.loads(gate_path.read_text())
    protected = [gate_path, Path('DECISIONS.md'), Path('results/phase2/konvid_freeze.json')]
    protected += [root / name for name in gate['outputs_sha256']]
    before = {str(p): sha256(p) for p in protected}
    code_before = source_code_sha256()
    assert gate['status'] == 'passed with qualification' and gate['control_status'] == 'failed'
    freeze = json.loads(Path('results/phase2/konvid_freeze.json').read_text())
    config = freeze['refit_by_feature']['resnet50']['mean_pool']['config']
    alpha = freeze['refit_by_feature']['resnet50']['ridge']['phase3_alpha']
    datasets = {role: CachedFeatureDataset(Path(config['cache_root']), Path(config['inventory']),
                Path(config['splits']), 0, role, 'resnet50') for role in ('train', 'val', 'test')}
    ids = {role: data.ids for role, data in datasets.items()}
    assert not (set(ids['train']) & set(ids['val']) or set(ids['train']) & set(ids['test'])
                or set(ids['val']) & set(ids['test']))
    saved_mapping = json.loads((root / 'control_fixed_epoch/permutations.json').read_text())
    labels, features, shuffled = {}, {}, {}
    label_scores = {}
    for role, data in datasets.items():
        labels[role] = np.array([item['mos'] for item in data.items])
        features[role] = np.array([item['features'].numpy().mean(0, dtype=np.float64) for item in data.items])
        if role != 'test':
            items, mapping = permute_control_role(data, role)
            assert mapping == saved_mapping[role], 'historical permutation differs'
            shuffled[role] = np.array([item['mos'] for item in items])
            label_scores[role] = float(spearmanr(labels[role], shuffled[role]).statistic)
    x = np.concatenate([features['train'], features['val']])
    y = np.concatenate([labels['train'], labels['val']])
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    test_z = (features['test'] - mean) / scale
    # Factor once: identical fixed features/alpha for every permuted target vector.
    factor = cho_factor(z @ z.T + alpha * np.eye(len(x)))
    targets = []
    for seed in range(1000):
        targets.append(np.concatenate([labels[role][np.random.default_rng(seed).permutation(len(labels[role]))]
                                       for role in ('train', 'val')]))
    targets = np.array(targets).T
    assert np.array_equal(targets[:, 0], np.concatenate([shuffled['train'], shuffled['val']]))
    target_std = y.std()
    predictions = (test_z @ z.T @ cho_solve(factor, (targets - y.mean()) / target_std)) * target_std + y.mean()
    assert np.isfinite(predictions).all()
    # Check the batched solve against the shared implementation on distinct targets.
    for seed in (0, 999):
        expected = predict_ridge(fit_ridge_fixed_alpha(x, targets[:, seed], alpha), features['test'])
        np.testing.assert_allclose(predictions[:, seed], expected, rtol=1e-9, atol=1e-9)
    scores = np.array([spearmanr(labels['test'], p).statistic for p in predictions.T])
    assert np.isfinite(scores).all()
    reference = json.loads((root / 'control_fixed_epoch/investigation.json').read_text())['phase1_phase2_comparable_trained_ensemble']['srcc']
    lower, upper = np.quantile(scores, [0.025, 0.975])
    rows = [{'permutation_seed': seed, 'test_srcc': float(score)} for seed, score in enumerate(scores)]
    with (output / 'ridge_null.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'historical_permutation_ridge_predictions.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_id', 'mos', 'prediction'])
        writer.writerows(zip(ids['test'], labels['test'], predictions[:, 0], strict=True))
    result = {
        'status': 'post_hoc_diagnostic_only', 'date': '2026-10-29',
        'protocol': 'PROTOCOL.md', 'alpha': alpha, 'feature_set': 'resnet50', 'fold': 0,
        'roles': {role: {'count': len(values), 'ids': values} for role, values in ids.items()},
        'historical_permuted_vs_true_label_srcc': label_scores,
        'null': {'permutations': len(scores), 'seeds': [0, 999], 'mean': float(scores.mean()),
                 'sd': float(scores.std(ddof=1)), 'central_95_percent_interval': [float(lower), float(upper)],
                 'historical_permutation_ridge_srcc': float(scores[0]),
                 'fraction_abs_srcc_at_least_0_15': float(np.mean(np.abs(scores) >= .15)),
                 'neural_ensemble_reference_srcc': reference,
                 'reference_inside_central_95_percent_interval': bool(lower <= reference <= upper),
                 'fraction_at_or_below_reference': float(np.mean(scores <= reference)),
                 'fraction_abs_at_least_reference': float(np.mean(np.abs(scores) >= abs(reference)))},
        'limitations': 'Ridge is a surrogate, not the three-epoch neural mean-pool procedure. Tail fractions are descriptive, not calibrated neural-control p-values. This does not establish the cause or rule out leakage.',
        'source_code_sha256': code_before, 'script_sha256': sha256(Path(__file__)),
        'protected_inputs_sha256': before,
        'data_inputs_sha256': {str(p): sha256(p) for p in map(Path, [config['inventory'], config['splits'], 'cache/v2/meta.json', 'configs/control_v2.yaml'])},
        'outputs_sha256': {name: sha256(output / name) for name in ('ridge_null.csv', 'historical_permutation_ridge_predictions.csv')},
    }
    assert before == {str(p): sha256(p) for p in protected}, 'protected evidence changed'
    assert code_before == source_code_sha256(), 'gate-bound code changed'
    atomic_json(output / 'summary.json', result)
    report = f"""# Post-hoc permutation diagnostics

Date: {result['date']}. Diagnostic only; the original control remains failed and Phase 3 remains passed with qualification.

The historical permuted-versus-true label SRCC is {label_scores['train']:.6f} for training and {label_scores['val']:.6f} for validation. These correlations alone cannot measure alignment with a quality-related feature direction.

The ridge null uses {len(scores)} predetermined permutation seeds, independently within train and validation, and fits their combined rows as in the fixed-epoch control. The frozen ridge alpha is {alpha:g}; no alpha selection uses permuted validation or test scores. Feature standardization uses fitting rows only. Test labels are used only for evaluation.

The surrogate null has mean {scores.mean():.6f}, SD {scores.std(ddof=1):.6f}, and central 95% interval [{lower:.6f}, {upper:.6f}]. This is a permutation interval, not a bootstrap confidence interval. The historical permutation produces ridge test SRCC {scores[0]:.6f}.

The neural ensemble reference SRCC is {reference:.6f}; it {'falls inside' if lower <= reference <= upper else 'falls outside'} this surrogate interval. The fraction of ridge permutations at or below it is {np.mean(scores <= reference):.3f}; the fraction with at least its absolute magnitude is {np.mean(np.abs(scores) >= abs(reference)):.3f}. These are descriptive cross-model comparisons, not neural permutation p-values. The fraction with |SRCC| at least 0.15 is {np.mean(np.abs(scores) >= .15):.3f}; bootstrap-CI inclusion of zero was not tested here, so this is not the full original-rule failure rate.

{result['limitations']} A matched neural permutation null would be needed to assess the failed neural statistic directly. Consistency of signs across model seeds sharing one permutation does not establish a systematic defect across permutations.

All protected gate outputs, decision text and source-code hashes remained unchanged. The batched calculation matched the shared ridge implementation on the first and last permutation. See [summary.json](summary.json), [per-permutation scores](ridge_null.csv), and [historical-permutation test predictions](historical_permutation_ridge_predictions.csv).
"""
    (output / 'report.md').write_text(report)
    print(json.dumps({'label_srcc': label_scores, 'null': result['null']}, indent=2))


if __name__ == '__main__':
    main()
