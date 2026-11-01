"""Post-hoc exploratory alpha-grid follow-up, predeclared in DECISIONS.md (2026-11-01, commit fbe046c).

    .venv/bin/python -m results.posthoc_dino.grid_followup.run

Reruns A0, L6, L9, L12, ALL and the ResNet-50 ridge with the extended alpha grid under the Phase 1/2 ridge protocol.
Writes only under results/posthoc_dino/grid_followup/. Unadjusted, exploratory; feeds back into nothing.
"""
import csv
import json
from pathlib import Path

import numpy as np

from results.posthoc_dino.part_a import CACHE, KONVID, LIVE, SPLITS, inventory, summary
from results.posthoc_dino.report import part_a_figure
from src.baselines import fit_ridge, fit_ridge_fixed_alpha, predict_ridge
from src.cache import atomic_json, atomic_write, csv_bytes, index_manifest, read_branch, read_manifest
from src.features import sha256
from src.metrics import (bootstrap_indices_hash, classify_difference, correlations, fit_logistic_plcc,
                         make_bootstrap_indices, paired_bootstrap)

OUT = Path('results/posthoc_dino/grid_followup')
ALPHAS = [.001, .01, .1, 1, 10, 100, 1000, 1e4, 1e5, 1e6]
ORIGINAL_TOP = 1000
ROWS = {
    'A0': ['dino_cls', 'dino_patchmean_L12'],
    'L6': ['dino_patchmean_L6', 'dino_patchstd_L6'],
    'L9': ['dino_patchmean_L9', 'dino_patchstd_L9'],
    'L12': ['dino_patchmean_L12', 'dino_patchstd_L12'],
    'ALL': ['dino_cls', 'dino_patchmean_L6', 'dino_patchstd_L6', 'dino_patchmean_L9', 'dino_patchstd_L9',
            'dino_patchmean_L12', 'dino_patchstd_L12'],
    'ResNet-50': ['resnet50'],
}
COMPARISONS = [(v, 'A0') for v in ('L6', 'L9', 'L12', 'ALL')] + [(v, 'ResNet-50') for v in ('A0', 'L6', 'L9', 'L12', 'ALL')]


def pooled(dataset, rows):
    records = index_manifest(read_manifest(CACHE))
    branches = sorted({b for names in ROWS.values() for b in names})
    out = {name: {} for name in ROWS}
    for row in rows:
        arrays = {b: read_branch(CACHE, dataset, row['video_id'], b, records, row) for b in branches}
        for name, names in ROWS.items():
            out[name][row['video_id']] = np.concatenate([arrays[b] for b in names], 1).astype(np.float32).mean(0, dtype=np.float64)
    return out


def original_alphas(pattern):
    return [json.loads(Path(pattern.format(fold=f)).read_text())['alpha'] for f in range(5)]


def main():
    konvid, live = inventory(KONVID), inventory(LIVE)
    mos = {r['video_id']: float(r['mos']) for r in konvid}
    splits = list(csv.DictReader(SPLITS.open()))
    features = pooled('konvid', konvid)
    fold_rows, oof, alphas = [], {n: {} for n in ROWS}, {n: [] for n in ROWS}
    for name in ROWS:
        for fold in range(5):
            ids = {role: [r['video_id'] for r in splits if int(r['outer_fold']) == fold and r['role'] == role]
                   for role in ('train', 'val', 'test')}
            X = {role: np.array([features[name][k] for k in keys]) for role, keys in ids.items()}
            y = {role: np.array([mos[k] for k in keys]) for role, keys in ids.items()}
            artifact = fit_ridge(X['train'], y['train'], X['val'], y['val'], ALPHAS)
            values = predict_ridge(artifact, X['test'])
            score = correlations(y['test'], values)
            alphas[name].append(artifact['alpha'])
            fold_rows.append(dict(row=name, fold=fold, alpha=artifact['alpha'], alpha_at_grid_edge=artifact['alpha'] == ALPHAS[-1],
                                  above_original_grid=artifact['alpha'] > ORIGINAL_TOP, validation_srcc=artifact['validation_srcc'],
                                  test_srcc=score['srcc'], test_krcc=score['krcc']))
            oof[name].update(zip(ids['test'], map(float, values)))
        print(json.dumps(dict(row=name, alphas=alphas[name])), flush=True)

    table = []
    for name in ROWS:
        rows = [r for r in fold_rows if r['row'] == name]
        table.append(dict(row=name, **summary({r['fold']: dict(srcc=r['test_srcc'], krcc=r['test_krcc']) for r in rows}),
                          median_alpha=float(np.median(alphas[name])), alphas=' '.join(f'{a:g}' for a in alphas[name]),
                          grid_edge_folds=sum(r['alpha_at_grid_edge'] for r in rows)))

    order = [r['video_id'] for r in konvid]
    preds = {name: np.array([oof[name][k] for k in order]) for name in ROWS}
    y = np.array([mos[k] for k in order])
    bank = make_bootstrap_indices(len(order), 0, 10000)
    comparisons = []
    for a, b in COMPARISONS:
        for metric in ('srcc', 'krcc'):
            result = paired_bootstrap(y, preds[a], preds[b], metric, 0, 10000, resample_indices=bank)
            comparisons.append(dict(a=a, b=b, metric=metric, delta=result['delta'], ci95_low=result['ci95'][0],
                                    ci95_high=result['ci95'][1], p_raw=result['p_raw'], valid_draws=result['valid_draws'],
                                    classification=classify_difference(result['ci95']), adjusted=False, label='exploratory'))

    live_features = pooled('livevqc', live)
    live_mos = np.array([float(r['mos']) for r in live])
    live_rows = []
    for name in ROWS:
        alpha = float(np.median(alphas[name]))
        artifact = fit_ridge_fixed_alpha(np.array([features[name][k] for k in order]), y, alpha)
        values = predict_ridge(artifact, np.array([live_features[name][r['video_id']] for r in live]))
        score, mapped = correlations(live_mos, values), fit_logistic_plcc(live_mos, values)
        live_rows.append(dict(row=name, alpha=alpha, videos=len(live), srcc=score['srcc'], krcc=score['krcc'],
                              plcc_mapped=mapped['plcc_mapped'], mapping_status=mapped['status']))

    original = dict(resnet50=original_alphas('results/phase2/resnet50/ridge/fold{fold}/run.json'),
                    dino_a0=original_alphas('results/phase1/ridge/fold{fold}/run.json'))
    l6 = next(c for c in comparisons if (c['a'], c['b'], c['metric']) == ('L6', 'ResNet-50', 'srcc'))
    excludes_zero = l6['ci95_low'] > 0 or l6['ci95_high'] < 0
    worst_edge = max(r['grid_edge_folds'] for r in table)
    verdict = dict(robust=excludes_zero and worst_edge <= 1, l6_minus_resnet_srcc_ci95=[l6['ci95_low'], l6['ci95_high']],
                   ci_excludes_zero=excludes_zero, max_folds_at_largest_grid_value=worst_edge, edge_condition_met=worst_edge <= 1,
                   original_resnet50_alphas=original['resnet50'],
                   original_resnet50_at_original_grid_edge=sum(a == ORIGINAL_TOP for a in original['resnet50']),
                   original_dino_a0_alphas=original['dino_a0'])

    oof_rows = [dict(video_id=k, mos=mos[k], **{n: float(preds[n][i]) for n in preds}) for i, k in enumerate(order)]
    outputs = {'per_fold.csv': fold_rows, 'summary.csv': table, 'comparisons.csv': comparisons, 'live.csv': live_rows,
               'oof_predictions.csv': oof_rows}
    for filename, rows in outputs.items():
        atomic_write(OUT / filename, csv_bytes(rows, list(dict.fromkeys(k for r in rows for k in r))))
    atomic_json(OUT / 'grid_followup.json', dict(status='complete', label='post-hoc exploratory, unadjusted', predeclaration_commit='fbe046c',
        alpha_grid=ALPHAS, script_sha256=sha256(__file__), bootstrap_indices_sha256=bootstrap_indices_hash(bank), verdict=verdict,
        summary=table, comparisons=comparisons, live=live_rows, outputs_sha256={f: sha256(OUT / f) for f in outputs}))
    write_report()
    print(json.dumps(verdict))


def write_report():
    """Generate report.md and the figure from the saved outputs only."""
    record = json.loads((OUT / 'grid_followup.json').read_text())
    table, comparisons, live, verdict = record['summary'], record['comparisons'], record['live'], record['verdict']
    atomic_write(OUT / 'srcc.svg', part_a_figure([dict(row=r['row'], srcc_mean=r['srcc_mean'], srcc_sd=r['srcc_sd']) for r in table]).encode())
    f = lambda v: f'{float(v):.3f}'
    ci = verdict['l6_minus_resnet_srcc_ci95']
    lines = ['# Post-hoc exploratory follow-up: DINO layers with an extended ridge alpha grid', '',
             '**Post-hoc, exploratory and unadjusted.** Predeclared in DECISIONS.md on 2026-11-01 (commit '
             f"`{record['predeclaration_commit']}`) before any computation. Nothing here changes a frozen configuration, deployed model "
             'or phase result. Every number below is generated from the CSV/JSON files in this folder.', '',
             f"Alpha grid: {', '.join(f'{a:g}' for a in record['alpha_grid'])} (the original grid ended at 1000).", '',
             '## Robustness verdict (predeclared rule)', '',
             f"**The L6 finding is {'robust' if verdict['robust'] else 'NOT robust'}.** "
             f"The L6 − ResNet-50 SRCC 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] {'excludes' if verdict['ci_excludes_zero'] else 'includes'} 0, "
             f"and the most folds any row placed at the largest grid value (1e6) is {verdict['max_folds_at_largest_grid_value']} of 5 "
             f"(rule: at most 1), so the edge condition is {'met' if verdict['edge_condition_met'] else 'not met'}.", '',
             f"Original Phase 2 ResNet-50 ridge alphas by fold: {', '.join(f'{a:g}' for a in verdict['original_resnet50_alphas'])}; "
             f"{verdict['original_resnet50_at_original_grid_edge']} of 5 folds sat at the original grid edge (1000). "
             f"Original Phase 1/2 DINO A0 alphas: {', '.join(f'{a:g}' for a in verdict['original_dino_a0_alphas'])}.", '',
             '## KoNViD per-fold results (extended grid)', '',
             '| Row | SRCC mean ± SD | KRCC mean ± SD | Alphas by fold | Median alpha | Folds at 1e6 |', '|---|---|---|---|---|---|']
    lines += [f"| {r['row']} | {f(r['srcc_mean'])} ± {f(r['srcc_sd'])} | {f(r['krcc_mean'])} ± {f(r['krcc_sd'])} | {r['alphas']} | "
              f"{r['median_alpha']:g} | {r['grid_edge_folds']} |" for r in table]
    lines += ['', '![Per-fold SRCC](srcc.svg)', '',
              '### Paired bootstrap of pooled out-of-fold predictions (10,000 draws, seed 0, one shared bank; unadjusted)', '',
              '| A − B | Metric | Δ | 95% CI | PRD §7 classification |', '|---|---|---|---|---|']
    lines += [f"| {c['a']} − {c['b']} | {c['metric'].upper()} | {c['delta']:+.4f} | [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] | "
              f"{c['classification']} |" for c in comparisons]
    lines += ['', '### LIVE-VQC (exploratory; full-KoNViD fit at the median CV alpha; feeds back into nothing)', '',
              '| Row | Alpha | Videos | SRCC | KRCC | PLCC (logistic-mapped) |', '|---|---|---|---|---|---|']
    lines += [f"| {r['row']} | {r['alpha']:g} | {r['videos']} | {f(r['srcc'])} | {f(r['krcc'])} | {f(r['plcc_mapped'])} |" for r in live]
    lines += ['', '## Limitations', '',
              '- Post-hoc and exploratory: a follow-up chosen after seeing the grid-edge result; nine unadjusted comparisons per metric.',
              '- The bootstrap captures test-set sampling only, not fold or training variance; LIVE-VQC rows are single fits without intervals.',
              '- The ResNet-50 row is a rerun with the extended grid, not the frozen Phase 2 ridge; the deployed model is unchanged.', '']
    atomic_write(OUT / 'report.md', '\n'.join(lines).encode())


if __name__ == '__main__':
    main()
