"""Post-hoc exploratory Part A: DINO layer variants under the Phase 1/2 ridge protocol.

Predeclared in DECISIONS.md (2026-10-31, commit 448efe2). Run from the repository root:
    .venv/bin/python -m results.posthoc_dino.part_a
Writes only under results/posthoc_dino/. Unadjusted, exploratory.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.baselines import fit_ridge, fit_ridge_fixed_alpha, predict_ridge
from src.cache import atomic_json, atomic_write, csv_bytes, index_manifest, read_branch, read_manifest
from src.features import sha256
from src.metrics import (bootstrap_indices_hash, classify_difference, correlations, fit_logistic_plcc,
                         make_bootstrap_indices, paired_bootstrap)

OUT = Path('results/posthoc_dino')
CACHE = Path('cache/v2')
KONVID, LIVE, SPLITS = Path('data/manifests/konvid.csv'), Path('data/manifests/livevqc.csv'), Path('data/splits/konvid_folds.csv')
ALPHAS = [.001, .01, .1, 1, 10, 100, 1000]
VARIANTS = {
    'A0': ['dino_cls', 'dino_patchmean_L12'],
    'L6': ['dino_patchmean_L6', 'dino_patchstd_L6'],
    'L9': ['dino_patchmean_L9', 'dino_patchstd_L9'],
    'L12': ['dino_patchmean_L12', 'dino_patchstd_L12'],
    'ALL': ['dino_cls', 'dino_patchmean_L6', 'dino_patchstd_L6', 'dino_patchmean_L9', 'dino_patchstd_L9',
            'dino_patchmean_L12', 'dino_patchstd_L12'],
}
COMPARISONS = [(v, 'A0') for v in ('L6', 'L9', 'L12', 'ALL')] + [(v, 'ResNet-50') for v in VARIANTS]
TOLERANCE = 1e-9


def inventory(path):
    return [r for r in csv.DictReader(path.open()) if r['status'] == 'ok']


def pooled(dataset, rows):
    """Per-video mean over frames of the concatenated branches, exactly as the Phase 1/2 ridge."""
    records = index_manifest(read_manifest(CACHE))
    branches = sorted({b for names in VARIANTS.values() for b in names})
    out = {name: {} for name in VARIANTS}
    for row in rows:
        arrays = {b: read_branch(CACHE, dataset, row['video_id'], b, records, row) for b in branches}
        for name, names in VARIANTS.items():
            x = np.concatenate([arrays[b] for b in names], axis=1).astype(np.float32, copy=True)
            out[name][row['video_id']] = x.mean(0, dtype=np.float64)
    return out


def saved_rows(pattern):
    return [row for fold in range(5) for row in json.loads(Path(pattern.format(fold=fold)).read_text())]


def per_fold(rows):
    return {fold: correlations([r['mos'] for r in rows if r['fold'] == fold],
                               [r['prediction'] for r in rows if r['fold'] == fold]) for fold in range(5)}


def summary(scores):
    srcc, krcc = ([scores[f][m] for f in range(5)] for m in ('srcc', 'krcc'))
    return dict(srcc_mean=float(np.mean(srcc)), srcc_sd=float(np.std(srcc, ddof=1)),
                krcc_mean=float(np.mean(krcc)), krcc_sd=float(np.std(krcc, ddof=1)))


def main():
    konvid, live = inventory(KONVID), inventory(LIVE)
    mos = {r['video_id']: float(r['mos']) for r in konvid}
    splits = list(csv.DictReader(SPLITS.open()))
    features = pooled('konvid', konvid)
    phase1 = saved_rows('results/phase1/ridge/fold{fold}/predictions.json')
    resnet = saved_rows('results/phase2/resnet50/ridge/fold{fold}/predictions.json')
    saved_a0 = per_fold(phase1)

    fold_rows, oof, alphas = [], {name: {} for name in VARIANTS}, {name: [] for name in VARIANTS}
    record = dict(status='running', label='post-hoc exploratory, unadjusted', predeclaration_commit='448efe2',
                  script_sha256=sha256(__file__),
                  inputs_sha256={str(p): sha256(p) for p in (KONVID, LIVE, SPLITS, CACHE / 'manifest.csv', CACHE / 'meta.json')})
    for name in VARIANTS:  # A0 first: its reproduction check gates every other variant.
        for fold in range(5):
            ids = {role: [r['video_id'] for r in splits if int(r['outer_fold']) == fold and r['role'] == role]
                   for role in ('train', 'val', 'test')}
            X = {role: np.array([features[name][k] for k in keys]) for role, keys in ids.items()}
            y = {role: np.array([mos[k] for k in keys]) for role, keys in ids.items()}
            artifact = fit_ridge(X['train'], y['train'], X['val'], y['val'], ALPHAS)
            values = predict_ridge(artifact, X['test'])
            score = correlations(y['test'], values)
            alphas[name].append(artifact['alpha'])
            fold_rows.append(dict(variant=name, fold=fold, alpha=artifact['alpha'],
                                  alpha_at_grid_edge=artifact['alpha'] in (ALPHAS[0], ALPHAS[-1]),
                                  validation_srcc=artifact['validation_srcc'], test_srcc=score['srcc'],
                                  test_krcc=score['krcc'], dimensions=X['train'].shape[1]))
            oof[name].update(zip(ids['test'], map(float, values)))
        if name == 'A0':
            differences = [abs(fold_rows[f]['test_srcc'] - saved_a0[f]['srcc']) for f in range(5)]
            record['a0_reproduction'] = dict(per_fold_absolute_difference=differences, tolerance=TOLERANCE,
                                             saved_per_fold_srcc=[saved_a0[f]['srcc'] for f in range(5)],
                                             passed=max(differences) <= TOLERANCE)
            if not record['a0_reproduction']['passed']:
                record['status'] = 'stopped: A0 does not reproduce the saved Phase 1/2 DINO ridge'
                atomic_write(OUT / 'per_fold.csv', csv_bytes(fold_rows, list(fold_rows[0])))
                atomic_json(OUT / 'part_a.json', record)
                print(json.dumps(record['a0_reproduction']))
                return 1

    scores = {name: {r['fold']: dict(srcc=r['test_srcc'], krcc=r['test_krcc']) for r in fold_rows if r['variant'] == name}
              for name in VARIANTS}
    table = [dict(row='Phase 2 ResNet-50 ridge (saved)', **summary(per_fold(resnet))),
             dict(row='Phase 2 DINO ridge (saved)', **summary(saved_a0))]
    table += [dict(row=name, **summary(scores[name]), median_alpha=float(np.median(alphas[name])),
                   grid_edge_folds=sum(r['alpha_at_grid_edge'] for r in fold_rows if r['variant'] == name))
              for name in VARIANTS]
    a0, gap = table[2]['srcc_mean'], table[0]['srcc_mean'] - table[2]['srcc_mean']
    for row in table[2:]:
        row['gap_recovered_fraction'] = (row['srcc_mean'] - a0) / gap
        row['closes_most_of_gap'] = row['gap_recovered_fraction'] > .5

    # Paired bootstrap over pooled OOF predictions; one shared seed-0 bank in inventory order.
    order = [r['video_id'] for r in konvid]
    resnet_oof = {r['video_id']: r['prediction'] for r in resnet}
    for name, values in (*oof.items(), ('ResNet-50', resnet_oof)):
        if sorted(values) != sorted(order):
            raise ValueError(f'{name}: OOF predictions do not cover the inventory exactly once')
    preds = {name: np.array([oof[name][k] for k in order]) for name in VARIANTS}
    preds['ResNet-50'] = np.array([resnet_oof[k] for k in order])
    y = np.array([mos[k] for k in order])
    bank = make_bootstrap_indices(len(order), 0, 10000)
    comparisons = []
    for a, b in COMPARISONS:
        for metric in ('srcc', 'krcc'):
            result = paired_bootstrap(y, preds[a], preds[b], metric, 0, 10000, resample_indices=bank)
            comparisons.append(dict(a=a, b=b, metric=metric, delta=result['delta'], ci95_low=result['ci95'][0],
                                    ci95_high=result['ci95'][1], p_raw=result['p_raw'], valid_draws=result['valid_draws'],
                                    classification=classify_difference(result['ci95']), adjusted=False, label='exploratory'))
            print(json.dumps(comparisons[-1]), flush=True)

    # LIVE-VQC, exploratory: full-KoNViD fit at the median CV-selected alpha; feeds back into nothing.
    live_features = pooled('livevqc', live)
    live_mos = np.array([float(r['mos']) for r in live])
    live_rows, live_predictions = [], []
    for name in VARIANTS:
        alpha = float(np.median(alphas[name]))
        artifact = fit_ridge_fixed_alpha(np.array([features[name][k] for k in order]), y, alpha)
        values = predict_ridge(artifact, np.array([live_features[name][r['video_id']] for r in live]))
        score, mapped = correlations(live_mos, values), fit_logistic_plcc(live_mos, values)
        live_rows.append(dict(variant=name, alpha=alpha, videos=len(live), srcc=score['srcc'], krcc=score['krcc'],
                              plcc_mapped=mapped['plcc_mapped'], mapping_status=mapped['status']))
        live_predictions += [dict(variant=name, video_id=r['video_id'], prediction=float(v)) for r, v in zip(live, values)]

    oof_rows = [dict(video_id=k, mos=mos[k], **{name: float(preds[name][i]) for name in preds}) for i, k in enumerate(order)]
    outputs = {'per_fold.csv': fold_rows, 'summary.csv': table, 'comparisons.csv': comparisons, 'live.csv': live_rows,
               'oof_predictions.csv': oof_rows, 'live_predictions.csv': live_predictions}
    for filename, rows in outputs.items():
        fields = list(dict.fromkeys(k for r in rows for k in r))
        atomic_write(OUT / filename, csv_bytes([{k: r.get(k, '') for k in fields} for r in rows], fields))
    record.update(status='complete', gap_srcc=gap, summary=table, comparisons=comparisons, live=live_rows, folds=fold_rows,
                  bootstrap_indices_sha256=bootstrap_indices_hash(bank),
                  outputs_sha256={f: sha256(OUT / f) for f in outputs})
    atomic_json(OUT / 'part_a.json', record)
    print(json.dumps(dict(status='complete', a0_reproduction=record['a0_reproduction'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
