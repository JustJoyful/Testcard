"""Generate report.md and SVG figures for the post-hoc DINO analysis from saved outputs only.

    .venv/bin/python -m results.posthoc_dino.report
"""
import csv
import json
from html import escape
from pathlib import Path

from src.cache import atomic_write

OUT = Path('results/posthoc_dino')
COLORS = ['#1b6ca8', '#d1495b', '#edae49', '#00798c', '#66a182', '#8e6c8a', '#30362f']


def rows(name):
    return list(csv.DictReader((OUT / name).open()))


def f(value, digits=3):
    return f'{float(value):.{digits}f}'


def svg(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="sans-serif" font-size="12"><rect width="100%" height="100%" fill="white"/>{body}</svg>\n')


def part_a_figure(summary):
    """Mean ± SD of per-fold SRCC per row, as a horizontal dot plot."""
    lo, hi, left, width, step = .65, .86, 230, 420, 30
    x = lambda v: left + (float(v) - lo) / (hi - lo) * width
    body = ['<text x="10" y="20" font-weight="bold">KoNViD per-fold SRCC, mean ± SD (post-hoc, exploratory)</text>']
    for i, row in enumerate(summary):
        y = 50 + i * step
        mean, sd = float(row['srcc_mean']), float(row['srcc_sd'])
        body += [f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{escape(row["row"])}</text>',
                 f'<line x1="{x(mean - sd)}" x2="{x(mean + sd)}" y1="{y}" y2="{y}" stroke="#555"/>',
                 f'<circle cx="{x(mean)}" cy="{y}" r="4" fill="{COLORS[i % len(COLORS)]}"/>',
                 f'<text x="{x(mean + sd) + 6}" y="{y + 4}" fill="#555">{f(mean)}</text>']
    axis = 50 + len(summary) * step
    body.append(f'<line x1="{left}" x2="{left + width}" y1="{axis}" y2="{axis}" stroke="#000"/>')
    for tick in (.65, .70, .75, .80, .85):
        body.append(f'<text x="{x(tick)}" y="{axis + 16}" text-anchor="middle">{tick:.2f}</text>')
    return svg(left + width + 80, axis + 30, ''.join(body))


def part_b_figure(results):
    """Ratio vs distortion level; one panel per distortion, one line per branch."""
    kinds, branches = ['blur', 'noise', 'jpeg'], list(dict.fromkeys(r['branch'] for r in results))
    labels = {'blur': 'Gaussian blur σ', 'noise': 'Gaussian noise σ', 'jpeg': 'JPEG quality'}
    top = max(float(r['ratio']) for r in results) * 1.1
    pw, ph, gap, left = 240, 200, 50, 50
    body = ['<text x="10" y="20" font-weight="bold">Distortion-to-content ratio (higher = more quality-sensitive)</text>']
    for p, kind in enumerate(kinds):
        ox, oy = left + p * (pw + gap), 40
        levels = [r['level'] for r in results if r['distortion'] == kind and r['branch'] == branches[0]]
        x = lambda i: ox + 20 + i * (pw - 40) / (len(levels) - 1)
        y = lambda v: oy + ph - float(v) / top * ph
        body += [f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>',
                 f'<text x="{ox + pw / 2}" y="{oy + ph + 32}" text-anchor="middle">{labels[kind]}</text>']
        body += [f'<text x="{x(i)}" y="{oy + ph + 14}" text-anchor="middle">{escape(level)}</text>' for i, level in enumerate(levels)]
        for b, branch in enumerate(branches):
            points = [(x(i), y(r['ratio'])) for i, r in enumerate(r for r in results if r['distortion'] == kind and r['branch'] == branch)]
            body.append(f'<polyline fill="none" stroke="{COLORS[b]}" stroke-width="2" points="{" ".join(f"{a:.1f},{c:.1f}" for a, c in points)}"/>')
            body += [f'<circle cx="{a:.1f}" cy="{c:.1f}" r="3" fill="{COLORS[b]}"/>' for a, c in points]
    for tick in range(5):
        value = top * tick / 4
        body.append(f'<text x="{left - 6}" y="{40 + ph - value / top * ph + 4}" text-anchor="end">{value:.2f}</text>')
    legend_y = 40 + ph + 55
    for b, branch in enumerate(branches):
        body += [f'<rect x="{left + b * 170}" y="{legend_y - 9}" width="12" height="12" fill="{COLORS[b]}"/>',
                 f'<text x="{left + b * 170 + 18}" y="{legend_y + 1}">{escape(branch)}</text>']
    return svg(left + 3 * (pw + gap), legend_y + 20, ''.join(body))


def main():
    record = json.loads((OUT / 'part_a.json').read_text())
    summary, comparisons, live, folds = rows('summary.csv'), rows('comparisons.csv'), rows('live.csv'), rows('per_fold.csv')
    atomic_write(OUT / 'part_a_srcc.svg', part_a_figure(summary).encode())
    reproduction = record['a0_reproduction']
    edges = sum(r['alpha_at_grid_edge'] == 'True' for r in folds)
    by_row = {r['row']: r for r in summary}
    closers = [name for name in ('L6', 'L9', 'ALL') if by_row[name]['closes_most_of_gap'] == 'True']
    lines = ['# Post-hoc exploratory analysis: why did DINOv2 lose to ResNet-50 on KoNViD?', '',
             '**Post-hoc, exploratory and unadjusted.** Predeclared in DECISIONS.md on 2026-10-31 (commit '
             f"`{record['predeclaration_commit']}`) before any computation. Nothing here changes a frozen configuration, "
             'deployed model, phase gate or headline result. Every number below is generated from the CSV/JSON files in this folder.', '',
             '## Part A: DINO layer variants (ridge, cached features)', '',
             f"A0 reproduction check: maximum per-fold absolute SRCC difference from the saved Phase 1/2 DINO ridge "
             f"{max(reproduction['per_fold_absolute_difference']):.3g} (tolerance {reproduction['tolerance']:g}): "
             f"**{'passed' if reproduction['passed'] else 'failed'}**.", '',
             '| Row | SRCC mean ± SD | KRCC mean ± SD | Median alpha | Folds at grid edge | Gap recovered |',
             '|---|---|---|---|---|---|']
    for r in summary:
        recovered = f"{float(r['gap_recovered_fraction']):.0%}" if r.get('gap_recovered_fraction') else ''
        lines.append(f"| {r['row']} | {f(r['srcc_mean'])} ± {f(r['srcc_sd'])} | {f(r['krcc_mean'])} ± {f(r['krcc_sd'])} | "
                     f"{r.get('median_alpha', '')} | {r.get('grid_edge_folds', '')} | {recovered} |")
    lines += ['', f"Gap (saved ResNet-50 ridge minus A0, mean per-fold SRCC): {record['gap_srcc']:.4f}. "
              '"Closes most of the gap" means recovering more than 50% of it (predeclared).', '',
              '![Per-fold SRCC](part_a_srcc.svg)', '',
              '### Paired bootstrap of pooled out-of-fold predictions (10,000 draws, seed 0, one shared bank; unadjusted)', '',
              '| A − B | Metric | Δ | 95% CI | PRD §7 classification |', '|---|---|---|---|---|']
    lines += [f"| {r['a']} − {r['b']} | {r['metric'].upper()} | {float(r['delta']):+.4f} | "
              f"[{float(r['ci95_low']):+.4f}, {float(r['ci95_high']):+.4f}] | {r['classification']} |" for r in comparisons]
    lines += ['', '### LIVE-VQC (exploratory; full-KoNViD fit at the median CV alpha; feeds back into nothing)', '',
              '| Variant | Alpha | Videos | SRCC | KRCC | PLCC (logistic-mapped) |', '|---|---|---|---|---|---|']
    lines += [f"| {r['variant']} | {r['alpha']} | {r['videos']} | {f(r['srcc'])} | {f(r['krcc'])} | {f(r['plcc_mapped'])} |" for r in live]

    lines += ['', '## Part B: distortion sensitivity', '']
    part_b = OUT / 'part_b.csv'
    part_b_supports = None
    if part_b.is_file():
        results = rows('part_b.csv')
        atomic_write(OUT / 'part_b_ratio.svg', part_b_figure(results).encode())
        meta = json.loads((OUT / 'part_b.json').read_text())
        means = {}
        for r in results:
            means.setdefault(r['branch'], []).append(float(r['ratio']))
        means = {k: sum(v) / len(v) for k, v in means.items()}
        lines += [f"{len(meta['frames'])} frames from the 20 fixed Phase 0 KoNViD clips, device {meta['device']}. "
                  'Ratio = mean (1 − cosine) clean vs distorted, divided by mean (1 − cosine) between different clean frames.', '',
                  '| Branch | Distortion | Level | Ratio |', '|---|---|---|---|']
        lines += [f"| {r['branch']} | {r['distortion']} | {r['level']} | {f(r['ratio'])} |" for r in results]
        lines += ['', '| Branch | Mean ratio over all 9 levels |', '|---|---|']
        lines += [f'| {k} | {f(v)} |' for k, v in sorted(means.items(), key=lambda kv: -kv[1])]
        part_b_supports = (max(means['dino_patchmean_L6'], means['dino_patchmean_L9'])
                           > max(means['dino_patchmean_L12'], means['dino_cls_L12']))
        lines += ['', '![Distortion ratios](part_b_ratio.svg)', '']
    else:
        lines += ['Not run yet (GPU). Part A conclusions below stand on their own until it is.', '']

    lines += ['## Interpretation (against the predeclared question)', '']
    if closers:
        lines.append(f"Part A: {', '.join(closers)} recover more than 50% of the A0→ResNet-50 KoNViD gap, so the predeclared "
                     'rule reads as **quality information present in earlier DINO layers and lost at the last layer**.'
                     + (' The block-12 patch-statistics variant (L12) does not, so the loss is not specific to the CLS token.'
                        if by_row['L12']['closes_most_of_gap'] == 'False' else ''))
    else:
        lines.append('Part A: no earlier-layer variant recovers more than 50% of the gap, so the predeclared rule reads the deficit as **model-wide**.')
    if part_b_supports is None:
        lines.append('Part B: pending.')
    else:
        agrees = part_b_supports == bool(closers)
        lines.append(f"Part B {'supports' if agrees else 'contradicts'} Part A: the most distortion-sensitive earlier-layer DINO "
                     f"patch-mean branch has a {'higher' if part_b_supports else 'lower or equal'} mean ratio than the most "
                     'sensitive block-12 DINO branch (this summary rule was fixed after Part A results, before Part B ran).'
                     + ('' if agrees else ' The evidence is therefore **mixed**.'))
        # Descriptive checks added after Part B ran; they qualify the verdict, they do not replace it.
        monotone = means['dino_patchmean_L6'] > means['dino_patchmean_L9'] > means['dino_patchmean_L12']
        l9_vs_cls = means['dino_patchmean_L9'] > means['dino_cls_L12']
        resnet_top = means['resnet50'] > max(v for k, v in means.items() if k.startswith('dino'))
        caveats = []
        if not l9_vs_cls or not monotone:
            caveats.append('sensitivity does not fall monotonically with depth (patch-mean L9 is '
                           f"{'above' if l9_vs_cls else 'below'} CLS L12)")
        if resnet_top and closers:
            caveats.append('ResNet-50 is more distortion-sensitive than every DINO branch, yet the best earlier-layer DINO '
                           'variants outscore it in Part A, so raw distortion sensitivity alone does not explain the Part A ranking')
        if caveats:
            lines.append('Descriptive checks added after Part B ran (not part of any predeclared rule): ' + '; '.join(caveats)
                         + '. Overall the evidence is **mixed**: Part B is consistent with information loss at block 12 '
                         'but does not by itself establish it.')
    lines += ['', '## Limitations', '',
              '- Post-hoc and exploratory: chosen after the Phase 1–3 results; five variants and nine unadjusted comparisons per metric.',
              f"- Alpha grid edge: {edges} of {len(folds)} variant-folds selected the largest grid alpha (1000). As predeclared, the grid "
              'was not extended, so every variant (including the original A0) may be under-regularized and the differences could change with a wider grid.',
              '- The bootstrap captures test-set sampling only, not fold or training variance; LIVE-VQC rows are single fits without intervals.',
              '- Nothing here changes the deployed ResNet-50 ridge model or any phase result.', '']
    atomic_write(OUT / 'report.md', '\n'.join(lines).encode())
    print('report written; part B', 'included' if part_b.is_file() else 'pending')


if __name__ == '__main__':
    main()
