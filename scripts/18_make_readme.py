"""Generate README.md from README.template.md, filling every number, table and figure from results/ (Rules 15, 22).

    .venv-serving/bin/python -m scripts.18_make_readme            # write README.md and results/figures/*.svg
    .venv-serving/bin/python -m scripts.18_make_readme --check    # regenerate in memory; fail on any difference

Fails if a source file is missing, a placeholder is unresolved, or a filled value is not finite. The SOTA table is
rendered only when every row of results/sota_citations.csv has a value and verified_by_user=yes.
"""
import csv
import difflib
import json
import math
import re
import statistics
import sys
from collections import Counter
from html import escape
from pathlib import Path

import yaml

TEMPLATE, README, FIGURES = Path('README.template.md'), Path('README.md'), Path('results/figures')
HEADS = ('mean_pool', 'bigru', 'transformer', 'mamba')
NAMES = {'mean_pool': 'Mean-pool MLP', 'bigru': 'Bi-GRU', 'transformer': 'Transformer', 'mamba': 'Mamba', 'ridge': 'Ridge (Baseline 0)'}
FEATURES = {'resnet50': 'ResNet-50', 'dino': 'DINOv2 (last layer)'}
# Declared design constants, not results: PRD §7 margin and RULES 10 control threshold.
MARGIN, CONTROL_ABS = 0.01, 0.15


def load(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'missing source file: {path}')
    if path.suffix == '.json':
        return json.loads(path.read_text())
    if path.suffix == '.csv':
        return list(csv.DictReader(path.open()))
    return yaml.safe_load(path.read_text()) if path.suffix in ('.yaml', '.yml') else path.read_text()


def num(value, digits=3):
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f'non-finite value: {value}')
    return f'{value:.{digits}f}'


def pm(mean, sd, digits=3):
    return f'{num(mean, digits)} ± {num(sd, digits)}'


def ci(low, high, digits=3):
    return f'[{float(low):+.{digits}f}, {float(high):+.{digits}f}]'


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def ci_plot(items, title, xlabel, margin=None):
    """Horizontal interval plot as SVG: items are (label, centre, low, high)."""
    bounds = [v for i in items for v in i[2:]] + ([-margin, margin] if margin else [])
    pad = (max(bounds) - min(bounds)) * .08 or .01
    lo, hi = min(bounds) - pad, max(bounds) + pad
    left, width, step = 250, 430, 26
    x = lambda v: left + (v - lo) / (hi - lo) * width
    axis = 50 + len(items) * step
    body = [f'<text x="10" y="20" font-weight="bold">{escape(title)}</text>']
    if margin:
        body.append(f'<rect x="{x(-margin):.1f}" y="35" width="{x(margin) - x(-margin):.1f}" height="{axis - 35}" fill="#eeeeee"/>')
    if lo < 0 < hi:
        body.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="35" y2="{axis}" stroke="#999" stroke-dasharray="3,3"/>')
    for n, (label, centre, low, high) in enumerate(items):
        y = 50 + n * step
        body += [f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{escape(label)}</text>',
                 f'<line x1="{x(low):.1f}" x2="{x(high):.1f}" y1="{y}" y2="{y}" stroke="#333" stroke-width="1.5"/>',
                 f'<circle cx="{x(centre):.1f}" cy="{y}" r="4" fill="#1b6ca8"/>']
    body.append(f'<line x1="{left}" x2="{left + width}" y1="{axis}" y2="{axis}" stroke="#000"/>')
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        body.append(f'<text x="{x(v):.1f}" y="{axis + 15}" text-anchor="middle" font-size="10">{v:.3f}</text>')
    body.append(f'<text x="{left + width / 2}" y="{axis + 32}" text-anchor="middle">{escape(xlabel)}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 30}" height="{axis + 45}" font-family="sans-serif" '
            f'font-size="12"><rect width="100%" height="100%" fill="white"/>{"".join(body)}</svg>\n')


def renamed(short):
    """Current id of a commit quoted in results/ (author dates were redistributed before the first push)."""
    mapping = load('results/history_rewrite.json')['old_to_new']
    matches = [new for old, new in mapping.items() if old.startswith(short)]
    return matches[0][:7] if len(matches) == 1 else short


def build():
    V, figures = {}, {}
    konvid = [r for r in load('data/manifests/konvid.csv') if r['status'] == 'ok']
    live = [r for r in load('data/manifests/livevqc.csv') if r['status'] == 'ok']
    lengths = load('results/phase3/lengths.json')
    features_cfg, protocol = load('configs/features_v2.yaml'), load('configs/protocol.yaml')
    params = load('results/phase1/head_parameters.json')
    head2 = {(r['feature_set'], r['head']): r for r in load('results/phase2/headline.csv')}
    comp2 = load('results/phase2/comparisons.csv')
    selection = load('results/phase2/selection.json')
    freeze = load('results/phase2/konvid_freeze.json')
    head3 = {(r['feature_set'], r['head']): r for r in load('results/phase3/headline.csv')}
    comp3 = load('results/phase3/comparisons.csv')
    gate3 = load('results/phase3/phase_gate.json')
    neural_null = load('results/phase3/posthoc_permutation/neural/summary.json')
    ridge_null = load('results/phase3/posthoc_permutation/summary.json')
    overlap = load('results/inner_val_overlap.json')
    part_a = load('results/posthoc_dino/part_a.json')
    part_b = load('results/posthoc_dino/part_b.csv')
    grid = load('results/posthoc_dino/grid_followup/grid_followup.json')
    deployment = load('results/phase3/deployment.json')
    refit = load('results/phase3/refit/resnet50/ridge/seedNone/run.json')
    latency = load('results/phase4/latency_summary.json')
    served = load('results/phase4b/latency.json')
    gate4b = load('results/phase4b/phase_gate.json')
    clean = load('results/phase4/clean_build/steps.json')
    acceptance = load('results/phase4/frontend_acceptance.json')
    time1, time2 = load('results/phase1/time_accounting.json'), load('results/phase2/time_accounting.json')
    sota = load('results/sota_citations.csv')
    for linked in ('results/phase4b/browser/container-real-result.png', 'results/phase1/parity.json', 'results/phase4b/report.md',
                   'results/posthoc_dino/report.md', 'results/posthoc_dino/grid_followup/report.md', 'docs/dataset_licenses.md',
                   'results/phase3/posthoc_permutation/report.md', 'results/phase3/posthoc_permutation/neural/report.md',
                   'results/phase1/control_final/audit.md', 'results/phase1/control_final/independent_audit.md',
                   'results/phase4/latency.md', 'results/phase4/clean_build.md', 'results/phase4/frontend_acceptance.md'):
        if not Path(linked).is_file():
            raise SystemExit(f'missing linked file: {linked}')

    # ---------- Method ----------
    def dataset_row(name, rows, key, role):
        durations = [float(r['duration_s']) for r in rows]
        size = Counter(f"{r['width']}×{r['height']}" for r in rows).most_common(1)[0]
        t = lengths[key]
        return [name, len(rows), rows[0]['mos_scale'], f'{statistics.median(durations):.1f} [{min(durations):.1f}–{max(durations):.1f}]',
                f'{size[0]} ({size[1]} of {len(rows)})', f"{t['median']:.0f} [{t['minimum']}–{t['maximum']}]", role]
    V['datasets_table'] = table(['Dataset', 'Videos', 'MOS scale', 'Duration s, median [min–max]', 'Most common resolution',
                                 'Sampled frames T, median [min–max]', 'Role'],
                                [dataset_row('KoNViD-1k', konvid, 'konvid', 'training, 5-fold CV'),
                                 dataset_row('LIVE-VQC', live, 'livevqc', 'cross-dataset test only')])
    pre = features_cfg['preprocessing']
    V.update(fps=str(pre['spatial_fps']), crop=str(pre['spatial_crop']), interpolation=pre['spatial_interpolation'],
             precision=pre['offline_precision'], n_konvid=str(len(konvid)), n_live=str(len(live)),
             n_seeds=str(len(protocol['seeds'])), max_epochs=str(protocol['max_epochs']), patience=str(protocol['patience']),
             trials=str(protocol['trials_per_head']), rank_weight=str(protocol['rank_weight']), margin=str(MARGIN),
             control_abs=str(CONTROL_ABS),
             alpha_grid=', '.join(f'{a:g}' for a in freeze['refit_by_feature']['resnet50']['ridge']['alpha_grid']))
    prow = [[feature, NAMES[head], f"{params[dim][head]['front_end']:,}", f"{params[dim][head]['head']:,}", f"{params[dim][head]['total']:,}"]
            for dim, feature in (('2048', 'ResNet-50'), ('1536', 'DINOv2')) for head in HEADS]
    V['params_table'] = table(['Features', 'Head', 'Shared front end', 'Head only', 'Total trainable'], prow)
    head_only = [params[d][h]['head'] for d in params for h in HEADS]
    V['head_param_spread'] = f'{(max(head_only) - min(head_only)) / min(head_only):.1%}'

    # ---------- Head comparison (selected feature set) ----------
    chosen = selection['selected_feature_set']
    V['chosen'] = FEATURES[chosen]
    def headline_row(head):
        r = head2[(chosen, head)]
        basis = f"{r['run_count']} folds (one deterministic fit each)" if head == 'ridge' else f"{r['run_count']} fold × seed runs"
        return [NAMES[head], pm(r['srcc_mean'], r['srcc_std']), pm(r['plcc_mapped_mean'], r['plcc_mapped_std']),
                pm(r['krcc_mean'], r['krcc_std']), basis]
    V['head_table'] = table(['Head', 'SRCC', 'PLCC (logistic-mapped)', 'KRCC', 'SD across'], [headline_row(h) for h in (*HEADS, 'ridge')])
    label = lambda s: NAMES[s.split(':')[1]]
    primary = [r for r in comp2 if r['family'] == 'primary_srcc']
    ridge_rows = [r for r in comp2 if r['pair_id'].startswith('ridge/') and r['metric'] == 'srcc']
    V['pair_table'] = table(['A − B', 'Δ SRCC', '95% CI', 'Holm p', f'Classification (±{MARGIN})'],
                            [[f"{label(r['a'])} − {label(r['b'])}", f"{float(r['delta']):+.4f}", ci(r['ci_low'], r['ci_high'], 4),
                              num(r['p_holm'], 3) if r['p_holm'] else 'secondary, unadjusted', r['classification']] for r in primary + ridge_rows])
    classes = Counter(r['classification'] for r in primary)
    meaningful = classes['A better than B (practically meaningful)'] + classes['B better than A (practically meaningful)']
    V.update(n_pairs=str(len(primary)), n_detectable=str(classes['detectable difference, not shown to exceed the margin']),
             n_inconclusive=str(classes['no detectable difference (inconclusive at this sample size)']),
             n_equivalent=str(classes['equivalent within margin']), n_meaningful=str(meaningful))
    V['head_verdict'] = ('no head beats another by more than the margin, so the four heads tie' if meaningful == 0
                         else f'{meaningful} of {len(primary)} head pairs differ by a practically meaningful amount')
    detect = [r for r in primary if r['classification'] == 'detectable difference, not shown to exceed the margin']
    def higher(r):
        a, b = (r['a'], r['b']) if float(r['delta']) > 0 else (r['b'], r['a'])
        return f"{label(a)} higher than {label(b)}, detectable but not shown to exceed the margin (Δ {abs(float(r['delta'])):.4f})"
    V['detectable_pairs'] = '; '.join(map(higher, detect)) or 'none'
    figures['head_comparison.svg'] = ci_plot([(f"{label(r['a'])} − {label(r['b'])}", float(r['delta']), float(r['ci_low']), float(r['ci_high']))
                                              for r in primary], f'Head pairs on {FEATURES[chosen]} (KoNViD-1k): Δ SRCC, 95% CI',
                                             f'Δ SRCC (grey band: ±{MARGIN} margin)', margin=MARGIN)
    ridge_detect = [r for r in ridge_rows if r['classification'] == 'detectable difference, not shown to exceed the margin']
    ridge_higher = all(float(r['delta']) > 0 for r in ridge_rows)
    V['ridge_tldr'] = 'numerically above every neural head, without a practically meaningful margin' if ridge_higher and not any(
        r['classification'].endswith('(practically meaningful)') for r in ridge_rows) else 'compared with the neural heads in the pairwise table'
    V['ridge_vs_heads'] = (f"Ridge was {'numerically higher than every neural head' if ridge_higher else 'not uniformly higher than the neural heads'}; "
                           f"the difference was detectable but not shown to exceed the margin for {len(ridge_detect)} of {len(ridge_rows)} heads, "
                           'and not detectable for the others')

    # ---------- Feature ablation ----------
    frow, fitems = [], []
    feature_rows = [r for r in comp2 if r['pair_id'].startswith('feature/') and r['metric'] == 'srcc']
    for r in feature_rows:
        head = r['pair_id'].split('/')[1]
        d, rn = head2[('dino', head)], head2[('resnet50', head)]
        frow.append([NAMES[head], pm(d['srcc_mean'], d['srcc_std']), pm(rn['srcc_mean'], rn['srcc_std']),
                     f"{float(r['delta']):+.4f}", ci(r['ci_low'], r['ci_high'], 4), r['classification']])
        for feature, row in (('DINOv2', d), ('ResNet-50', rn)):
            m, s = float(row['srcc_mean']), float(row['srcc_std'])
            fitems.append((f'{NAMES[head]} · {feature}', m, m - s, m + s))
    V['feature_table'] = table(['Head', 'DINOv2 SRCC', 'ResNet-50 SRCC', 'Δ (DINOv2 − ResNet-50)', '95% CI', 'Classification'], frow)
    resnet_wins = sum(r['classification'] == 'B better than A (practically meaningful)' for r in feature_rows)
    V['feature_verdict'] = f'ResNet-50 beats the DINOv2 last-layer representation for {resnet_wins} of {len(feature_rows)} heads'
    V['val_dino'], V['val_resnet'] = (num(selection['per_feature_validation_mean'][k], 4) for k in ('dino', 'resnet50'))
    figures['feature_ablation.svg'] = ci_plot(fitems, 'KoNViD-1k per-fold SRCC, mean ± SD, by head and feature set', 'SRCC')
    V.update(overlap_videos=str(overlap['unique_inner_validation_videos_also_held_out']), overlap_total=str(overlap['videos']),
             overlap_fraction=f"{overlap['fraction_of_videos']:.1%}", overlap_per_fold=str(overlap['folds'][0]['val']),
             overlap_rows=str(overlap['inner_validation_rows']))

    # ---------- LIVE-VQC ----------
    lrow = []
    for feature in ('resnet50', 'dino'):
        for head in (*HEADS, 'ridge'):
            r = head3[(feature, head)]
            single = head == 'ridge'
            fmt = (lambda m, s: num(m)) if single else pm
            lrow.append([FEATURES[feature], NAMES[head], r['analysis'], 'single deterministic fit' if single else f"{r['n_seeds']} seeds",
                         fmt(r['srcc_mean'], r['srcc_sd']), fmt(r['krcc_mean'], r['krcc_sd']), fmt(r['plcc_mapped_mean'], r['plcc_mapped_sd'])])
    V['live_table'] = table(['Features', 'Head', 'Analysis', 'Fits', 'SRCC', 'KRCC', 'PLCC (logistic-mapped)'], lrow)
    lab3 = lambda s: NAMES[s.split('/')[1]]
    live_primary = [r for r in comp3 if r['family'] == 'primary' and r['metric'] == 'srcc']
    V['live_pair_table'] = table(['A − B', 'Δ SRCC', '95% CI', 'Holm p', 'Classification'],
                                 [[f"{lab3(r['a'])} − {lab3(r['b'])}", f"{float(r['delta']):+.4f}", ci(r['ci95_low'], r['ci95_high'], 4),
                                   num(r['p_holm'], 3), r['classification']] for r in live_primary])
    none_detect = sum(r['classification'] == 'no detectable difference (inconclusive at this sample size)' for r in live_primary)
    V['live_verdict'] = f'{none_detect} of {len(live_primary)} head pairs show no detectable difference on LIVE-VQC'
    deployed_live = head3[('resnet50', 'ridge')]
    V.update(deployed_live_srcc=num(deployed_live['srcc_mean']), deployed_konvid_srcc=num(head2[('resnet50', 'ridge')]['srcc_mean']),
             deployed_live_plcc=num(deployed_live['plcc_mapped_mean']))

    # ---------- Post-hoc (exploratory, unadjusted) ----------
    a_rows = {r['row']: r for r in part_a['summary']}
    V['parta_table'] = table(['Variant', 'SRCC mean ± SD (5 folds)', 'Folds at the original grid edge'],
                             [[k, pm(a_rows[k]['srcc_mean'], a_rows[k]['srcc_sd']), a_rows[k]['grid_edge_folds']] for k in ('A0', 'L6', 'L9', 'L12', 'ALL')])
    g_rows = {r['row']: r for r in grid['summary']}
    V['grid_table'] = table(['Row', 'SRCC mean ± SD', 'KRCC mean ± SD', 'Alphas by fold', 'Folds at 1e6'],
                            [[k, pm(r['srcc_mean'], r['srcc_sd']), pm(r['krcc_mean'], r['krcc_sd']), r['alphas'], r['grid_edge_folds']]
                             for k, r in g_rows.items()])
    figures['dino_layers_posthoc.svg'] = ci_plot([(k, float(r['srcc_mean']), float(r['srcc_mean']) - float(r['srcc_sd']),
                                                   float(r['srcc_mean']) + float(r['srcc_sd'])) for k, r in g_rows.items()],
                                                  'Post-hoc, exploratory, unadjusted: DINOv2 layers vs ResNet-50 ridge',
                                                  'KoNViD-1k SRCC, mean ± SD over 5 folds (extended alpha grid)')
    gc = {(c['a'], c['b'], c['metric']): c for c in grid['comparisons']}
    l6, verdict = gc[('L6', 'ResNet-50', 'srcc')], grid['verdict']
    l6_words = {'A better than B (practically meaningful)': 'beat',
                'detectable difference, not shown to exceed the margin': 'were higher than, detectably but not beyond the margin,',
                'no detectable difference (inconclusive at this sample size)': 'showed no detectable difference from',
                'equivalent within margin': 'were equivalent within the margin to',
                'B better than A (practically meaningful)': 'lost to'}
    V['l6_phrase'] = l6_words[l6['classification']]
    V.update(l6_delta=f"{l6['delta']:+.4f}", l6_ci=ci(l6['ci95_low'], l6['ci95_high'], 4), l6_class=l6['classification'],
             robust='robust' if verdict['robust'] else 'not robust', worst_edge=str(verdict['max_folds_at_largest_grid_value']),
             resnet_old_edge=str(verdict['original_resnet50_at_original_grid_edge']), grid_values=', '.join(f'{a:g}' for a in grid['alpha_grid']),
             a0_repro=f"{max(part_a['a0_reproduction']['per_fold_absolute_difference']):g}",
             predeclare_a=renamed(part_a['predeclaration_commit']), predeclare_grid=renamed(grid['predeclaration_commit']),
             parta_edge=str(sum(r['alpha_at_grid_edge'] for r in part_a['folds'])), parta_folds=str(len(part_a['folds'])))
    ratio = {}
    for r in part_b:
        ratio.setdefault(r['branch'], []).append(float(r['ratio']))
    ratio = {k: sum(v) / len(v) for k, v in ratio.items()}
    V['ratio_resnet'], V['ratio_l6'], V['ratio_l12'] = (num(ratio[k]) for k in ('resnet50', 'dino_patchmean_L6', 'dino_patchmean_L12'))
    glive = {r['row']: r for r in grid['live']}
    V['live_l6'], V['live_resnet_grid'], V['live_all'] = (num(glive[k]['srcc']) for k in ('L6', 'ResNet-50', 'ALL'))

    # ---------- SOTA (Rule 21): rendered only when every row is verified ----------
    blocking = [f"{r['model']} / {r['dataset']} / {r['metric']} / {r['table/page']}" for r in sota
                if r['value'].strip().upper() == 'TODO' or r['verified_by_user'].strip().lower() != 'yes']
    if blocking:
        print(f'SOTA table blocked by {len(blocking)} unverified or TODO rows in results/sota_citations.csv:', file=sys.stderr)
        for b in blocking:
            print('  -', b, file=sys.stderr)
        V['sota_table'] = (f'> The comparison table is withheld: {len(blocking)} of {len(sota)} rows in '
                           '[results/sota_citations.csv](results/sota_citations.csv) are not yet verified against the original papers. '
                           'It is generated automatically once every row is marked verified.')
        V['sota_tldr'] = 'the gap to published end-to-end models is stated once the cited values are verified'
    else:
        V['sota_table'] = table(['Model', 'Dataset', 'Protocol', 'Metric', 'Value', 'Source'],
                                [[r['model'], r['dataset'], r['split/protocol'], r['metric'], r['value'], f"[{r['table/page']}]({r['url']})"] for r in sota])
        best = max(float(r['value']) for r in sota if r['dataset'] == 'KoNViD-1k' and r['metric'] == 'SRCC')
        V['sota_tldr'] = (f'published end-to-end models report KoNViD-1k SRCC up to {best:.3f} under their own protocols, '
                          f"against {V['deployed_konvid_srcc']} here")

    # ---------- Validity ----------
    V.update(gate3_status=gate3['status'], control_status=gate3['control_status'],
             control_srcc=num(neural_null['historical_ensemble_srcc']), null_perms=str(neural_null['permutations']),
             null_models=str(neural_null['models_per_permutation']), null_interval=ci(*neural_null['central_95_percent_null_interval']),
             null_interpretation=neural_null['interpretation'], ridge_perms=str(ridge_null['null']['permutations']),
             ridge_interval=ci(*ridge_null['null']['central_95_percent_interval']))

    # ---------- Demo ----------
    lat = {(s['feature_set'], s['head']): s for s in latency['summary']}
    lrows = []
    for feature in ('resnet50', 'dino'):
        for head in ('ridge', *HEADS):
            s = lat[(feature, head)]
            lrows.append([FEATURES[feature], NAMES[head] + (' (deployed)' if s['deployed'] else ''),
                          f"{num(s['extraction_seconds_median'], 2)} [{num(s['extraction_seconds_q1'], 2)}–{num(s['extraction_seconds_q3'], 2)}]",
                          num(1000 * s['head_seconds_median'], 3), num(s['seconds_per_10s_median'], 2)])
    V['latency_table'] = table(['Features', 'Head', 'Extraction s, median [IQR]', 'Head ms, median', 'Total s per 10 s of video'], lrows)
    check = latency['serving_consistency']
    V.update(lat_threads=str(latency['cpu_threads']), lat_clips=str(len(latency['clip_ids'])),
             host_extraction=num(check['script_median_extraction_seconds'], 2), container_extraction=num(check['serving_median_extraction_seconds'], 2),
             container_gap=f"{check['relative_difference']:.1%}", container_cli=num(check['container_cli_median_extraction_seconds'], 2),
             host_cpus=str(check['host_cpus']), quota=str(check['container_cpu_quota']),
             served_median=num(served['summary']['median_seconds_per_ten_seconds'], 2),
             served_max=num(served['summary']['maximum_seconds_per_ten_seconds'], 2),
             served_budget=str(served['summary']['budget_seconds_per_ten_seconds']))
    result = clean['result']
    V.update(deployed_id=deployment['candidate_id'], deployed_alpha=f"{refit['alpha']:g}", model_hash=deployment['checkpoint_sha256'][:12],
             weights_hash=result['reported']['weights_sha256'][:12], preprocessing_hash=result['reported']['preprocessing_sha256'][:12],
             gate4b_status=gate4b['status'], image_id=gate4b['image_id'][:19],
             acceptance_items=str(len(acceptance['items'])), acceptance_passed=str(sum(i['status'] == 'pass' for i in acceptance['items'])),
             build_minutes=num(result['build_seconds'] / 60, 1), image_gb=num(result['image_size_bytes'] / 1e9, 2),
             clean_score=num(result['status']['result']['prediction']['score'], 4),
             clean_total=num(result['status']['result']['prediction']['timings_seconds']['total'], 2))
    release = load('results/phase4/testcard_tag.json')
    published = load('results/phase4/hf_space.json')
    if published['local_image_id'] != gate4b['image_id']:
        raise SystemExit('published image is not the gated image')
    V.update(pinned_image=published['pinned_reference'], pinned_tag=published['image'])
    if release['image_id'] != gate4b['image_id']:
        raise SystemExit('testcard tag does not point at the gated image')
    V.update(release_tag=release['tag'], release_image=release['image_id'][:19], model_name=release['model_name'])
    steps = {s['step']: s['command'] for s in clean['steps']}
    V['clean_steps'] = '\n'.join([
        'git clone https://github.com/JustJoyful/Testcard.git',
        'cd Testcard', steps['lfs_install'], steps['lfs_model'], steps['lfs_phase3'], steps['venv'], steps['install'], steps['prepare_bundle'],
        steps['docker_build'].replace(' vqa:clean ', f" {release['tag']} "),
        steps['docker_run'].replace('vqa-clean-test', 'testcard-demo').replace(':8002:', ':8001:').replace(' vqa:clean', f" {release['tag']}")])

    # ---------- Hardware and time ----------
    V.update(gpu=load('results/posthoc_dino/part_b.json')['device'], gpu_hours_p1=num(time1['gpu_job_wall_hours'], 1),
             gpu_hours_p2=num(time2['phase2_gpu_job_wall_hours'], 1))
    return V, figures


def render():
    V, figures = build()
    text = load(TEMPLATE)
    missing = sorted(set(re.findall(r'\{\{(\w+)\}\}', text)) - set(V))
    if missing:
        raise SystemExit(f'unresolved placeholders: {missing}')
    for key, value in V.items():
        text = text.replace('{{' + key + '}}', str(value))
    if '{{' in text:
        raise SystemExit('placeholder left after rendering')
    header = '<!-- Generated by scripts/18_make_readme.py from README.template.md and results/. Do not edit by hand. -->\n'
    return header + text, figures


def main():
    text, figures = render()
    if sys.argv[1:] == ['--check']:
        problems = []
        for path, content in [(README, text)] + [(FIGURES / n, s) for n, s in figures.items()]:
            current = path.read_text() if path.is_file() else ''
            if current != content:
                problems.append(path)
                sys.stdout.writelines(difflib.unified_diff(current.splitlines(True), content.splitlines(True), str(path), 'regenerated', n=1))
        if problems:
            raise SystemExit(f'--check failed: {len(problems)} file(s) differ from regenerated output')
        print('--check passed: README.md and figures match regenerated output')
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, svg in figures.items():
        (FIGURES / name).write_text(svg)
    README.write_text(text)
    print(f'wrote {README} and {len(figures)} figures')


if __name__ == '__main__':
    main()
