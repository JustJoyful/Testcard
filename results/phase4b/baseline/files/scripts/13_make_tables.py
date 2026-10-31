"""Publish complete source-bound Phase 2 tables and freeze; never train or select on test metrics."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import src.phase2 as phase2
from src.cache import atomic_json, atomic_write, csv_bytes
from src.features import sha256
from src.metrics import correlations, holm_adjust, classify_difference, make_bootstrap_indices, bootstrap_indices_hash
from src.training import config_hash


def validate_comparisons(context,selection,outputs,source_hash):
    bootstrap=importlib.import_module('scripts.07_bootstrap')
    root=Path(context['policy']['results_root'])
    records=json.loads((root/'comparisons.json').read_text())
    meta=json.loads((root/'comparisons.meta.json').read_text())
    ids,y,values=phase2.aligned_ensembles(context,outputs)
    policy=context['policy']['statistics']
    expected_meta={'source_identity':context['source_identity'],'cv_sources_sha256':source_hash,
                   'selection_sha256':sha256(root/'selection.json'),'video_ids_sha256':config_hash(ids),
                   'policy':policy,'draws':10000,'seed':0,'pvalue_method':'centered_bootstrap_add_one',
                   'json_sha256':sha256(root/'comparisons.json'),'csv_sha256':sha256(root/'comparisons.csv'),
                   'resample_indices_sha256':bootstrap_indices_hash(make_bootstrap_indices(len(ids)))}
    if any(meta.get(k)!=v for k,v in expected_meta.items()):
        raise ValueError('stale/mismatched comparison sources/policy/index bank/checksums')
    roster={r['pair_id']:r for r in bootstrap.comparison_roster(selection['selected_feature_set'])}
    keys=[r['pair_id'] for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=set(roster):
        raise ValueError('missing/duplicate comparison pair roster')
    for r in records:
        expected=roster[r['pair_id']]
        if any(r.get(k)!=v for k,v in expected.items()) or set(r['metrics'])!={'srcc','krcc'}:
            raise ValueError('wrong comparison pair or metric roster')
        a=correlations(y,values[tuple(r['a'])]);b=correlations(y,values[tuple(r['b'])])
        for metric,m in r['metrics'].items():
            if (m.get('status')!='ok' or m.get('metric')!=metric or m.get('seed')!=0 or m.get('draws')!=10000
                    or m.get('pvalue_method')!='centered_bootstrap_add_one'
                    or m.get('resample_indices_sha256')!=meta['resample_indices_sha256']
                    or type(m.get('valid_draws')) is not int or type(m.get('invalid_draws')) is not int
                    or m['valid_draws']<9900 or m['invalid_draws']<0 or m['valid_draws']+m['invalid_draws']!=10000
                    or not np.isfinite([m['delta'],m['p_raw'],*m['ci95']]).all() or len(m['ci95'])!=2
                    or not -2<=m['ci95'][0]<=m['ci95'][1]<=2 or not 0<=m['p_raw']<=1
                    or not np.isclose(m['delta'],a[metric]-b[metric],rtol=0,atol=1e-12)):
                raise ValueError('undefined/invalid comparison metric or draw accounting')
            primary=r['family']=='primary_srcc' and metric=='srcc'
            if primary!=('p_holm' in m) or primary!=('holm_reject' in m):
                raise ValueError('Holm belongs only to six primary SRCC pairs')
            if metric=='srcc' and m.get('classification')!=classify_difference(m['ci95'],policy['margin']):
                raise ValueError('wrong ordered SRCC CI classification')
            if metric=='krcc' and 'classification' in m:
                raise ValueError('no KRCC equivalence margin was declared')
    adjusted=holm_adjust({r['pair_id']:r['metrics']['srcc']['p_raw'] for r in records if r['family']=='primary_srcc'})
    for r in records:
        if r['pair_id'] in adjusted:
            m=r['metrics']['srcc']
            if m['p_holm']!=adjusted[r['pair_id']] or m['holm_reject']!=(m['p_holm']<=.05):
                raise ValueError('incorrect primary Holm adjustment/rejection')
    if (root/'comparisons.csv').read_bytes()!=bootstrap.comparison_csv(records):
        raise ValueError('comparison CSV/JSON disagree')
    return records


def table_bytes(rows):
    return csv_bytes(rows,list(rows[0]))


def comparison_table(records):
    rows=[]
    for r in records:
        row={'pair_id':r['pair_id'],'family':r['family'],'a':':'.join(r['a']),'b':':'.join(r['b'])}
        for metric,m in r['metrics'].items():
            row.update({f'{metric}_delta':m['delta'],f'{metric}_ci_low':m['ci95'][0],f'{metric}_ci_high':m['ci95'][1],
                        f'{metric}_p_raw':m['p_raw']})
        row.update(srcc_p_holm=r['metrics']['srcc'].get('p_holm'),srcc_holm_reject=r['metrics']['srcc'].get('holm_reject'),
                   srcc_classification=r['metrics']['srcc']['classification'])
        rows.append(row)
    return table_bytes(rows)


def phase2_time_accounting(context):
    prior=json.loads((Path(context['policy']['phase1_root'])/'time_accounting.json').read_text())
    gpu=cpu=0.
    for path in Path(context['resnet_protocol']['results_root']).rglob('run.json'):
        run=json.loads(path.read_text())
        hours=run.get('wall_seconds',0)/3600
        if path.parent.parent.name=='ridge':
            cpu+=hours
        elif run.get('config',{}).get('device','').startswith('cuda'):
            gpu+=hours
    return {**prior,'phase1_gpu_job_wall_hours':prior['gpu_job_wall_hours'],'phase2_gpu_job_wall_hours':gpu,
            'gpu_job_wall_hours':prior['gpu_job_wall_hours']+gpu,'remaining_gpu_allowance_hours':100-prior['gpu_job_wall_hours']-gpu,
            'cpu_ridge_wall_hours':prior['cpu_ridge_wall_hours']+cpu,
            'measurement':'Cumulative saved run/attempt wall time, not kernel utilization. Untimed interruptions and hands-on time unmeasured; remaining allowance is an upper bound. CPU statistics/report overhead is not measured.'}


def update_decisions(selection):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    path=Path('DECISIONS.md')
    marker='## '+datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat()+': Phase 2 exit verified'
    text=path.read_text()
    if marker not in text:
        outcome=(f"\n{marker}\n\nGenerated validation-only selection: {selection['selected_feature_set']}. "
                 f"Mean inner-validation SRCC by representation: {json.dumps(selection['per_feature_validation_mean'],sort_keys=True)}. "
                 'Source-bound tables, paired CIs/Holm, full-refit inputs and the KoNViD freeze are saved in results/phase2/. '
                 'Deployment CPU eligibility remains unresolved and must be resolved before LIVE-VQC predictions. '
                 'Original failed control and post-hoc v2 amendment are disclosed in summary.json. No LIVE-VQC predictions were made.\n')
        atomic_write(path,(text+outcome).encode())


def phase2_report(config_path: Path,features: Path):
    root=Path(phase2.POLICY['results_root'])
    gate_path=root/'phase_gate.json'
    atomic_json(gate_path,{'status':'failed','reason':'Phase 2 report regeneration incomplete'})
    try:
        context=phase2.load_phase2_context(config_path,features)
        controls=phase2.require_resnet_controls(context)
        selection=phase2.save_selection(context)
        outputs,source_hash=phase2.collect_feature_outputs(context)
        comparisons=validate_comparisons(context,selection,outputs,source_hash)
        report=importlib.import_module('scripts.06_eval')
        headline=[]
        for feature,output in outputs.items():
            destination=root/'tables'/feature
            heads,undefined=report.summarize(output['rows'],output['neural_runs'],destination)
            if undefined or len(heads)!=5 or any(r['status']!='complete' for r in heads.values()):
                raise ValueError('undefined mandatory per-fold metrics')
            output['headlines']=heads
            for head,scores in heads.items():
                if scores['run_count']!=(5 if head=='ridge' else 25):
                    raise ValueError('incomplete headline runs')
                headline.append({'feature_set':feature,'head':head,**{k:v for k,v in scores.items() if k not in ('best_epochs','status')}})
        atomic_write(root/'headline.csv',table_bytes(headline))
        features_table=[{'feature_set':feature,'inner_validation_srcc_mean':selection['per_feature_validation_mean'][feature],
                         'selected':feature==selection['selected_feature_set'],**{h+'_inner_validation_srcc':v for h,v in selection['per_head_validation_means'][feature].items()}}
                        for feature in ('dino','resnet50')]
        atomic_write(root/'feature_comparison.csv',table_bytes(features_table))
        atomic_write(root/'head_comparison.csv',comparison_table([r for r in comparisons if not r['pair_id'].startswith('feature/')]))
        atomic_write(root/'feature_pairs.csv',comparison_table([r for r in comparisons if r['pair_id'].startswith('feature/')]))
        freeze=phase2.build_konvid_freeze(context,selection,outputs)
        history=report.control_history(Path(context['policy']['phase1_root']))
        summary={'selection':selection,'headlines':{f:o['headlines'] for f,o in outputs.items()},'controls':controls,
                 'control_history':history,'parameter_counts':context['parameter_counts'],'audit':context['audit'],
                 'source_identity':context['source_identity'],'cv_sources_sha256':source_hash,
                 'notes':['Headline: mean per-fold metrics; fold/seed sample SDs are separate from bootstrap CIs.',
                          'Paired CIs: seed-averaged pooled OOF ensemble; test sampling noise only, not training/fold uncertainty.',
                          'Pointwise 95% CIs are unadjusted; Holm-adjusted SRCC p-values are reported separately for six neural pairs.',
                          '95% equivalence CIs are conservative relative to 90% TOST. KRCC and ridge/feature pairs are secondary/unadjusted.',
                          'Feature/tuning inner-validation videos are test videos in other folds; selection overlap limits unbiased claims.',
                          'Raw and 4-parameter target-label-mapped PLCC are reported; PLCC is not bootstrapped.',
                          'Ridge Baseline 0 and negative/inconclusive findings are retained with the same prominence.',
                          'Warning-only determinism; no bitwise reproducibility guarantee. YouTube-UGC, cloud, motion and layer search are cut.',
                          'Deployment candidate CPU eligibility/winner unresolved; no LIVE-VQC predictions allowed yet.']}
        atomic_json(root/'summary.json',summary)
        atomic_json(root/'time_accounting.json',phase2_time_accounting(context))
        atomic_json(root/'konvid_freeze.json',freeze)
        update_decisions(selection)
        names=['effective_configs.json','selection.json','comparisons.json','comparisons.csv','comparisons.meta.json',
               'headline.csv','feature_comparison.csv','head_comparison.csv','feature_pairs.csv','summary.json','time_accounting.json','konvid_freeze.json']
        names.extend(str(p.relative_to(root)) for p in sorted((root/'tables').rglob('*')) if p.is_file())
        gate={'status':'passed','source_identity':context['source_identity'],'cv_sources_sha256':source_hash,
              'outputs_sha256':{name:sha256(root/name) for name in names}}
        atomic_json(gate_path,gate)
        return gate
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        gate={'status':'failed','reason':str(exc)}
        atomic_json(gate_path,gate)
        return gate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    args=p.parse_args()
    gate=phase2_report(args.config,args.features)
    print(json.dumps(gate,indent=2))
    return gate['status']!='passed'


if __name__=='__main__':
    raise SystemExit(main())
