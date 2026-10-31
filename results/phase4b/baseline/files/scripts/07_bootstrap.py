"""One shared paired video-bootstrap bank for KoNViD primary/secondary comparisons."""
import argparse
import importlib
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cache import atomic_json, atomic_write, csv_bytes
from src.features import sha256
from src.metrics import make_bootstrap_indices, paired_bootstrap, classify_difference, holm_adjust, bootstrap_indices_hash
from src.phase2 import HEADS, load_phase2_context, require_resnet_controls, collect_feature_outputs, aligned_ensembles, save_selection
from src.training import config_hash


def comparison_roster(chosen):
    rows=[{'pair_id':f'neural/{a}-{b}','family':'primary_srcc','a':[chosen,a],'b':[chosen,b]} for a,b in combinations(HEADS,2)]
    rows.extend({'pair_id':f'ridge/{h}','family':'secondary','a':[chosen,'ridge'],'b':[chosen,h]} for h in HEADS)
    rows.extend({'pair_id':f'feature/{h}','family':'secondary','a':['dino',h],'b':['resnet50',h]} for h in (*HEADS,'ridge'))
    return rows


def compare_ensembles(y,values,chosen,*,draws=10000,seed=0,margin=.01):
    indices=make_bootstrap_indices(len(y),seed,draws)
    bank_hash=bootstrap_indices_hash(indices)
    records=[]
    for n,pair in enumerate(comparison_roster(chosen),1):
        print(f"CPU bootstrap: pair {n}/15 {pair['pair_id']}, {draws} paired draws per metric",flush=True)
        metrics={metric:paired_bootstrap(y,values[tuple(pair['a'])],values[tuple(pair['b'])],metric,
                                         seed,draws,resample_indices=indices) for metric in ('srcc','krcc')}
        if any(r['status']!='ok' or r['resample_indices_sha256']!=bank_hash for r in metrics.values()):
            raise ValueError(f"failed metric or mismatched index bank: {pair['pair_id']}")
        metrics['srcc']['classification']=classify_difference(metrics['srcc']['ci95'],margin)
        records.append({**pair,'metrics':metrics})
    adjusted=holm_adjust({r['pair_id']:r['metrics']['srcc']['p_raw'] for r in records if r['family']=='primary_srcc'})
    for r in records:
        if r['pair_id'] in adjusted:
            r['metrics']['srcc'].update(p_holm=adjusted[r['pair_id']],holm_reject=adjusted[r['pair_id']]<=.05)
    return records,{'resample_indices_sha256':bank_hash,'draws':draws,'seed':seed,
                    'pvalue_method':'centered_bootstrap_add_one','ci_scope':'pointwise 95%; seed-averaged ensemble; test sampling noise only'}


def comparison_csv(records):
    rows=[]
    for r in records:
        for metric,m in r['metrics'].items():
            rows.append({'pair_id':r['pair_id'],'family':r['family'] if metric=='srcc' else 'secondary',
                         'metric':metric,'a':':'.join(r['a']),'b':':'.join(r['b']),
                         'delta':m['delta'],'ci_low':m['ci95'][0],'ci_high':m['ci95'][1],
                         'p_raw':m['p_raw'],'p_holm':m.get('p_holm'),'holm_reject':m.get('holm_reject'),
                         'classification':m.get('classification')})
    return csv_bytes(rows,list(rows[0]))


def bootstrap_phase2(config,features):
    context=load_phase2_context(config,features)
    require_resnet_controls(context)
    selection=save_selection(context)
    outputs,sources=collect_feature_outputs(context)
    ids,y,values=aligned_ensembles(context,outputs)
    records,meta=compare_ensembles(y,values,selection['selected_feature_set'])
    root=Path(context['policy']['results_root'])
    atomic_json(root/'comparisons.json',records)
    atomic_write(root/'comparisons.csv',comparison_csv(records))
    meta.update(source_identity=context['source_identity'],cv_sources_sha256=sources,
                selection_sha256=sha256(root/'selection.json'),video_ids_sha256=config_hash(ids),
                policy=context['policy']['statistics'],json_sha256=sha256(root/'comparisons.json'),csv_sha256=sha256(root/'comparisons.csv'))
    atomic_json(root/'comparisons.meta.json',meta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    args=p.parse_args()
    bootstrap_phase2(args.config,args.features)


if __name__=='__main__':
    main()
