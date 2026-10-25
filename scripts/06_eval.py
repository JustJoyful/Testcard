"""Generate completeness-first Phase 1 diagnostics from saved artifacts."""
import argparse
from collections import Counter, defaultdict
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cache import atomic_json, atomic_write, csv_bytes
from src.features import sha256
from src.metrics import correlations, fit_logistic_plcc
from src.training import config_hash
from src.tuning import require_control, same_inputs


def validate_oof(rows,fold_by_id,heads,seeds):
    expected={(key,head,seed) for key in fold_by_id for head in heads for seed in seeds}
    counts=Counter()
    wrong=[]
    invalid=[]
    for r in rows:
        key=(r.get('video_id'),r.get('head'),r.get('seed'))
        counts[key]+=1
        if r.get('video_id') not in fold_by_id or r.get('fold')!=fold_by_id.get(r.get('video_id')):
            wrong.append(r.get('video_id'))
        if (key not in expected or r.get('run_kind')!='cv' or r.get('label_mode')!='real'
                or not isinstance(r.get('prediction'),(float,int)) or not np.isfinite(r['prediction'])
                or not isinstance(r.get('mos'),(float,int)) or not np.isfinite(r['mos'])):
            invalid.append(list(key))
    missing=sorted(expected-counts.keys(),key=str)
    duplicates=[key for key,n in counts.items() if n>1]
    return {'status':'complete' if not(missing or duplicates or wrong or invalid) else 'failed',
            'missing_keys':[list(k) for k in missing],'duplicate_keys':[list(k) for k in duplicates],
            'wrong_fold_ids':wrong,'invalid_rows':invalid}


def summarize(rows,runs,root):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['head'],row['fold'],row['seed'])].append(row)
    metrics=[]
    failed=[]
    for (head,fold,seed),members in sorted(groups.items(),key=lambda p:str(p[0])):
        y=np.array([r['mos'] for r in members]); p=np.array([r['prediction'] for r in members])
        score=correlations(y,p)
        mapping=fit_logistic_plcc(y,p)
        key=f'{head}/fold{fold}/seed{seed}'
        if score['status']!='ok' or mapping['status']!='ok':
            failed.append(key)
        metrics.append({'head':head,'fold':fold,'seed':seed,**score,'plcc_mapped':mapping['plcc_mapped'],
                        'mapping_status':mapping['status'],'mapping_reason':mapping['reason']})
        atomic_json(root/'mapped_plcc'/f'{head}-fold{fold}-seed{seed}.json',mapping)
    atomic_write(root/'metrics.csv',csv_bytes(metrics,list(metrics[0]) if metrics else ['head','fold','seed']))
    summary={}
    for head in sorted({r['head'] for r in rows}):
        members=[r for r in metrics if r['head']==head]
        if any(r['status']!='ok' or r['mapping_status']!='ok' for r in members):
            summary[head]={'status':'failed','reason':'mandatory undefined metrics retained in metrics.csv'}
            continue
        summary[head]={'status':'complete','run_count':len(members),'headline_quantity':'mean per-fold SRCC across fold × seed runs',
                       'srcc_mean':float(np.mean([r['srcc'] for r in members])),
                       'srcc_std':float(np.std([r['srcc'] for r in members],ddof=1)),
                       'krcc_mean':float(np.mean([r['krcc'] for r in members])),
                       'plcc_raw_mean':float(np.mean([r['plcc_raw'] for r in members])),
                       'plcc_mapped_mean':float(np.mean([r['plcc_mapped'] for r in members])),
                       'plcc_mapping_domain':'target_labels'}
        for field in ('krcc','plcc_raw','plcc_mapped'):
            summary[head][field+'_std']=float(np.std([r[field] for r in members],ddof=1))
        for field in ('seed','fold'):
            means=[np.mean([r['srcc'] for r in members if r[field]==value]) for value in {r[field] for r in members}]
            summary[head][field+'_srcc_std']=float(np.std(means,ddof=1)) if len(means)>1 else None
        if head!='ridge':
            summary[head]['best_epochs']=[r['best_epoch'] for r in runs if r['provenance']['head']==head]
    ensembles=[]
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['head'],r['video_id'])].append(r)
    for (head,key),members in sorted(grouped.items()):
        ensembles.append({'head':head,'video_id':key,'fold':members[0]['fold'],'mos':members[0]['mos'],
                          'prediction':float(np.mean([r['prediction'] for r in members])),
                          'quantity':'seed-averaged pooled OOF ensemble'})
    atomic_json(root/'oof_ensemble.json',ensembles)
    atomic_json(root/'summary.json',{'heads':summary,'undefined_metric_runs':failed,
                                    'note':'Definitive fixed-DINO comparison; Phase 2 reuses these runs. Target-label logistic mapping; no bootstrap PLCC.'})
    return summary,failed


def collect_cv(protocol,provenance,configs,root):
    training=importlib.import_module('scripts.05_train')
    rows,runs=[],[]
    for head in training.HEADS+['ridge']:
        for fold in range(5):
            for seed in ([None] if head=='ridge' else protocol['seeds']):
                directory=root/'ridge'/f'fold{fold}' if head=='ridge' else root/'cv'/head/f'fold{fold}'/f'seed{seed}'
                run=json.loads((directory/'run.json').read_text())
                artifact=directory/('model.npz' if head=='ridge' else 'best.pt')
                if run['status']!='complete' or sha256(artifact)!=run['checkpoint_sha256']:
                    raise ValueError(f'incomplete/corrupt run: {directory}')
                expected_config=protocol if head=='ridge' else configs[head]
                if run['identity']['config_sha256']!=config_hash(expected_config) or not same_inputs(run['identity']['provenance'],provenance):
                    raise ValueError(f'wrong config/split/cache: {directory}')
                if head!='ridge':
                    identity=run['identity']['provenance']
                    if identity['head']!=head or identity['fold']!=fold or identity['run_kind']!='cv' or identity['label_mode']!='real' or run['seed']!=seed:
                        raise ValueError(f'wrong run identity: {directory}')
                    if json.loads((directory/'config.json').read_text())!=expected_config:
                        raise ValueError(f'changed resolved run config: {directory}')
                    runs.append(run)
                path=directory/'predictions.json'
                metadata=json.loads(path.with_suffix('.meta.json').read_text())
                expected_identity={'run_identity':run['identity'],'checkpoint_sha256':run['checkpoint_sha256']}
                if metadata['status']!='complete' or metadata['identity']!=expected_identity or metadata['predictions_sha256']!=sha256(path):
                    raise ValueError(f'corrupt/stale predictions: {path}')
                predictions=json.loads(path.read_text())
                if any(r['head']!=head or r['fold']!=fold or r['seed']!=seed or r['checkpoint_sha256']!=run['checkpoint_sha256'] or r['feature_set']!='dino' or r['dataset']!='konvid' for r in predictions):
                    raise ValueError(f'prediction provenance mismatch: {path}')
                rows.extend(predictions)
    return rows,runs


def validate_parameter_budget(protocol):
    from src.heads import build_head
    from src.heads.common import parameter_counts
    tier=yaml.safe_load(Path(protocol['head_tiers']).read_text())['matched']
    expected=json.loads(Path('results/phase1/head_parameters.json').read_text())
    actual={str(d):{h:parameter_counts(build_head(h,d,{'tier':tier,'input_dropout':protocol['input_dropout'],'head_dropout':protocol['head_dropout']}))
                     for h in ('mean_pool','bigru','transformer','mamba')} for d in (1536,2048)}
    if actual!=expected or any(max(r['head'] for r in group.values())/min(r['head'] for r in group.values())>1.05 for group in actual.values()):
        raise ValueError('parameter budget changed/failed')
    return actual


def validate_runtime(protocol,provenance,root):
    from src.tuning import planned_fit_counts,calendar_gate
    pilot=json.loads((root/'training_pilot.json').read_text())
    budget=json.loads((root/'runtime_budget.json').read_text())
    if (pilot['status']!='complete' or set(pilot['runs'])!={'mean_pool','bigru','transformer','mamba'}
            or not same_inputs(pilot['provenance'],provenance) or budget['status']!='go'
            or budget['protocol_sha256']!=config_hash(protocol) or not same_inputs(budget['provenance'],provenance)
            or budget['pilot_sha256']!=sha256(root/'training_pilot.json')
            or budget['reserved_gpu_hours']!=30 or budget['planning_2x_hours']>budget['remaining_head_hours']):
        raise ValueError('missing/failed/stale actual pilot/runtime budget')
    calendar_gate(pilot['completed_at'])
    for head,run in pilot['runs'].items():
        if (run['status']!='complete' or run['config']['device']!='cuda' or run['peak_cuda_bytes']<=0
                or run['seed']!=0 or run['completed_epochs']<1 or run['wall_seconds']<=0
                or not same_inputs(run['provenance'],provenance)
                or json.loads((root/'pilot'/head/'run.json').read_text())!=run
                or sha256(root/'pilot'/head/'best.pt')!=run['checkpoint_sha256']):
            raise ValueError(f'invalid actual pilot: {head}')
    counts=planned_fit_counts(yaml.safe_load(Path('configs/scope.yaml').read_text()),protocol)
    if budget['planned_counts']!=counts:
        raise ValueError('runtime budget fit counts changed')
    return {'status':'go','planning_2x_hours':budget['planning_2x_hours'],'remaining_head_hours':budget['remaining_head_hours']}


def phase_report(protocol,root,features):
    root=Path(root)
    training=importlib.import_module('scripts.05_train')
    failures=[]
    evidence={}
    def gate(name,fn):
        try:
            evidence[name]=fn()
            return evidence[name]
        except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
            failures.append({'gate':name,'reason':str(exc)})
            return None
    provenance=gate('cache_and_committed_splits',lambda:training.prerequisites(protocol,features))
    configs=gate('frozen_selection',lambda:training.selected_configs(protocol,provenance or {}))
    gate('initial_control',lambda:require_control(root/'control_initial',provenance or {}))
    gate('final_control',lambda:require_control(root/'control_final',provenance or {}))
    parity=importlib.import_module('scripts.04_parity_check')
    gate('feature_parity',lambda:parity.validate_parity(root/'parity.json',provenance or {},Path(protocol['cache_root'])))
    gate('runtime_pilot',lambda:validate_runtime(protocol,provenance or {},root))
    gate('parameter_budget',lambda:validate_parameter_budget(protocol))
    if provenance and configs:
        collected=gate('cv_artifacts',lambda:collect_cv(protocol,provenance,configs,root))
        if collected:
            rows,runs=collected
            inventory=list(csv.DictReader(Path(protocol['inventory']).open()))
            source={r['video_id']:float(r['mos']) for r in inventory}
            splits=list(csv.DictReader(Path(protocol['splits']).open()))
            fold_by_id={r['video_id']:int(r['outer_fold']) for r in splits if r['role']=='test'}
            for name,heads,seeds in [('neural_oof',training.HEADS,protocol['seeds']),('ridge_oof',['ridge'],[None])]:
                check=validate_oof([r for r in rows if r['head'] in heads],fold_by_id,heads,seeds)
                evidence[name]=check
                if check['status']!='complete':
                    failures.append({'gate':name,'reason':'missing, duplicate, wrong-fold or invalid prediction rows'})
            if any(r['video_id'] not in source or r['mos']!=source[r['video_id']] for r in rows):
                failures.append({'gate':'labels','reason':'prediction labels mismatch authoritative inventory'})
            if not failures:
                summary,undefined=summarize(rows,runs,root)
                evidence['summary']=summary
                if undefined:
                    failures.append({'gate':'metrics','reason':f'undefined mandatory metrics: {undefined}'})
    gate_result={'status':'failed' if failures else 'passed','missing_or_failed_gates':failures,'evidence':evidence}
    atomic_json(root/'phase_gate.json',gate_result)
    atomic_write(root/'report.md',('# Phase 1 core slice\n\nStatus: '+gate_result['status']+'\n\n'
        +'Definitive fixed-DINO head comparison; Phase 2 must reuse completed CV and ridge artifacts.\n\n'
        +'Headline: mean per-fold SRCC across 25 fold × seed runs. Ensemble OOF is a separate quantity.\n'
        +'PLCC logistic mapping uses target labels. Warning-only determinism; no bitwise reproducibility claim.\n\n'
        +'\n'.join(f"- {r['gate']}: {r['reason']}" for r in failures)+'\n').encode())
    return gate_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    args=p.parse_args()
    result=phase_report(yaml.safe_load(args.config.read_text()),args.results,args.features)
    print(json.dumps({'status':result['status'],'missing_or_failed_gates':result['missing_or_failed_gates']},indent=2))
    return result['status']!='passed'


if __name__=='__main__':
    raise SystemExit(main())
