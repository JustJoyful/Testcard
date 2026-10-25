"""Train cached spatial heads under the frozen, validation-only protocol."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cache import atomic_json, validate_cache
from src.data import CachedFeatureDataset
from src.features import sha256
from src.heads import build_head
from src.splits import freeze_splits
from src.training import fit_one, predict, seed_everything, load_completed, config_hash
from src.tuning import require_control, require_selection_gates, make_trials, select_trial, same_inputs

HEADS=['mean_pool','bigru','transformer','mamba']


def prerequisites(protocol,features):
    cache=validate_cache(Path(protocol['cache_root']),features)
    freeze_splits(protocol['inventory'],protocol['splits'])
    for path in (protocol['splits'],str(Path(protocol['splits']).with_suffix('.meta.json'))):
        subprocess.run(['git','ls-files','--error-unmatch',path],check=True,capture_output=True)
        subprocess.run(['git','diff','--exit-code','HEAD','--',path],check=True,capture_output=True)
    return {'config_feature_sha256':sha256(features),'split_sha256':sha256(protocol['splits']),
            'cache_sha256':cache['manifest_sha256'],'cache_meta_sha256':sha256(Path(protocol['cache_root'])/'meta.json'),
            'lock_sha256':sha256('env/requirements.lock'),'weights_sha256':cache['weights_sha256'],
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}


def resolved_config(protocol):
    return {**protocol,'tier':yaml.safe_load(Path(protocol['head_tiers']).read_text())['matched']}


def role_data(protocol,fold,role):
    return CachedFeatureDataset(Path(protocol['cache_root']),Path(protocol['inventory']),Path(protocol['splits']),
                                fold,role,protocol['feature_set'])


def fit_run(protocol,config,provenance,head,fold,seed,output,train=None,val=None,run_kind='single',label_mode='real'):
    seed_everything(seed,protocol['cpu_threads'])
    model=build_head(head,1536 if protocol['feature_set']=='dino' else 2048,config)
    run_provenance={**provenance,'head':head,'fold':fold,'run_kind':run_kind,'label_mode':label_mode}
    identity={'config_sha256':config_hash(config),'seed':seed,
              'provenance':{k:v for k,v in run_provenance.items() if k!='code_commit'}}
    model.to(protocol['device'])
    existing=load_completed(model,output,identity,protocol['device'])
    if existing:
        return model,existing
    if train is None:
        train=role_data(protocol,fold,'train')
    if val is None:
        val=role_data(protocol,fold,'val')
    result=fit_one(model,train,val,config,output,seed,
                   run_provenance)
    return model,result


def control_runs(protocol,provenance,final=False):
    import numpy as np
    from src.metrics import bootstrap_srcc
    from src.training import config_hash
    root=Path(protocol['results_root'])/('control_final' if final else 'control_initial')
    root.mkdir(parents=True,exist_ok=True)
    if protocol['seeds']!=list(range(5)):
        raise ValueError('controls require all five seeds')
    config=selected_configs(protocol,provenance)['mean_pool'] if final else resolved_config(protocol)
    snapshot=root/'config.json'
    if snapshot.exists():
        saved=json.loads(snapshot.read_text())
        if final and saved!=config:
            raise ValueError('changed selected final-control config')
        config=saved
    else:
        atomic_json(snapshot,config)
    train=role_data(protocol,0,'train')
    val=role_data(protocol,0,'val')
    order=np.random.default_rng(0).permutation(len(train))
    permutation=[{'video_id':train[i]['video_id'],'label_source_id':train[int(j)]['video_id']} for i,j in enumerate(order)]
    permutation_path=Path(protocol['results_root'])/'control_permutation.json'
    if permutation_path.exists() and json.loads(permutation_path.read_text())!=permutation:
        raise ValueError('fixed training permutation changed')
    if not permutation_path.exists():
        atomic_json(permutation_path,permutation)
    shuffled=[{**train[i],'mos':train[int(j)]['mos']} for i,j in enumerate(order)]
    outputs=[]
    for seed in protocol['seeds']:
        model,run=fit_run(protocol,config,{**provenance,'permutation_sha256':sha256(permutation_path)},
                          'mean_pool',0,seed,root/f'seed{seed}',shuffled,val,
                          'control_final' if final else 'control_initial','shuffled')
        test=role_data(protocol,0,'test')
        rows=predict(model,test,run['target_stats'],protocol['device'])
        labels={test[i]['video_id']:test[i]['mos'] for i in range(len(test))}
        rows=[{**r,'mos':labels[r['video_id']],'seed':seed,'run_kind':'control_final' if final else 'control_initial',
               'label_mode':'shuffled','checkpoint_sha256':run['checkpoint_sha256']} for r in rows]
        atomic_json(root/f'seed{seed}'/'predictions.json',rows)
        outputs.append(rows)
    ids=[r['video_id'] for r in outputs[0]]
    ensemble=np.array([[r['prediction'] for r in rows] for rows in outputs]).mean(0)
    result=bootstrap_srcc(np.array([r['mos'] for r in outputs[0]]),ensemble)
    passed=result['status']=='ok' and result['ci95'][0]<=0<=result['ci95'][1] and abs(result['srcc'])<.15
    diagnostic={'status':'passed' if passed else 'failed','bootstrap':result,'seeds':protocol['seeds'],
                'provenance':provenance,'config_sha256':config_hash(config),
                'permutation_sha256':sha256(permutation_path),'video_ids':ids,
                'seed_averaged_predictions':ensemble.tolist()}
    atomic_json(root/'diagnostic.json',diagnostic)
    if not passed:
        raise ValueError(f'shuffled-label control failed: {root}/diagnostic.json; investigate without rerolling')
    return diagnostic


def ridge_pilot(train,val):
    import time
    import numpy as np
    from src.baselines import fit_ridge
    def pooled(data):
        return np.array([data[i]['features'].numpy().mean(0,dtype=np.float64) for i in range(len(data))])
    started=time.perf_counter()
    fit_ridge(pooled(train),np.array([train[i]['mos'] for i in range(len(train))]),
              pooled(val),np.array([val[i]['mos'] for i in range(len(val))]),[.001,.01,.1,1,10,100,1000])
    elapsed=time.perf_counter()-started
    return {'status':'complete','device':'cpu','alpha_fits':7,'wall_seconds':elapsed,
            'projected_five_fold_seconds':5*elapsed,'assumption':'five times measured fold-0 CPU ridge work; no test-role loading'}


def pilot_runs(protocol,provenance):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    require_control(Path(protocol['results_root'])/'control_initial',provenance)
    root=Path(protocol['results_root'])
    current=resolved_config(protocol)
    snapshot=root/'pilot'/'config.json'
    if snapshot.exists():
        config=json.loads(snapshot.read_text())
        mutable={'status','tuning_seeds'}
        if {k:v for k,v in config.items() if k not in mutable}!={k:v for k,v in current.items() if k not in mutable}:
            raise ValueError('incompatible immutable pilot fitting snapshot')
    else:
        config=current
        atomic_json(snapshot,config)
    train=role_data(protocol,0,'train')
    val=role_data(protocol,0,'val')
    runs={}
    for head in HEADS:
        _,runs[head]=fit_run(protocol,config,provenance,head,0,0,root/'pilot'/head,train,val,run_kind='pilot')
    existing_path=root/'training_pilot.json'
    if existing_path.exists():
        existing=json.loads(existing_path.read_text())
        if existing['status']=='complete' and existing['runs']==runs and same_inputs(existing['provenance'],provenance):
            return existing
    result={'status':'complete','runs':runs,'ridge_cpu':ridge_pilot(train,val),'provenance':provenance,
            'completed_at':datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()}
    atomic_json(root/'training_pilot.json',result)
    return result


def freeze_protocol(protocol,config_path,provenance,pilot):
    from src.tuning import calendar_gate, runtime_budget
    from src.training import config_hash
    from src.cache import atomic_write
    try:
        calendar_gate(pilot['completed_at'])
    except ValueError as exc:
        record_scope_stop(protocol,str(exc),{'provenance':provenance,'gate':'calendar'})
        raise
    scope=yaml.safe_load(Path('configs/scope.yaml').read_text())
    extraction=json.loads((Path(protocol['results_root'])/'extraction.json').read_text())
    actual=sum(r['wall_seconds'] for r in extraction['invocations'])/3600
    actual_extraction=actual
    actual+=sum(r['wall_seconds'] for r in pilot['runs'].values())/3600
    for seed in range(5):
        run=json.loads((Path(protocol['results_root'])/'control_initial'/f'seed{seed}'/'run.json').read_text())
        actual+=run['wall_seconds']/3600
    projection=json.loads(Path('results/phase0/scoped_feasibility.json').read_text())
    extraction_planning=sum(projection['projections'][h]['combined_planning_seconds'] for h in ('dino','resnet50'))/3600
    budget=runtime_budget(scope,protocol,pilot['runs'],extraction_planning,actual,actual_extraction_hours=actual_extraction)
    if budget['status']!='go' and protocol['tuning_seeds']==list(range(5)):
        protocol={**protocol,'tuning_seeds':[0,1]}
        with Path('DECISIONS.md').open('a') as f:
            f.write('\n## '+local_date()+': Measured tuning-only fallback\n\n'
                    +f"Five-seed tuning projected {budget['planning_2x_hours']:.6f} GPU hours against {budget['remaining_head_hours']:.6f} remaining; activate common rank/head tuning seeds [0,1] before selection. CV and both controls retain seeds 0–4. See results/phase1/runtime_budget.json.\n")
        budget=runtime_budget(scope,protocol,pilot['runs'],extraction_planning,actual,actual_extraction_hours=actual_extraction)
    if budget['status']=='go':
        protocol={**protocol,'status':'frozen'}
    atomic_write(config_path,yaml.safe_dump(protocol,sort_keys=False).encode())
    budget.update(ridge_cpu=pilot['ridge_cpu'],provenance=provenance,protocol_sha256=config_hash(protocol),pilot_sha256=sha256(Path(protocol['results_root'])/'training_pilot.json'))
    atomic_json(Path(protocol['results_root'])/'runtime_budget.json',budget)
    if budget['status']!='go':
        record_scope_stop(protocol,'runtime no-go after the allowed tuning-seed fallback',budget)
        raise ValueError('runtime no-go after allowed fallback: stop and record further scope proposal')
    return protocol


def local_date():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat()


def record_scope_stop(protocol,reason,budget):
    proposal='Defer the Phase 4 demo; revise the Phase 1 target to November 1 and reserve November 2–4 for ResNet and LIVE-VQC. This is a proposal requiring a scope decision, not an activated change.'
    record={**budget,'status':'no-go','date':local_date(),'reason':reason,'scope_proposal':proposal}
    atomic_json(Path(protocol['results_root'])/'runtime_budget.json',record)
    with Path('DECISIONS.md').open('a') as f:
        f.write('\n## '+local_date()+': Phase 1 stopped before tuning\n\n'+reason+'\n\n'+proposal+'\n')


def without_commit(value):
    if isinstance(value,dict):
        return {k:without_commit(v) for k,v in value.items() if k!='code_commit'}
    if isinstance(value,list):
        return [without_commit(v) for v in value]
    return value


def persist_yaml(path,value):
    from src.cache import atomic_write
    path=Path(path)
    if path.exists():
        existing=yaml.safe_load(path.read_text())
        if without_commit(existing)!=without_commit(value):
            raise ValueError(f'incompatible immutable config: {path}')
        return existing
    atomic_write(path,yaml.safe_dump(value,sort_keys=False).encode())
    return value


def rank_runs(protocol,provenance):
    from src.training import config_hash
    require_selection_gates(protocol,provenance)
    root=Path(protocol['results_root'])/'rank'
    train,val=role_data(protocol,0,'train'),role_data(protocol,0,'val')
    rows=[]
    for candidate,weight in enumerate(protocol['rank_candidates']):
        config={**resolved_config(protocol),'rank_weight':weight}
        for seed in protocol['tuning_seeds']:
            _,run=fit_run(protocol,config,provenance,'mean_pool',0,seed,root/f'candidate{candidate}'/f'seed{seed}',train,val,run_kind='rank')
            rows.append({'candidate_id':candidate,'seed':seed,'validation_srcc':run['best_validation_srcc'],
                         'status':run['status'],'provenance':provenance})
    winner=select_trial(rows,protocol['tuning_seeds'])
    result={'status':'frozen','rank_weight':protocol['rank_candidates'][winner['candidate_id']],
            'protocol_sha256':config_hash(protocol),'provenance':provenance,'winner':winner,'rows':rows}
    result=persist_yaml('configs/rank_loss.yaml',result)
    atomic_json(root/'selection.json',result)
    return result


def tune_runs(protocol,provenance):
    from src.training import config_hash
    rank=require_selection_gates(protocol,provenance,need_rank=True)
    candidates=make_trials(protocol)
    # Persist the same ordered candidates once, before any head is fitted.
    persist_yaml('configs/tuning_trials.yaml',{'status':'frozen','protocol_sha256':config_hash(protocol),'trials':candidates})
    train,val=role_data(protocol,0,'train'),role_data(protocol,0,'val')
    for head in HEADS:
        rows=[]
        root=Path(protocol['results_root'])/'tune'/head
        for trial in candidates:
            candidate=trial['candidate_id']
            config={**resolved_config(protocol),**{k:v for k,v in trial.items() if k!='candidate_id'},'rank_weight':rank['rank_weight']}
            for seed in protocol['tuning_seeds']:
                _,run=fit_run(protocol,config,provenance,head,0,seed,root/f'candidate{candidate}'/f'seed{seed}',train,val,run_kind='tune')
                rows.append({'candidate_id':candidate,'seed':seed,'validation_srcc':run['best_validation_srcc'],
                             'status':run['status'],'provenance':provenance})
        winner=select_trial(rows,protocol['tuning_seeds'])
        config={**resolved_config(protocol),**{k:v for k,v in candidates[winner['candidate_id']].items() if k!='candidate_id'},
                'rank_weight':rank['rank_weight']}
        result={'status':'frozen','head':head,'protocol_sha256':config_hash(protocol),'provenance':provenance,
                'config':config,'winner':winner,'rows':rows}
        result=persist_yaml(f'configs/heads/{head}.yaml',result)
        atomic_json(root/'selection.json',result)


def selected_configs(protocol,provenance):
    from src.training import config_hash
    rank=require_selection_gates(protocol,provenance,need_rank=True)
    result={}
    for head in HEADS:
        path=Path(f'configs/heads/{head}.yaml')
        if not path.exists():
            raise ValueError(f'missing selected head config: {head}')
        record=yaml.safe_load(path.read_text())
        if (record['status']!='frozen' or record['head']!=head or record['protocol_sha256']!=config_hash(protocol)
                or not same_inputs(record['provenance'],provenance) or record['config']['rank_weight']!=rank['rank_weight']):
            raise ValueError(f'changed/incompatible head selection: {head}')
        result[head]=record['config']
    return result


def prediction_rows(protocol,model,run,test,head,fold,seed,run_id):
    labels={test[i]['video_id']:test[i]['mos'] for i in range(len(test))}
    return [{**r,'dataset':'konvid','feature_set':protocol['feature_set'],'head':head,'fold':fold,'seed':seed,
             'mos':labels[r['video_id']],'checkpoint_sha256':run['checkpoint_sha256'],'run_id':run_id,
             'run_kind':'cv','label_mode':'real'} for r in predict(model,test,run['target_stats'],protocol['device'])]


def save_predictions(path,rows,run):
    path=Path(path)
    meta_path=path.with_suffix('.meta.json')
    identity={'run_identity':run['identity'],'checkpoint_sha256':run['checkpoint_sha256']}
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['identity']!=identity or meta['predictions_sha256']!=sha256(path):
            raise ValueError(f'incompatible/corrupt completed predictions: {path}')
        if json.loads(path.read_text())!=rows:
            raise ValueError(f'changed prediction rows: {path}')
        return
    atomic_json(path,rows)
    atomic_json(meta_path,{'status':'complete','identity':identity,'predictions_sha256':sha256(path)})


def ridge_run(protocol,provenance,fold,train,val,test):
    import io
    import time
    import numpy as np
    from src.cache import atomic_write
    from src.baselines import fit_ridge,predict_ridge
    root=Path(protocol['results_root'])/'ridge'/f'fold{fold}'
    root.mkdir(parents=True,exist_ok=True)
    identity={'config_sha256':config_hash(protocol),'provenance':{k:v for k,v in provenance.items() if k!='code_commit'},'fold':fold}
    metadata=root/'run.json'
    start=time.perf_counter()
    def pool(data):
        return np.array([data[i]['features'].numpy().mean(0,dtype=np.float64) for i in range(len(data))])
    if metadata.exists():
        record=json.loads(metadata.read_text())
        if record['status']=='complete':
            if record['identity']!=identity or sha256(root/'model.npz')!=record['checkpoint_sha256']:
                raise ValueError(f'incompatible ridge artifact: {root}')
            with np.load(root/'model.npz',allow_pickle=False) as archive:
                artifact={k:archive[k] for k in archive.files}
        else:
            raise ValueError('unexpected partial ridge metadata; investigate')
    else:
        artifact=fit_ridge(pool(train),np.array([train[i]['mos'] for i in range(len(train))]),
                           pool(val),np.array([val[i]['mos'] for i in range(len(val))]),[.001,.01,.1,1,10,100,1000])
        buffer=io.BytesIO()
        np.savez(buffer,**{k:v for k,v in artifact.items() if k!='trials'})
        atomic_write(root/'model.npz',buffer.getvalue())
        record={'status':'complete','identity':identity,'checkpoint_sha256':sha256(root/'model.npz'),
                'alpha':artifact['alpha'],'validation_srcc':artifact['validation_srcc'],'trials':artifact['trials'],
                'wall_seconds':time.perf_counter()-start,'code_commit':provenance['code_commit']}
        atomic_json(metadata,record)
    values=predict_ridge(artifact,pool(test))
    rows=[{'video_id':test[i]['video_id'],'dataset':'konvid','feature_set':protocol['feature_set'],'head':'ridge',
           'fold':fold,'seed':None,'mos':test[i]['mos'],'prediction':float(value),'run_kind':'cv','label_mode':'real',
           'run_id':f'ridge/fold{fold}','checkpoint_sha256':record['checkpoint_sha256']} for i,value in enumerate(values)]
    save_predictions(root/'predictions.json',rows,record)
    return rows


def cv_runs(protocol,provenance):
    configs=selected_configs(protocol,provenance)
    final=require_control(Path(protocol['results_root'])/'control_final',provenance)
    if final['config_sha256']!=config_hash(configs['mean_pool']):
        raise ValueError('final control does not match selected mean-pool config')
    if protocol['seeds']!=list(range(5)) or protocol['feature_set']!='dino':
        raise ValueError('Phase 1 requires DINO and five seeds')
    for fold in range(5):
        train,val=role_data(protocol,fold,'train'),role_data(protocol,fold,'val')
        for head in HEADS:
            for seed in protocol['seeds']:
                run_id=f'cv/{head}/fold{fold}/seed{seed}'
                output=Path(protocol['results_root'])/run_id
                model,run=fit_run(protocol,configs[head],provenance,head,fold,seed,output,train,val,run_kind='cv')
                # Test-role loading is deliberately after checkpoint selection/restoration.
                test=role_data(protocol,fold,'test')
                rows=prediction_rows(protocol,model,run,test,head,fold,seed,run_id)
                save_predictions(output/'predictions.json',rows,run)
        test=role_data(protocol,fold,'test')
        ridge_run(protocol,provenance,fold,train,val,test)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['single','control-initial','pilot','rank','tune','control','cv'],required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--head',choices=HEADS)
    p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--seed',type=int,choices=range(5))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    protocol=yaml.safe_load(args.config.read_text())
    provenance=prerequisites(protocol,args.features)
    if args.mode=='control-initial':
        control_runs(protocol,provenance)
        return
    if args.mode=='pilot':
        pilot=pilot_runs(protocol,provenance)
        freeze_protocol(protocol,args.config,provenance,pilot)
        return
    if args.mode=='rank':
        rank_runs(protocol,provenance)
        return
    if args.mode=='tune':
        tune_runs(protocol,provenance)
        return
    if args.mode=='control':
        control_runs(protocol,provenance,final=True)
        return
    if args.mode=='cv':
        cv_runs(protocol,provenance)
        return
    if any(v is None for v in (args.head,args.fold,args.seed,args.output)):
        p.error('single requires --head, --fold, --seed and --output')
    _,result=fit_run(protocol,resolved_config(protocol),provenance,args.head,args.fold,args.seed,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
