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
from src.training import fit_one, predict, seed_everything
from src.tuning import require_control

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
    if train is None:
        train=role_data(protocol,fold,'train')
    if val is None:
        val=role_data(protocol,fold,'val')
    result=fit_one(model,train,val,config,output,seed,
                   {**provenance,'head':head,'fold':fold,'run_kind':run_kind,'label_mode':label_mode})
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


def pilot_runs(protocol,provenance):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    require_control(Path(protocol['results_root'])/'control_initial',provenance)
    root=Path(protocol['results_root'])
    config=resolved_config(protocol)
    train=role_data(protocol,0,'train')
    val=role_data(protocol,0,'val')
    runs={}
    for head in HEADS:
        _,runs[head]=fit_run(protocol,config,provenance,head,0,0,root/'pilot'/head,train,val,run_kind='pilot')
    result={'status':'complete','runs':runs,'provenance':provenance,
            'completed_at':datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()}
    atomic_json(root/'training_pilot.json',result)
    return result


def freeze_protocol(protocol,config_path,provenance,pilot):
    from src.tuning import calendar_gate, runtime_budget
    from src.training import config_hash
    from src.cache import atomic_write
    calendar_gate(pilot['completed_at'])
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
            f.write('\n## 2026-10-25: Measured tuning-only fallback\n\n'
                    +f"Five-seed tuning projected {budget['planning_2x_hours']:.6f} GPU hours against {budget['remaining_head_hours']:.6f} remaining; activate common rank/head tuning seeds [0,1] before selection. CV and both controls retain seeds 0–4. See results/phase1/runtime_budget.json.\n")
        budget=runtime_budget(scope,protocol,pilot['runs'],extraction_planning,actual,actual_extraction_hours=actual_extraction)
    if budget['status']=='go':
        protocol={**protocol,'status':'frozen'}
    atomic_write(config_path,yaml.safe_dump(protocol,sort_keys=False).encode())
    budget.update(provenance=provenance,protocol_sha256=config_hash(protocol),pilot_sha256=sha256(Path(protocol['results_root'])/'training_pilot.json'))
    atomic_json(Path(protocol['results_root'])/'runtime_budget.json',budget)
    if budget['status']!='go':
        raise ValueError('runtime no-go after allowed fallback: stop and record further scope proposal')
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['single','control-initial','pilot'],required=True)
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
    if any(v is None for v in (args.head,args.fold,args.seed,args.output)):
        p.error('single requires --head, --fold, --seed and --output')
    _,result=fit_run(protocol,resolved_config(protocol),provenance,args.head,args.fold,args.seed,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
