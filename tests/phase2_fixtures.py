"""Small real artifact tree; only hardware/backbone and legacy control audits are doubled."""
import importlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.cache import atomic_json, atomic_write, csv_bytes
from src.features import sha256
from src.training import config_hash

REPO=Path(__file__).resolve().parents[1]
HEADS=['mean_pool','bigru','transformer','mamba']


def phase2_tree(tmp_path,monkeypatch):
    # Keep declared relative interfaces identical to a real checkout.
    for name in ['protocol.yaml','head_tiers.yaml','control_v2.yaml']:
        path=tmp_path/'configs'/name; path.parent.mkdir(exist_ok=True)
        path.write_bytes((REPO/'configs'/name).read_bytes())
    policy={'schema_version':1,'source_protocol':'configs/protocol.yaml','phase1_root':'results/phase1',
            'results_root':'results/phase2','features':'configs/features_v2.yaml',
            'selection':{'aggregation':'equal_head_fold_seed_mean','tie_break':'dino'},
            'statistics':{'draws':10000,'seed':0,'margin':.01,'primary_metric':'srcc',
                          'pvalue_method':'centered_bootstrap_add_one','min_finite_fraction':.99}}
    (tmp_path/'configs/phase2.yaml').write_text(yaml.safe_dump(policy))
    monkeypatch.chdir(tmp_path)
    protocol=yaml.safe_load(Path('configs/protocol.yaml').read_text())
    inventory=[{'video_id':f'v{i:02}','mos':float(i%4+1)} for i in range(20)]
    splits=[{'video_id':r['video_id'],'outer_fold':fold,'role':'test' if i//4==fold else ('val' if i//4==(fold+1)%5 else 'train')}
            for fold in range(5) for i,r in enumerate(inventory)]
    atomic_write(protocol['inventory'],csv_bytes(inventory,['video_id','mos']))
    atomic_write(protocol['splits'],csv_bytes(splits,['video_id','outer_fold','role']))
    Path('configs/features_v2.yaml').write_text('cache_version: v2\n')
    provenance={'split_sha256':sha256(protocol['splits']),'code_commit':'fixture'}
    configs={h:{**protocol,'tier':yaml.safe_load(Path('configs/head_tiers.yaml').read_text())['matched']} for h in HEADS}
    for h,c in configs.items():
        Path('configs/heads').mkdir(exist_ok=True)
        Path(f'configs/heads/{h}.yaml').write_text(yaml.safe_dump({'config':c}))
    Path('configs/rank_loss.yaml').write_text('rank_weight: 0.3\n')
    training=importlib.import_module('scripts.05_train')
    report=importlib.import_module('scripts.06_eval')
    parity=importlib.import_module('scripts.04_parity_check')
    monkeypatch.setattr(training,'prerequisites',lambda *a:provenance)
    monkeypatch.setattr(training,'selected_configs',lambda *a:configs)
    monkeypatch.setattr(report,'validate_parameter_budget',lambda *a:{'fixture':'matched'})
    monkeypatch.setattr(parity,'validate_parity',lambda *a:{'status':'passed'})
    # require_control itself is tested with actual artifacts elsewhere; retain config binding here.
    def control(directory,p):
        c=configs['mean_pool'] if 'final' in str(directory) else training.resolved_config(protocol)
        return {'status':'passed','config_sha256':config_hash(c)}
    monkeypatch.setattr('src.tuning.require_control',control)
    make_cv(protocol,configs,provenance)
    atomic_json('results/phase1/phase_gate.json',{'status':'passed','evidence':{
        'cache_and_committed_splits':provenance,'frozen_selection':configs}})
    return policy,protocol,configs,provenance


def make_cv(protocol,configs,provenance):
    training=importlib.import_module('scripts.05_train')
    root=Path(protocol['results_root'])
    for head in HEADS+['ridge']:
        for fold in range(5):
            for seed in ([None] if head=='ridge' else range(5)):
                run_id=f'ridge/fold{fold}' if head=='ridge' else f'cv/{head}/fold{fold}/seed{seed}'
                directory=root/run_id; directory.mkdir(parents=True,exist_ok=True)
                config=protocol if head=='ridge' else configs[head]
                identity={'config_sha256':config_hash(config),'provenance':{k:v for k,v in provenance.items() if k!='code_commit'}}
                if head=='ridge':
                    identity['fold']=fold
                else:
                    identity.update(seed=seed)
                    identity['provenance'].update(head=head,fold=fold,run_kind='cv',label_mode='real')
                artifact=directory/('model.npz' if head=='ridge' else 'best.pt')
                if head=='ridge':
                    np.savez(artifact,alpha=1.,coefficient=np.ones(2))
                else:
                    artifact.write_bytes(b'fixture checkpoint')
                run={'status':'complete','identity':identity,'provenance':{**provenance,**identity['provenance']},
                     'config':config,'seed':seed,'best_epoch':fold*5+(seed or 0)+1,'best_validation_srcc':.5,
                     'checkpoint_sha256':sha256(artifact),'wall_seconds':1.,'alpha':1.,'validation_srcc':.5,
                     'trials':[{'alpha':a,'validation_srcc':.5 if a==1 else .4,'status':'ok'} for a in [.001,.01,.1,1,10,100,1000]]}
                atomic_json(directory/'run.json',run)
                atomic_json(directory/'config.json',config)
                rows=[{'video_id':f'v{i:02}','mos':float(i%4+1),'prediction':float(i%4+1)+.001*(seed or 0),
                       'dataset':'konvid','feature_set':protocol['feature_set'],'head':head,'fold':fold,'seed':seed,
                       'checkpoint_sha256':run['checkpoint_sha256'],'run_id':run_id,'run_kind':'cv','label_mode':'real'}
                      for i in range(fold*4,fold*4+4)]
                training.save_predictions(directory/'predictions.json',rows,run)
