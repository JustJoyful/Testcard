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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['single'],required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--head',choices=HEADS)
    p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--seed',type=int,choices=range(5))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    protocol=yaml.safe_load(args.config.read_text())
    provenance=prerequisites(protocol,args.features)
    if any(v is None for v in (args.head,args.fold,args.seed,args.output)):
        p.error('single requires --head, --fold, --seed and --output')
    _,result=fit_run(protocol,resolved_config(protocol),provenance,args.head,args.fold,args.seed,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
