"""Gated ResNet controls/CV and validation-only feature choice; no DINO refits."""
import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.phase2 import load_phase2_context, require_resnet_controls
from src.progress import ProgressReporter
from src.training import config_hash
from src.tuning import control_directory, require_control


def run_mode(mode,config,features):
    c=load_phase2_context(config,features)
    if mode=='check':
        print('Phase 1 sources verified; Phase 2 effective configs bound.',flush=True)
        return
    if mode in ('cv','select'):
        require_resnet_controls(c)
    p=c['resnet_protocol'];provenance=c['base_provenance']
    training=importlib.import_module('scripts.05_train')
    show=ProgressReporter(Path(c['policy']['results_root'])/'progress.json')
    show({'event':'stage_start','stage':mode,'device':p['device'],'feature_set':'resnet50','control_complete':0})
    if mode=='control-initial':
        training.control_runs(p,provenance,config=training.resolved_config(p),progress=show)
    elif mode=='control':
        initial=require_control(control_directory(p['results_root']),provenance)
        if initial['config_sha256']!=config_hash(training.resolved_config(p)):
            raise ValueError('initial ResNet control config mismatch')
        training.control_runs(p,provenance,final=True,config=c['resnet_configs']['mean_pool'],progress=show)
    elif mode=='cv':
        training.fit_cv_grid(p,provenance,c['resnet_configs'],progress=show)
    elif mode=='select':
        from src.phase2 import save_selection
        save_selection(c)
    else:
        raise ValueError('unknown Phase 2 mode')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['check','control-initial','control','cv','select'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    args=p.parse_args()
    run_mode(args.mode,args.config,args.features)


if __name__=='__main__':
    main()
