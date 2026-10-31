"""Phase 2 provenance, immutable source reuse and validation-only selection."""
import csv
import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from src.cache import atomic_json
from src.features import sha256
from src.training import config_hash
from src.tuning import control_directory, require_control, same_inputs

HEADS=('mean_pool','bigru','transformer','mamba')
POLICY={'schema_version':1,'source_protocol':'configs/protocol.yaml','phase1_root':'results/phase1',
        'results_root':'results/phase2','features':'configs/features_v2.yaml',
        'selection':{'aggregation':'equal_head_fold_seed_mean','tie_break':'dino'},
        'statistics':{'draws':10000,'seed':0,'margin':.01,'primary_metric':'srcc',
                      'pvalue_method':'centered_bootstrap_add_one','min_finite_fraction':.99}}


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError(f'incompatible immutable snapshot: {path}')
    else:
        atomic_json(path,value)


def source_audit():
    root=Path(__file__).resolve().parents[1]
    return {'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'dirty_working_tree':bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
            'source_sha256':{str(p.relative_to(root)):sha256(p) for folder in ('src','scripts')
                             for p in sorted((root/folder).rglob('*')) if p.suffix in ('.py','.sh')}}


def authoritative_ids(protocol):
    inventory=list(csv.DictReader(Path(protocol['inventory']).open()))
    labels={r['video_id']:float(r['mos']) for r in inventory}
    tests=[r for r in csv.DictReader(Path(protocol['splits']).open()) if r['role']=='test']
    folds={r['video_id']:int(r['outer_fold']) for r in tests}
    if len(labels)!=len(inventory) or len(folds)!=len(tests) or set(labels)!=set(folds):
        raise ValueError('incomplete/duplicate authoritative inventory or test folds')
    return labels,folds


def validate_rows(protocol,rows):
    report=importlib.import_module('scripts.06_eval')
    labels,folds=authoritative_ids(protocol)
    for heads,seeds in [(HEADS,list(range(5))),(['ridge'],[None])]:
        check=report.validate_oof([r for r in rows if r['head'] in heads],folds,heads,seeds)
        if check['status']!='complete':
            raise ValueError(f'incomplete/invalid OOF coverage: {check}')
    if any(r['head'] not in (*HEADS,'ridge') or r['mos']!=labels.get(r['video_id']) for r in rows):
        raise ValueError('OOF MOS differs from authoritative inventory')


def load_phase2_context(config_path: Path,features: Path):
    training=importlib.import_module('scripts.05_train')
    report=importlib.import_module('scripts.06_eval')
    policy=yaml.safe_load(Path(config_path).read_text())
    if policy!=POLICY or Path(features)!=Path(policy['features']):
        raise ValueError('Phase 2 policy/feature override is forbidden')
    base=yaml.safe_load(Path(policy['source_protocol']).read_text())
    if base['feature_set']!='dino' or base['seeds']!=list(range(5)) or base['results_root']!=policy['phase1_root'] or base['status']!='frozen':
        raise ValueError('Phase 1 must be frozen DINO with all five seeds')
    gate_path=Path(policy['phase1_root'])/'phase_gate.json'
    gate=json.loads(gate_path.read_text())
    if gate['status']!='passed':
        raise ValueError('Phase 1 gate is not passed')
    provenance=training.prerequisites(base,features)
    configs=training.selected_configs(base,provenance)
    if not same_inputs(gate['evidence']['cache_and_committed_splits'],provenance) or gate['evidence']['frozen_selection']!=configs:
        raise ValueError('stale Phase 1 gate/configs')
    from src.tuning import require_control as check_control
    for final in (False,True):
        result=check_control(control_directory(base['results_root'],final),provenance)
        if result['config_sha256']!=config_hash(configs['mean_pool'] if final else training.resolved_config(base)):
            raise ValueError('Phase 1 control config mismatch')
    parity=importlib.import_module('scripts.04_parity_check')
    parity.validate_parity(Path(base['results_root'])/'parity.json',provenance,Path(base['cache_root']))
    counts=report.validate_parameter_budget(base)
    rows,runs=report.collect_cv(base,provenance,configs,Path(base['results_root']))
    validate_rows(base,rows)
    resnet={**base,'feature_set':'resnet50','results_root':str(Path(policy['results_root'])/'resnet50')}
    resnet_configs={h:{**c,'feature_set':'resnet50','results_root':resnet['results_root']} for h,c in configs.items()}
    sources=[Path(policy['source_protocol']),Path(features),Path(base['inventory']),Path(base['splits']),
             Path(base['head_tiers']),Path('configs/rank_loss.yaml'),Path('configs/control_v2.yaml'),gate_path]
    sources.extend(Path(f'configs/heads/{h}.yaml') for h in HEADS)
    sources.extend(p for p in Path(base['results_root']).rglob('*') if p.is_file() and (p.name in ('run.json','best.pt','model.npz','config.json','predictions.json','predictions.meta.json')))
    inputs={'policy':policy,'source_sha256':{str(p):sha256(p) for p in sorted(sources)},
            'base_provenance':{k:v for k,v in provenance.items() if k!='code_commit'},
            'resnet_protocol':resnet,'resnet_configs':resnet_configs}
    root=Path(policy['results_root'])
    snapshot=root/'effective_configs.json'
    audit=source_audit()
    if snapshot.exists():
        if json.loads(snapshot.read_text())['inputs']!=inputs:
            raise ValueError('incompatible effective configs/source snapshot')
    else:
        atomic_json(snapshot,{'inputs':inputs,'audit':audit})
    return {'policy':policy,'base_protocol':base,'base_provenance':provenance,'dino_configs':configs,
            'resnet_protocol':resnet,'resnet_configs':resnet_configs,'audit':audit,'parameter_counts':counts,
            'source_identity':config_hash(inputs)}


def require_resnet_controls(context):
    training=importlib.import_module('scripts.05_train')
    p=context['resnet_protocol']
    results={}
    for final in (False,True):
        directory=control_directory(p['results_root'],final)
        result=require_control(directory,context['base_provenance'])
        expected=context['resnet_configs']['mean_pool'] if final else training.resolved_config(p)
        if result['config_sha256']!=config_hash(expected) or json.loads((directory/'config.json').read_text())!=expected:
            raise ValueError('ResNet control feature/root/config mismatch')
        for seed in range(5):
            run=json.loads((directory/f'seed{seed}/run.json').read_text())
            if run['config']!=expected or json.loads((directory/f'seed{seed}/config.json').read_text())!=expected:
                raise ValueError('ResNet control seed config mismatch')
        results['final' if final else 'initial']=result
    return results


def load_validation_runs(context):
    report=importlib.import_module('scripts.06_eval')
    result={}
    for feature,protocol,configs in [('dino',context['base_protocol'],context['dino_configs']),
                                     ('resnet50',context['resnet_protocol'],context['resnet_configs'])]:
        records=[]
        for head in HEADS:
            for fold in range(5):
                for seed in range(5):
                    root=Path(protocol['results_root'])
                    run=report.load_cv_run(protocol,context['base_provenance'],configs,root,head,fold,seed)
                    epoch=run['best_epoch']
                    if type(epoch) is not int or not 1<=epoch<=configs[head]['max_epochs']:
                        raise ValueError('invalid CV best epoch')
                    if 'history' in run:
                        history=run['history']
                        defined=[r for r in history if r['validation_srcc'] is not None]
                        best=max(defined,key=lambda r:r['validation_srcc'])
                        if best['epoch']!=epoch or best['validation_srcc']!=run['best_validation_srcc']:
                            raise ValueError('best epoch/validation score differs from history')
                    records.append({'feature_set':feature,'head':head,'fold':fold,'seed':seed,
                                    'best_validation_srcc':run['best_validation_srcc'],
                                    'config_sha256':config_hash(configs[head]),
                                    'run_sha256':sha256(root/f'cv/{head}/fold{fold}/seed{seed}/run.json')})
        result[feature]=records
    return result


def select_feature(validation_runs):
    if set(validation_runs)!={'dino','resnet50'}:
        raise ValueError('both reported spatial feature sets required')
    expected={(h,f,s) for h in HEADS for f in range(5) for s in range(5)}
    fields={'feature_set','head','fold','seed','best_validation_srcc','config_sha256','run_sha256'}
    means,per_head,sources={},{},{}
    for feature,records in sorted(validation_runs.items()):
        seen=set()
        for r in records:
            if set(r)!=fields or r['feature_set']!=feature or type(r['fold']) is not int or type(r['seed']) is not int:
                raise ValueError('only validation projections with exact identities are accepted')
            key=(r['head'],r['fold'],r['seed'])
            score=r['best_validation_srcc']
            if (key not in expected or key in seen or not isinstance(score,(int,float)) or not np.isfinite(score)
                    or not -1<=score<=1 or any(not isinstance(r[k],str) or len(r[k])!=64 for k in ('config_sha256','run_sha256'))):
                raise ValueError('invalid/duplicate validation record')
            seen.add(key)
        if seen!=expected:
            raise ValueError('missing validation runs')
        ordered=sorted(records,key=lambda r:(r['head'],r['fold'],r['seed']))
        means[feature]=float(np.mean([r['best_validation_srcc'] for r in ordered]))
        per_head[feature]={h:float(np.mean([r['best_validation_srcc'] for r in ordered if r['head']==h])) for h in HEADS}
        sources[feature]=ordered
    return {'selected_feature_set':'resnet50' if means['resnet50']>means['dino'] else 'dino',
            'per_feature_validation_mean':means,'per_head_validation_means':per_head,'tie_break':'dino','sources':sources}


def save_selection(context):
    selection={**select_feature(load_validation_runs(context)),'source_identity':context['source_identity']}
    immutable_json(Path(context['policy']['results_root'])/'selection.json',selection)
    return selection


def collect_feature_outputs(context):
    report=importlib.import_module('scripts.06_eval')
    outputs={}
    source_hashes={}
    for feature,protocol,configs in [('dino',context['base_protocol'],context['dino_configs']),
                                     ('resnet50',context['resnet_protocol'],context['resnet_configs'])]:
        root=Path(protocol['results_root'])
        rows,runs=report.collect_cv(protocol,context['base_provenance'],configs,root)
        validate_rows(protocol,rows)
        ridge=[report.load_cv_run(protocol,context['base_provenance'],configs,root,'ridge',f,None) for f in range(5)]
        sources={str(p):sha256(p) for folder in ('cv','ridge') for p in sorted((root/folder).rglob('*'))
                 if p.is_file() and p.name in ('run.json','config.json','best.pt','model.npz','predictions.json','predictions.meta.json')}
        outputs[feature]={'rows':rows,'neural_runs':runs,'ridge_runs':ridge,'source_sha256':sources}
        source_hashes.update(sources)
    return outputs,config_hash(source_hashes)


def aligned_ensembles(context,outputs):
    labels,_=authoritative_ids(context['base_protocol'])
    ids=sorted(labels)
    values={}
    for feature,output in outputs.items():
        validate_rows(context['base_protocol'],output['rows'])
        members={(r['head'],r['video_id'],r['seed']):r['prediction'] for r in output['rows']}
        for head in (*HEADS,'ridge'):
            seeds=[None] if head=='ridge' else range(5)
            values[(feature,head)]=np.array([np.mean([members[(head,key,s)] for s in seeds]) for key in ids])
    return ids,np.array([labels[key] for key in ids]),values


RIDGE_ALPHAS=[.001,.01,.1,1,10,100,1000]


def build_konvid_freeze(context,selection,validated_runs):
    if selection!={**select_feature(load_validation_runs(context)),'source_identity':context['source_identity']}:
        raise ValueError('selection differs from validated inner scores')
    if set(validated_runs)!={'dino','resnet50'}:
        raise ValueError('both spatial result sets are required')
    policies={}
    candidates=[]
    for feature,output in sorted(validated_runs.items()):
        protocol=context['base_protocol'] if feature=='dino' else context['resnet_protocol']
        configs=context['dino_configs'] if feature=='dino' else context['resnet_configs']
        root=Path(protocol['results_root'])
        by_head={h:[] for h in HEADS}
        seen=set()
        for run in output['neural_runs']:
            head=run['provenance']['head'];fold=run['provenance']['fold'];seed=run['seed']
            key=(head,fold,seed)
            if head not in HEADS or fold not in range(5) or seed not in range(5) or key in seen:
                raise ValueError('duplicate/invalid neural freeze source')
            seen.add(key)
            path=root/f'cv/{head}/fold{fold}/seed{seed}/run.json'
            if json.loads(path.read_text())!=run or output['source_sha256'].get(str(path))!=sha256(path):
                raise ValueError('edited neural freeze source')
            epoch=run['best_epoch']
            if type(epoch) is not int or not 1<=epoch<=configs[head]['max_epochs']:
                raise ValueError('invalid freeze epoch')
            by_head[head].append({'fold':fold,'seed':seed,'best_epoch':epoch,'run_path':str(path),
                                  'run_sha256':sha256(path),'checkpoint_sha256':run['checkpoint_sha256']})
        if seen!={(h,f,s) for h in HEADS for f in range(5) for s in range(5)}:
            raise ValueError('incomplete neural freeze source grid')
        policies[feature]={h:{'config':configs[h],'median_best_epoch':int(np.median([r['best_epoch'] for r in records])),
                              'cv_epochs':sorted(records,key=lambda r:(r['fold'],r['seed']))} for h,records in by_head.items()}
        ridge_sources=[]
        for run in output['ridge_runs']:
            fold=run['identity']['fold']
            path=root/f'ridge/fold{fold}/run.json'
            if fold not in range(5) or json.loads(path.read_text())!=run or output['source_sha256'].get(str(path))!=sha256(path):
                raise ValueError('edited ridge freeze source')
            trials=run['trials']
            if [r['alpha'] for r in trials]!=RIDGE_ALPHAS:
                raise ValueError('ridge alpha grid changed')
            valid=[r for r in trials if r['status']=='ok' and r['validation_srcc'] is not None and np.isfinite(r['validation_srcc'])]
            if not valid:
                raise ValueError('undefined ridge validation selection')
            selected=max(valid,key=lambda r:(r['validation_srcc'],-r['alpha']))
            with np.load(path.parent/'model.npz',allow_pickle=False) as archive:
                model_alpha=float(archive['alpha'])
            if selected['alpha']!=run['alpha'] or selected['validation_srcc']!=run['validation_srcc'] or model_alpha!=run['alpha']:
                raise ValueError('ridge selection/checkpoint alpha mismatch')
            ridge_sources.append({'fold':fold,'alpha':run['alpha'],'run_path':str(path),'run_sha256':sha256(path),
                                  'checkpoint_sha256':run['checkpoint_sha256']})
        if sorted(r['fold'] for r in ridge_sources)!=list(range(5)):
            raise ValueError('missing/duplicate ridge fold sources')
        policies[feature]['ridge']={'alpha_grid':RIDGE_ALPHAS,'cv_selection':'inner_validation_srcc',
            'tie_break':'smaller_alpha','standardization':'training_only_features_and_targets',
            'full_refit_rule':'median_cv_selected_alpha','cv_selected_alphas':sorted(ridge_sources,key=lambda r:r['fold']),
            'phase3_alpha':float(np.median([r['alpha'] for r in ridge_sources]))}
        if set(output['headlines'])!=set((*HEADS,'ridge')):
            raise ValueError('missing headline/deployment candidates')
        for head in (*HEADS,'ridge'):
            score=output['headlines'][head]['srcc_mean']
            if not np.isfinite(score) or not -1<=score<=1:
                raise ValueError('undefined deployment headline score')
            candidates.append({'candidate_id':f'{feature}/{head}','feature_set':feature,'head':head,
                               'headline_cv_srcc':score,'config':protocol if head=='ridge' else configs[head],
                               'refit':policies[feature][head],'cpu_latency_seconds_per_10s':None,'eligibility':'unmeasured'})
    scope=yaml.safe_load(Path('configs/scope.yaml').read_text())
    budget=scope['serving']['cpu_budget_seconds_per_10s']
    if budget!=120:
        raise ValueError('predeclared deployment CPU budget changed')
    candidates.sort(key=lambda r:(-r['headline_cv_srcc'],r['candidate_id']))
    for rank,candidate in enumerate(candidates,1):
        candidate['cv_rank']=rank
    chosen=selection['selected_feature_set']
    return {'schema_version':1,'status':'research_configuration_frozen','source_identity':context['source_identity'],
            'selected_feature_set':chosen,'configs':context['dino_configs'] if chosen=='dino' else context['resnet_configs'],
            'refit':policies[chosen],'refit_by_feature':policies,'seeds':list(range(5)),
            'provenance':context['base_provenance'],'selection_sha256':config_hash(selection),
            'cv_sources_sha256':config_hash({p:v for o in validated_runs.values() for p,v in o['source_sha256'].items()}),
            'deployment':{'rule':'highest_headline_konvid_cv_srcc_within_measured_cpu_budget',
                          'cpu_budget_seconds_per_10s':budget,'tie_break':['lower_measured_cpu_latency','stable_candidate_id'],
                          'candidates':candidates,'winner':None,'status':'unresolved_cpu_eligibility',
                          'prediction_gate':'resolve CPU eligibility/winner using KoNViD evidence before LIVE-VQC prediction'},
            'limitation':'Full-KoNViD training has more steps per epoch than CV; median epochs are an approximation.'}
