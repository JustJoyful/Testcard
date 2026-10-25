"""Bounded selection helpers and measured compute gates."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

AUDIT_INPUTS=('config_feature_sha256','split_sha256','cache_sha256','cache_meta_sha256','lock_sha256','weights_sha256')


def same_inputs(record,provenance):
    return all(record.get(k)==provenance.get(k) for k in AUDIT_INPUTS)


def require_control(directory,provenance):
    from src.features import sha256
    from src.metrics import bootstrap_srcc
    from src.training import config_hash
    directory=Path(directory)
    path=directory/'diagnostic.json'
    if not path.exists():
        raise ValueError(f'missing control gate: {path}')
    result=json.loads(path.read_text())
    bootstrap=result.get('bootstrap',{})
    score=bootstrap.get('srcc')
    interval=bootstrap.get('ci95')
    if (result.get('status')!='passed' or not same_inputs(result.get('provenance',{}),provenance)
            or result.get('seeds')!=list(range(5)) or bootstrap.get('status')!='ok'
            or score is None or not np.isfinite(score) or abs(score)>=.15
            or interval is None or len(interval)!=2 or not np.isfinite(interval).all()
            or not interval[0]<=0<=interval[1]):
        raise ValueError(f'failed/stale control gate: {path}')
    try:
        permutation=directory.parent/'control_permutation.json'
        if sha256(permutation)!=result['permutation_sha256']:
            raise ValueError('changed fixed permutation')
        outputs=[]
        for seed in range(5):
            run=json.loads((directory/f'seed{seed}'/'run.json').read_text())
            rows=json.loads((directory/f'seed{seed}'/'predictions.json').read_text())
            kind='control_final' if directory.name=='control_final' else 'control_initial'
            if (run['status']!='complete' or run['seed']!=seed or not same_inputs(run['provenance'],provenance)
                    or run['provenance']['label_mode']!='shuffled' or run['provenance']['run_kind']!=kind
                    or run['provenance']['fold']!=0 or run['provenance']['permutation_sha256']!=result['permutation_sha256']
                    or run['identity']['config_sha256']!=result['config_sha256']
                    or config_hash(run['config'])!=result['config_sha256']
                    or sha256(directory/f'seed{seed}'/'best.pt')!=run['checkpoint_sha256']
                    or [r['video_id'] for r in rows]!=result['video_ids']
                    or len(set(result['video_ids']))!=len(result['video_ids'])
                    or any(r['seed']!=seed or r['label_mode']!='shuffled' or r['run_kind']!=kind
                           or r['checkpoint_sha256']!=run['checkpoint_sha256'] for r in rows)):
                raise ValueError(f'incomplete/stale seed {seed}')
            outputs.append(rows)
        import csv
        from src.splits import freeze_splits
        config=run['config']
        freeze_splits(config['inventory'],config['splits'])
        if sha256(config['splits'])!=provenance['split_sha256']:
            raise ValueError('control split identity mismatch')
        inventory=list(csv.DictReader(Path(config['inventory']).open()))
        lookup={r['video_id']:float(r['mos']) for r in inventory}
        roles=list(csv.DictReader(Path(config['splits']).open()))
        train_ids=[r['video_id'] for r in roles if int(r['outer_fold'])==0 and r['role']=='train']
        test_ids=[r['video_id'] for r in roles if int(r['outer_fold'])==0 and r['role']=='test']
        if result['video_ids']!=test_ids:
            raise ValueError('control IDs differ from authoritative fold-0 test role')
        if any(r['mos']!=lookup[r['video_id']] for rows in outputs for r in rows):
            raise ValueError('control MOS differs from authoritative inventory')
        order=np.random.default_rng(0).permutation(len(train_ids))
        expected=[{'video_id':key,'label_source_id':train_ids[int(j)]} for key,j in zip(train_ids,order)]
        if json.loads(permutation.read_text())!=expected:
            raise ValueError('control training permutation differs from fixed seed-0 mapping')
        labels=np.array([r['mos'] for r in outputs[0]])
        ensemble=np.array([[r['prediction'] for r in rows] for rows in outputs]).mean(0)
        if any([r['mos'] for r in rows]!=labels.tolist() for rows in outputs):
            raise ValueError('inconsistent control labels')
        actual=bootstrap_srcc(labels,ensemble)
        if actual!=bootstrap or ensemble.tolist()!=result['seed_averaged_predictions']:
            raise ValueError('control diagnostic differs from saved individual predictions')
    except (OSError,KeyError,TypeError,ValueError) as exc:
        raise ValueError(f'invalid control artifacts: {path}: {exc}') from exc
    return result


def planned_fit_counts(scope,protocol):
    seeds=scope.get('seeds',list(range(5)))
    tuning=protocol.get('tuning_seeds',seeds)
    heads=scope.get('heads',['mean_pool','bigru','transformer','mamba'])
    folds=scope.get('folds',5)
    return {'tune':len(heads)*scope.get('tuning',{}).get('trials_per_head',10)*len(tuning),
            'rank':len(protocol['rank_candidates'])*len(tuning),'cv':len(heads)*folds*len(seeds),
            'control_initial':len(seeds),'control_final':len(seeds),'pilot':len(heads),
            'ridge':folds,'ridge_alpha':7*folds}


def runtime_budget(scope,protocol,pilots,extraction_hours,spent_hours,gpu_hours=100.,actual_extraction_hours=None):
    counts=planned_fit_counts(scope,protocol)
    remaining={**counts,'control_initial':0,'pilot':0}
    actual_extraction_hours=spent_hours if actual_extraction_hours is None else actual_extraction_hours
    remaining_extraction=max(0.,extraction_hours-actual_extraction_hours)
    available=gpu_hours-spent_hours-remaining_extraction-30.
    per_head={h:float(p['wall_seconds'])/3600 for h,p in pilots.items()}
    mean=per_head['mean_pool']
    per_head_fits=remaining['tune']/len(pilots)+remaining['cv']/len(pilots)
    estimate=sum(per_head.values())*per_head_fits+mean*(remaining['rank']+remaining['control_final'])
    maximum={h:np.mean([e['wall_seconds'] for e in p['history']])*protocol['max_epochs']/3600 for h,p in pilots.items()}
    upper=sum(maximum.values())*per_head_fits+maximum['mean_pool']*(remaining['rank']+remaining['control_final'])
    return {'status':'go' if 2*estimate<=available else 'no-go','planned_counts':counts,'remaining_counts':remaining,
            'spent_gpu_hours':spent_hours,'remaining_extraction_allowance_hours':remaining_extraction,
            'reserved_gpu_hours':30.,'remaining_head_hours':available,
            'optimistic_hours':estimate,'planning_2x_hours':2*estimate,'max_epoch_upper_hours':upper,
            'assumption':'one complete observed fit per head; 2x planning gate; initial control and pilots already spent',
            'ridge_cpu_alpha_fits':counts['ridge_alpha']}


def calendar_gate(pilot_completed=None):
    deadline=datetime(2026,10,28,23,59,59,tzinfo=ZoneInfo('Asia/Kolkata'))
    observed=pilot_completed or datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
    if datetime.fromisoformat(observed)>deadline:
        raise ValueError('October 28 pilot checkpoint missed: record concrete scope cut/revised schedule before tuning')


def make_trials(protocol,count=10,seed=0):
    if count!=10 or seed!=0:
        raise ValueError('ten ordered seed-0 candidates required')
    rng=np.random.default_rng(seed)
    return [{'candidate_id':i,'learning_rate':float(np.exp(rng.uniform(np.log(1e-4),np.log(3e-3)))),
             'weight_decay':float(np.exp(rng.uniform(np.log(1e-6),np.log(1e-2)))),
             'input_dropout':float(rng.choice([0,.1,.2])),'head_dropout':float(rng.choice([0,.1,.2]))} for i in range(count)]


def select_trial(rows,expected_seeds):
    if not rows:
        raise ValueError('incomplete trial rows')
    candidates=sorted({r['candidate_id'] for r in rows})
    scores=[]
    for candidate in candidates:
        members=[r for r in rows if r['candidate_id']==candidate]
        if (sorted(r['seed'] for r in members)!=sorted(expected_seeds)
                or any(r['status']!='complete' or r.get('validation_srcc') is None or not np.isfinite(r['validation_srcc']) for r in members)):
            raise ValueError(f'incomplete/failed candidate {candidate}: every declared seed required')
        scores.append({'candidate_id':candidate,'mean_validation_srcc':float(np.mean([r['validation_srcc'] for r in members]))})
    return max(scores,key=lambda r:(r['mean_validation_srcc'],-r['candidate_id']))


def require_selection_gates(protocol,provenance,need_rank=False):
    from src.training import config_hash
    root=Path(protocol['results_root'])
    if protocol['status']!='frozen':
        raise ValueError('protocol must be frozen by measured runtime gate')
    require_control(root/'control_initial',provenance)
    path=root/'runtime_budget.json'
    if not path.exists():
        raise ValueError('missing measured runtime budget')
    budget=json.loads(path.read_text())
    if (budget['status']!='go' or budget['protocol_sha256']!=config_hash(protocol) or not same_inputs(budget['provenance'],provenance)
            or budget.get('reserved_gpu_hours')!=30 or budget.get('planning_2x_hours',float('inf'))>budget.get('remaining_head_hours',0)):
        raise ValueError('failed/stale runtime budget')
    pilot=json.loads((root/'training_pilot.json').read_text())
    from src.features import sha256
    if (pilot.get('status')!='complete' or not same_inputs(pilot.get('provenance',{}),provenance)
            or budget.get('pilot_sha256')!=sha256(root/'training_pilot.json')
            or set(pilot.get('runs',{}))!={'mean_pool','bigru','transformer','mamba'}
            or budget.get('planned_counts')!=planned_fit_counts({'seeds':list(range(5))},protocol)):
        raise ValueError('invalid actual pilot or runtime fit counts')
    calendar_gate(pilot['completed_at'])
    if need_rank:
        path=Path('configs/rank_loss.yaml')
        if not path.exists():
            raise ValueError('missing rank freeze')
        import yaml
        rank=yaml.safe_load(path.read_text())
        if rank['status']!='frozen' or rank['protocol_sha256']!=config_hash(protocol) or not same_inputs(rank['provenance'],provenance):
            raise ValueError('stale rank freeze')
        return rank
