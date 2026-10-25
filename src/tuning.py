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
    path=Path(directory)/'diagnostic.json'
    if not path.exists():
        raise ValueError(f'missing control gate: {path}')
    result=json.loads(path.read_text())
    if (result.get('status')!='passed' or not same_inputs(result.get('provenance',{}),provenance)
            or result.get('seeds')!=list(range(5))):
        raise ValueError(f'failed/stale control gate: {path}')
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
