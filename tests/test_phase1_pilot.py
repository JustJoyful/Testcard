import importlib
import pytest


def test_runtime_budget_counts_all_five_seed_trials():
    from src.tuning import planned_fit_counts
    scope={'heads':['mean_pool','bigru','transformer','mamba'],'folds':5,'seeds':[0,1,2,3,4], 'tuning':{'trials_per_head':10}}
    counts=planned_fit_counts(scope,{'rank_candidates':[0,.1,.3,1]})
    assert counts=={'tune':200,'rank':20,'cv':100,'control_initial':5,'control_final':5,'pilot':4,'ridge':5,'ridge_alpha':35}
    smaller=planned_fit_counts(scope,{'rank_candidates':[0,.1,.3,1],'tuning_seeds':[0,1]})
    assert smaller['tune']==80 and smaller['rank']==8 and smaller['cv']==100 and smaller['control_final']==5


def test_control_failure_blocks_pilot_and_tuning(tmp_path):
    from src.tuning import require_control
    with pytest.raises(ValueError,match='control'):
        require_control(tmp_path,{})


def test_pilot_never_reads_test_role(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.05_train')
    roles=[]
    monkeypatch.setattr(script,'role_data',lambda p,f,r:roles.append(r) or [])
    def fit(*a,**kw):
        return None,{'wall_seconds':10.,'completed_epochs':2,'best_epoch':1,'history':[{'wall_seconds':5.}], 'peak_cuda_bytes':100}
    monkeypatch.setattr(script,'fit_run',fit)
    monkeypatch.setattr(script,'require_control',lambda *a:None)
    monkeypatch.setattr(script,'resolved_config',lambda p:p)
    p={'results_root':str(tmp_path),'seeds':[0,1,2,3,4]}
    script.pilot_runs(p,{})
    assert roles==['train','val']


def test_runtime_budget_accounts_for_spent_and_reserve():
    from src.tuning import runtime_budget
    scope={'heads':['mean_pool','bigru','transformer','mamba'],'folds':5,'seeds':list(range(5)),'tuning':{'trials_per_head':10}}
    pilots={h:{'wall_seconds':3600.,'completed_epochs':100,'history':[{'wall_seconds':36.}]} for h in scope['heads']}
    actual=runtime_budget(scope,{'rank_candidates':[0,.1,.3,1],'max_epochs':200},pilots,10.,20.,gpu_hours=100.)
    assert actual['remaining_head_hours']==50.
    assert actual['status']=='no-go' and actual['remaining_counts']['pilot']==0 and actual['remaining_counts']['control_initial']==0


def test_runtime_reserves_unspent_extraction_even_after_pilots():
    from src.tuning import runtime_budget
    scope={'heads':['mean_pool','bigru','transformer','mamba'],'seeds':list(range(5))}
    pilots={h:{'wall_seconds':1.,'history':[{'wall_seconds':1.}]} for h in scope['heads']}
    result=runtime_budget(scope,{'rank_candidates':[0,.1,.3,1],'max_epochs':200},pilots,10.,8.,actual_extraction_hours=4.)
    assert result['remaining_extraction_allowance_hours']==6.
    assert result['remaining_head_hours']==56.


def test_claimed_pass_cannot_hide_failed_control_numbers(tmp_path):
    from src.cache import atomic_json
    from src.tuning import require_control
    atomic_json(tmp_path/'diagnostic.json',{'status':'passed','seeds':list(range(5)),
        'provenance':{},'bootstrap':{'status':'ok','srcc':.9,'ci95':[.8,.95],'valid_draws':10000,'invalid_draws':0}})
    with pytest.raises(ValueError,match='control'):
        require_control(tmp_path,{})
