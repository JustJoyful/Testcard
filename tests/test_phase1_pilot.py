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
    monkeypatch.setattr(script,'ridge_pilot',lambda *a:{'status':'complete'})
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


def test_pilot_resume_after_protocol_freeze_never_refits(tmp_path,monkeypatch):
    import torch
    import yaml
    from pathlib import Path
    script=importlib.import_module('scripts.05_train')
    p=yaml.safe_load(Path('configs/protocol.yaml').read_text())
    p.update(device='cpu',max_epochs=2,patience=2,cpu_threads=2,results_root=str(tmp_path))
    generator=torch.Generator().manual_seed(11)
    data=[{'video_id':str(i),'features':torch.randn(2,1536,generator=generator),'mos':float(i)} for i in range(8)]
    monkeypatch.setattr(script,'role_data',lambda *a:data)
    monkeypatch.setattr(script,'require_control',lambda *a:None)
    first=script.pilot_runs(p,{'code_commit':'a'})
    monkeypatch.setattr(script,'fit_one',lambda *a,**kw:pytest.fail('pilot refitted after freeze'))
    second=script.pilot_runs({**p,'status':'frozen','tuning_seeds':[0,1]},{'code_commit':'b'})
    assert second['completed_at']==first['completed_at']
    assert second['runs']==first['runs']


def test_calendar_stop_persists_failed_gate_and_scope_proposal(tmp_path,monkeypatch):
    import json
    script=importlib.import_module('scripts.05_train')
    monkeypatch.chdir(tmp_path)
    p={'results_root':str(tmp_path/'results')}
    with pytest.raises(ValueError,match='checkpoint'):
        script.freeze_protocol(p,tmp_path/'protocol.yaml',{}, {'completed_at':'2026-10-29T12:00:00+05:30'})
    failed=json.loads((tmp_path/'results/runtime_budget.json').read_text())
    assert failed['status']=='no-go' and failed['scope_proposal']
    assert (tmp_path/'DECISIONS.md').exists()


def test_ridge_pilot_times_cpu_work_without_test_rows():
    import torch
    script=importlib.import_module('scripts.05_train')
    data=[{'video_id':str(i),'features':torch.tensor([[float(i),1.]]),'mos':float(i)} for i in range(8)]
    result=script.ridge_pilot(data,data)
    assert result['status']=='complete' and result['alpha_fits']==7
    assert result['wall_seconds']>0 and result['projected_five_fold_seconds']==5*result['wall_seconds']


def test_cli_repairs_interrupted_protocol_freeze_without_refitting(tmp_path,monkeypatch):
    import json
    import sys
    import torch
    import yaml
    from pathlib import Path
    script=importlib.import_module('scripts.05_train')
    protocol=yaml.safe_load(Path('configs/protocol.yaml').read_text())
    protocol.update(device='cpu',max_epochs=2,patience=2,cpu_threads=2,results_root=str(tmp_path))
    generator=torch.Generator().manual_seed(12)
    data=[{'video_id':str(i),'features':torch.randn(2,1536,generator=generator),'mos':float(i)} for i in range(8)]
    monkeypatch.setattr(script,'role_data',lambda *a:data)
    monkeypatch.setattr(script,'require_control',lambda *a:None)
    pilot=script.pilot_runs(protocol,{'code_commit':'a'})
    original=script.atomic_json
    original(tmp_path/'extraction.json',{'invocations':[{'wall_seconds':10.}]})
    for seed in range(5):
        original(tmp_path/'control_initial_v2'/f'seed{seed}'/'run.json',{'wall_seconds':1.})
    config_path=tmp_path/'protocol.yaml'
    config_path.write_text(yaml.safe_dump(protocol))
    def interrupted(path,value):
        if Path(path)==tmp_path/'runtime_budget.json':
            raise OSError('interrupted budget publication')
        original(path,value)
    monkeypatch.setattr(script,'atomic_json',interrupted)
    with pytest.raises(OSError):
        script.freeze_protocol(protocol,config_path,{'code_commit':'a'},pilot)
    assert yaml.safe_load(config_path.read_text())['status']=='frozen'
    monkeypatch.setattr(script,'atomic_json',original)
    monkeypatch.setattr(script,'fit_one',lambda *a,**kw:pytest.fail('CLI repair refitted pilot'))
    monkeypatch.setattr(script,'prerequisites',lambda *a:{'code_commit':'b'})
    monkeypatch.setattr(sys,'argv',['05_train.py','--mode','pilot','--config',str(config_path),'--features','configs/features_v1.yaml'])
    script.main()
    assert json.loads((tmp_path/'runtime_budget.json').read_text())['status']=='go'
    assert json.loads((tmp_path/'training_pilot.json').read_text())['completed_at']==pilot['completed_at']
