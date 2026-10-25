import pytest


def test_trial_requires_every_declared_tuning_seed():
    from src.tuning import select_trial
    rows=[{'candidate_id':0,'seed':0,'validation_srcc':.5,'status':'complete'}]
    for seeds in ([0,1,2,3,4],[0,1]):
        with pytest.raises(ValueError,match='incomplete|seed'):
            select_trial(rows,seeds)


def test_ten_same_candidates_per_head():
    from src.tuning import make_trials
    trials=make_trials({})
    assert len(trials)==10 and trials==make_trials({})
    assert len({tuple(sorted(t.items())) for t in trials})==10
    assert all(1e-4<=t['learning_rate']<=3e-3 and 1e-6<=t['weight_decay']<=1e-2 for t in trials)


def test_selection_ignores_test_metrics_and_ties_by_id():
    from src.tuning import select_trial
    rows=[{'candidate_id':c,'seed':s,'validation_srcc':.5,'test_srcc':float(c),'status':'complete'} for c in [1,0] for s in [0,1]]
    assert select_trial(rows,[0,1])['candidate_id']==0
    rows[0]['test_srcc']=-999
    assert select_trial(rows,[0,1])['candidate_id']==0
    rows[0]['status']='failed'
    with pytest.raises(ValueError):
        select_trial(rows,[0,1])


def test_rank_frozen_before_head_tuning(tmp_path):
    from src.tuning import require_selection_gates
    with pytest.raises(ValueError,match='frozen|control|budget|rank'):
        require_selection_gates({'status':'draft','results_root':str(tmp_path)},{},need_rank=True)


def test_rank_resume_after_audit_commit_preserves_selection(tmp_path,monkeypatch):
    import importlib
    script=importlib.import_module('scripts.05_train')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script,'require_selection_gates',lambda *a,**kw:None)
    monkeypatch.setattr(script,'resolved_config',lambda p:p)
    monkeypatch.setattr(script,'role_data',lambda *a:[])
    monkeypatch.setattr(script,'fit_run',lambda *a,**kw:(None,{'status':'complete','best_validation_srcc':.5}))
    p={'results_root':str(tmp_path/'results'),'rank_candidates':[0,.1,.3,1],'tuning_seeds':[0,1]}
    script.rank_runs(p,{'code_commit':'a','split_sha256':'same'})
    path=tmp_path/'configs/rank_loss.yaml'
    before=path.read_bytes()
    script.rank_runs(p,{'code_commit':'b','split_sha256':'same'})
    assert path.read_bytes()==before


def test_changed_selection_input_still_rejects(tmp_path):
    import importlib
    persist=importlib.import_module('scripts.05_train').persist_yaml
    persist(tmp_path/'selection.yaml',{'provenance':{'code_commit':'a','split_sha256':'a'},'rank_weight':.1})
    with pytest.raises(ValueError,match='incompatible'):
        persist(tmp_path/'selection.yaml',{'provenance':{'code_commit':'b','split_sha256':'b'},'rank_weight':.1})


def test_partial_head_tuning_resumes_across_commit(tmp_path,monkeypatch):
    import importlib
    import json
    script=importlib.import_module('scripts.05_train')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(script,'require_selection_gates',lambda *a,**kw:{'rank_weight':.1})
    monkeypatch.setattr(script,'resolved_config',lambda p:p)
    roles=[]
    monkeypatch.setattr(script,'role_data',lambda p,f,r:roles.append(r) or [])
    new_fits=[]
    interrupt=[True]
    def fake_fit(protocol,config,provenance,head,fold,seed,output,*a,**kw):
        if head=='bigru' and interrupt[0]:
            raise RuntimeError('interrupted after mean-pool completed')
        path=output/'run.json'
        if path.exists():
            return None,json.loads(path.read_text())
        new_fits.append(head)
        output.mkdir(parents=True,exist_ok=True)
        result={'status':'complete','best_validation_srcc':.5}
        path.write_text(json.dumps(result))
        return None,result
    monkeypatch.setattr(script,'fit_run',fake_fit)
    protocol={'results_root':str(tmp_path/'results'),'tuning_seeds':[0,1]}
    with pytest.raises(RuntimeError):
        script.tune_runs(protocol,{'code_commit':'a','split_sha256':'same'})
    before=(tmp_path/'configs/heads/mean_pool.yaml').read_bytes()
    interrupt[0]=False
    script.tune_runs(protocol,{'code_commit':'b','split_sha256':'same'})
    assert new_fits.count('mean_pool')==20 and len(new_fits)==80
    assert (tmp_path/'configs/heads/mean_pool.yaml').read_bytes()==before
    assert len(list((tmp_path/'configs/heads').glob('*.yaml')))==4
    assert set(roles)=={'train','val'}
