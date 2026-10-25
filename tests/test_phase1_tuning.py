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
