import hashlib
import importlib

import numpy as np
import pytest
from scipy.stats import kendalltau,spearmanr


@pytest.mark.parametrize('metric,statistic',[('srcc',spearmanr),('krcc',kendalltau)])
def test_paired_bootstrap_matches_independent_scipy_loop_and_sign(metric,statistic):
    from src.metrics import make_bootstrap_indices,paired_bootstrap
    y=np.arange(8.);a=np.array([0,2,1,3,5,4,7,6.]);b=np.array([1,0,3,2,6,5,4,7.])
    indices=make_bootstrap_indices(8,seed=4,draws=300)
    expected=np.array([statistic(y[i],a[i]).statistic-statistic(y[i],b[i]).statistic for i in indices])
    expected=expected[np.isfinite(expected)]
    delta=statistic(y,a).statistic-statistic(y,b).statistic
    result=paired_bootstrap(y,a,b,metric,seed=4,draws=300,resample_indices=indices)
    assert result['status']=='ok'
    assert result['delta']==pytest.approx(delta)
    assert result['ci95']==pytest.approx(np.percentile(expected,[2.5,97.5]))
    assert result['p_raw']==pytest.approx((1+np.count_nonzero(abs(expected-delta)>=abs(delta)))/(len(expected)+1))
    assert result['resample_indices_sha256']==hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
    reverse=paired_bootstrap(y,b,a,metric,seed=4,draws=300,resample_indices=indices)
    assert reverse['ci95']==pytest.approx([-result['ci95'][1],-result['ci95'][0]])
    assert reverse['p_raw']==result['p_raw']
    same=paired_bootstrap(y,a,a,metric,draws=300,resample_indices=indices)
    assert same['delta']==0 and same['ci95']==[0.,0.] and same['p_raw']==1


def test_bank_validation_undefined_draws_and_no_rng_consumption(monkeypatch):
    from src.metrics import paired_bootstrap
    y=np.arange(6.)
    bank=np.zeros((100,6),dtype=np.int64)
    bank[0]=np.arange(6)
    result=paired_bootstrap(y,y,y,'srcc',draws=100,resample_indices=bank)
    assert result['status']=='failed' and result['valid_draws']==1 and result['invalid_draws']==99
    for bad in [bank.astype(float),bank[:50],bank-1,bank+6]:
        with pytest.raises(ValueError):paired_bootstrap(y,y,y,'srcc',draws=100,resample_indices=bad)
    monkeypatch.setattr(np.random,'default_rng',lambda *a:pytest.fail('supplied bank consumed RNG'))
    assert paired_bootstrap(y,y,y,'srcc',draws=100,resample_indices=bank)['valid_draws']==1
    for a,b in [(np.ones(6),y),(y[:-1],y),(y,np.full(6,np.nan))]:
        assert paired_bootstrap(y,a,b,'srcc',draws=100,resample_indices=bank)['status']=='failed'


@pytest.mark.parametrize('interval,want',[
    ([-.005,.005],'equivalent within margin'),([-.01,.01],'equivalent within margin'),
    ([.02,.03],'A better than B (practically meaningful)'),([-.03,-.02],'B better than A (practically meaningful)'),
    ([.001,.02],'detectable difference, not shown to exceed the margin'),
    ([.01,.02],'detectable difference, not shown to exceed the margin'),
    ([0,.02],'no detectable difference (inconclusive at this sample size)'),
    ([-.02,.02],'no detectable difference (inconclusive at this sample size)')])
def test_ordered_ci_classification(interval,want):
    from src.metrics import classify_difference
    assert classify_difference(interval)==want


def test_holm_requires_full_six_pair_family_and_matches_hand_values():
    from src.metrics import holm_adjust
    raw=dict(zip('abcdef',[.001,.01,.03,.04,.2,.9]))
    assert list(holm_adjust(raw).values())==pytest.approx([.006,.05,.12,.12,.4,.9])
    for bad in [{},dict(a=.1),{**raw,'a':float('nan')}]:
        with pytest.raises(ValueError):holm_adjust(bad)


def test_comparison_cli_uses_one_bank_for_all_pairs_and_metrics(monkeypatch):
    module=importlib.import_module('scripts.07_bootstrap')
    seen=[];banks=[]
    def make(n,seed,draws):
        bank=np.tile(np.arange(n),(draws,1));banks.append(bank);return bank
    def paired(y,a,b,metric,seed,draws,*,resample_indices):
        seen.append((metric,resample_indices,id(y)))
        return {'status':'ok','metric':metric,'delta':0.,'ci95':[0.,0.],'p_raw':1.,'valid_draws':draws,'invalid_draws':0,
                'draws':draws,'seed':seed,'pvalue_method':'centered_bootstrap_add_one',
                'resample_indices_sha256':hashlib.sha256(resample_indices.astype('<i8').tobytes()).hexdigest()}
    monkeypatch.setattr(module,'make_bootstrap_indices',make)
    monkeypatch.setattr(module,'paired_bootstrap',paired)
    values={(feature,head):np.arange(8.) for feature in ('dino','resnet50')
            for head in ('mean_pool','bigru','transformer','mamba','ridge')}
    records,meta=module.compare_ensembles(np.arange(8.),values,'resnet50',draws=30)
    assert len(banks)==1 and len(seen)==30 and len(records)==15
    assert all(bank is banks[0] for _,bank,_ in seen) and len({key for _,_,key in seen})==1
    assert sum('p_holm' in r['metrics']['srcc'] for r in records)==6
    assert not any('p_holm' in r['metrics']['krcc'] for r in records)
