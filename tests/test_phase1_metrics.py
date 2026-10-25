import numpy as np
import pytest


@pytest.mark.parametrize('sign',[1,-1])
def test_rank_correlations(sign):
    from src.metrics import correlations
    result=correlations(np.arange(6),sign*np.arange(6))
    assert result['srcc']==pytest.approx(sign) and result['krcc']==pytest.approx(sign)


@pytest.mark.parametrize('values',[[1,1,1],[1,float('nan'),3]])
def test_undefined_correlations_fail(values):
    from src.metrics import correlations
    actual=correlations(np.arange(3),values)
    assert actual['status']=='failed' and actual['srcc'] is None and actual['reason']


def test_constant_training_mos_fails():
    from src.metrics import fit_target_stats
    with pytest.raises(ValueError):
        fit_target_stats(np.ones(4))


def test_logistic_mapping_is_disclosed_and_finite():
    from src.metrics import fit_logistic_plcc
    x=np.linspace(-2,2,20)
    actual=fit_logistic_plcc(2+3/(1+np.exp(-x)),x)
    assert actual['status']=='ok' and actual['plcc_mapped']>.999
    assert actual['fitting_domain']=='target_labels' and len(actual['coefficients'])==4


def test_constant_prediction_mapping_fails_explicitly():
    from src.metrics import fit_logistic_plcc
    assert fit_logistic_plcc(np.arange(10),np.ones(10))['status']=='failed'


def test_bootstrap_is_seeded_and_retains_invalid_draw_counts():
    from src.metrics import bootstrap_srcc
    a=bootstrap_srcc(np.arange(6),np.arange(6),draws=300)
    b=bootstrap_srcc(np.arange(6),np.arange(6),draws=300)
    assert a==b and a['valid_draws']+a['invalid_draws']==300
    assert bootstrap_srcc(np.ones(4),np.arange(4),draws=100)['status']=='failed'
