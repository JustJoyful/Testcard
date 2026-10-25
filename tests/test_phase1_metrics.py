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
