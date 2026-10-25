import importlib
import numpy as np
import pytest


def cosine():
    return importlib.import_module('scripts.04_parity_check').minimum_cosine


def test_cosine_boundary():
    reference=np.array([[1.,0.]])
    assert cosine()(reference,reference)==pytest.approx(1.)
    assert cosine()(reference,np.array([[.999,np.sqrt(1-.999**2)]]))>=.999
    assert cosine()(reference,np.array([[.998,np.sqrt(1-.998**2)]]))<.999


@pytest.mark.parametrize('candidate',[np.zeros((1,2)),np.array([[np.nan,1]]),np.ones((2,2))])
def test_zero_nan_missing_timestep_fails(candidate):
    with pytest.raises(ValueError):
        cosine()(np.ones((1,2)),candidate)


def test_one_bad_timestep_cannot_be_averaged_away():
    reference=np.tile([1.,0.],(100,1))
    candidate=reference.copy();candidate[-1]=[.998,np.sqrt(1-.998**2)]
    assert cosine()(reference,candidate)<.999
