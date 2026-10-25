import numpy as np
import pytest


def test_ridge_standardizes_from_training_only():
    from src.baselines import fit_ridge, predict_ridge
    x=np.array([[0.,1.],[1.,1.],[2.,1.],[3.,1.]])
    y=np.arange(4.)
    model=fit_ridge(x,y,x+100,y,[.001,.1,1.])
    np.testing.assert_allclose(model['feature_mean'],[1.5,1.])
    assert model['feature_scale'][1]==1.
    assert np.isfinite(predict_ridge(model,x)).all()


def test_alpha_selected_by_validation_not_test():
    from src.baselines import fit_ridge
    x=np.arange(10.).reshape(-1,1)
    model=fit_ridge(x,x[:,0],x,x[:,0],[1.,.001,.1])
    assert model['alpha']==.001


def test_ridge_deterministic():
    from src.baselines import fit_ridge,predict_ridge
    x=np.random.default_rng(0).normal(size=(10,4));y=np.arange(10.)
    a=fit_ridge(x,y,x,y,[.1,1]);b=fit_ridge(x,y,x,y,[.1,1])
    np.testing.assert_array_equal(predict_ridge(a,x),predict_ridge(b,x))
