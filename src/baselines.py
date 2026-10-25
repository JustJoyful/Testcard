"""Deterministic dual ridge, standardized using training rows only."""
import numpy as np
from scipy.linalg import solve
from src.metrics import correlations, fit_target_stats


def fit_ridge(X_train,y_train,X_val,y_val,alphas):
    x,v=np.asarray(X_train,dtype=float),np.asarray(X_val,dtype=float)
    y,yv=np.asarray(y_train,dtype=float),np.asarray(y_val,dtype=float)
    if x.ndim!=2 or v.ndim!=2 or v.shape[1]!=x.shape[1] or len(x)!=len(y) or len(v)!=len(yv) or not np.isfinite(x).all() or not np.isfinite(v).all():
        raise ValueError('invalid ridge features/targets')
    stats=fit_target_stats(y)
    mean=x.mean(0)
    scale=x.std(0)
    scale[scale==0]=1.
    z=(x-mean)/scale
    target=(y-stats['mean'])/stats['std']
    kernel=z@z.T
    best=None
    trials=[]
    if not alphas or any(not np.isfinite(a) or a<=0 for a in alphas):
        raise ValueError('positive finite ridge alphas required')
    for alpha in sorted(alphas):
        dual=solve(kernel+alpha*np.eye(len(x)),target,assume_a='pos')
        coefficient=z.T@dual
        model={'feature_mean':mean,'feature_scale':scale,'coefficient':coefficient,
               'target_mean':stats['mean'],'target_std':stats['std'],'alpha':float(alpha)}
        score=correlations(yv,predict_ridge(model,v))
        trials.append({'alpha':float(alpha),'validation_srcc':score['srcc'],'status':score['status']})
        if score['status']=='ok' and (best is None or score['srcc']>best['validation_srcc']):
            best={**model,'validation_srcc':score['srcc']}
    if best is None:
        raise ValueError('ridge has no defined validation SRCC')
    return {**best,'trials':trials}


def predict_ridge(artifact,X):
    x=np.asarray(X,dtype=float)
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('invalid ridge prediction features')
    return ((x-artifact['feature_mean'])/artifact['feature_scale']@artifact['coefficient'])*artifact['target_std']+artifact['target_mean']
