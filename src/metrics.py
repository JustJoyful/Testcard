"""Explicitly failed diagnostics for undefined correlations."""
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr


def fit_target_stats(mos):
    mos=np.asarray(mos,dtype=float)
    if mos.ndim!=1 or len(mos)<2 or not np.isfinite(mos).all() or np.std(mos)==0:
        raise ValueError('training MOS must be finite with positive variance')
    return {'mean':float(np.mean(mos)),'std':float(np.std(mos))}


def correlations(y_true,y_pred):
    y,p=np.asarray(y_true,dtype=float),np.asarray(y_pred,dtype=float)
    result={'srcc':None,'krcc':None,'plcc_raw':None,'status':'failed','reason':None}
    if y.ndim!=1 or y.shape!=p.shape or len(y)<2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        return {**result,'reason':'mismatched, insufficient or nonfinite values'}
    if np.ptp(y)==0 or np.ptp(p)==0:
        return {**result,'reason':'constant target or prediction'}
    values=[float(f(y,p).statistic) for f in (spearmanr,kendalltau,pearsonr)]
    if not np.isfinite(values).all():
        return {**result,'reason':'undefined correlation'}
    return dict(zip(('srcc','krcc','plcc_raw'),values),status='ok',reason=None)
