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


def fit_logistic_plcc(y_true,y_pred):
    from scipy.optimize import least_squares
    from scipy.special import expit
    y,x=np.asarray(y_true,dtype=float),np.asarray(y_pred,dtype=float)
    result={'status':'failed','reason':None,'plcc_mapped':None,'coefficients':None,'fitting_domain':'target_labels'}
    if correlations(y,x)['status']!='ok' or len(y)<4:
        return {**result,'reason':'insufficient, constant or nonfinite mapping inputs'}
    tiny=max(np.ptp(x)*1e-8,1e-10)
    def mapping(b):
        return b[1]+(b[0]-b[1])*expit((x-b[2])/b[3])
    initial=[float(y.max()),float(y.min()),float(np.median(x)),max(float(np.std(x)),tiny*2)]
    if pearsonr(y,x).statistic<0:
        initial[0],initial[1]=initial[1],initial[0]
    try:
        fit=least_squares(lambda b:mapping(b)-y,initial,
                          bounds=([-np.inf,-np.inf,-np.inf,tiny],[np.inf,np.inf,np.inf,np.inf]),max_nfev=5000)
        mapped=mapping(fit.x)
        score=correlations(y,mapped)
        if not fit.success or score['status']!='ok':
            return {**result,'reason':f'logistic fit failed: {fit.message}'}
        return {**result,'status':'ok','reason':None,'plcc_mapped':score['plcc_raw'],
                'coefficients':fit.x.tolist(),'convergence':str(fit.message)}
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        return {**result,'reason':str(exc)}


def bootstrap_srcc(y_true,predictions,seed=0,draws=10000):
    from scipy.stats import rankdata
    y,p=np.asarray(y_true,dtype=float),np.asarray(predictions,dtype=float)
    point=correlations(y,p)
    result={'srcc':point['srcc'],'ci95':None,'valid_draws':0,'invalid_draws':draws,'status':'failed','reason':point['reason']}
    if point['status']!='ok' or draws<1:
        return result
    rng=np.random.default_rng(seed)
    values=[]
    for start in range(0,draws,500):
        indices=rng.integers(0,len(y),size=(min(500,draws-start),len(y)))
        a,b=rankdata(y[indices],axis=1),rankdata(p[indices],axis=1)
        a-=a.mean(1,keepdims=True); b-=b.mean(1,keepdims=True)
        norm=np.sqrt((a*a).sum(1)*(b*b).sum(1))
        good=norm>0
        values.extend(((a*b).sum(1)[good]/norm[good]).tolist())
    result.update(valid_draws=len(values),invalid_draws=draws-len(values))
    if len(values)<.99*draws:
        return {**result,'reason':'fewer than 99% finite bootstrap draws'}
    return {**result,'status':'ok','reason':None,'ci95':np.percentile(values,[2.5,97.5]).tolist()}
