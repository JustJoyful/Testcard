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


def make_bootstrap_indices(n_videos,seed=0,draws=10000):
    if type(n_videos) is not int or n_videos<2 or type(draws) is not int or draws<1:
        raise ValueError('positive draw count and at least two videos required')
    return np.random.default_rng(seed).integers(0,n_videos,size=(draws,n_videos),dtype=np.int64)


def bootstrap_indices_hash(indices):
    import hashlib
    return hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest()


def paired_bootstrap(y_true,pred_a,pred_b,metric,seed=0,draws=10000,*,resample_indices=None):
    from scipy.stats import rankdata
    if metric not in ('srcc','krcc'):
        raise ValueError('paired metric must be srcc or krcc')
    y,a,b=(np.asarray(v,dtype=float) for v in (y_true,pred_a,pred_b))
    result={'status':'failed','reason':None,'metric':metric,'delta':None,'ci95':None,'p_raw':None,
            'valid_draws':0,'invalid_draws':draws,'seed':seed,'draws':draws,
            'pvalue_method':'centered_bootstrap_add_one','resample_indices_sha256':None}
    ca,cb=correlations(y,a),correlations(y,b)
    if ca['status']!='ok' or cb['status']!='ok':
        return {**result,'reason':'constant, nonfinite, insufficient or misaligned inputs'}
    indices=make_bootstrap_indices(len(y),seed,draws) if resample_indices is None else np.asarray(resample_indices)
    if (indices.dtype.kind!='i' or indices.dtype.itemsize!=8 or indices.shape!=(draws,len(y))
            or np.any(indices<0) or np.any(indices>=len(y))):
        raise ValueError('invalid shared bootstrap index bank dtype/shape/bounds')
    result['resample_indices_sha256']=bootstrap_indices_hash(indices)
    delta=ca[metric]-cb[metric]
    values=[]
    for start in range(0,draws,500):
        bank=indices[start:start+500]
        if metric=='srcc':
            ranks=[rankdata(v[bank],axis=1) for v in (y,a,b)]
            for r in ranks:
                r-=r.mean(1,keepdims=True)
            def rho(r):
                denominator=np.sqrt((ranks[0]**2).sum(1)*(r**2).sum(1))
                return np.divide((ranks[0]*r).sum(1),denominator,out=np.full(len(bank),np.nan),where=denominator>0)
            samples=rho(ranks[1])-rho(ranks[2])
        else:
            samples=np.array([kendalltau(y[i],a[i]).statistic-kendalltau(y[i],b[i]).statistic for i in bank])
        values.extend(samples[np.isfinite(samples)].tolist())
    result.update(delta=float(delta),valid_draws=len(values),invalid_draws=draws-len(values))
    if len(values)<.99*draws:
        return {**result,'reason':'fewer than 99% finite bootstrap draws'}
    values=np.asarray(values)
    return {**result,'status':'ok','ci95':np.percentile(values,[2.5,97.5]).tolist(),
            'p_raw':float((1+np.count_nonzero(abs(values-delta)>=abs(delta)))/(len(values)+1))}


def classify_difference(ci95,margin=.01):
    if len(ci95)!=2 or not np.isfinite(ci95).all() or ci95[0]>ci95[1] or not np.isfinite(margin) or margin<=0:
        raise ValueError('ordered finite CI and positive margin required')
    lo,hi=ci95
    if -margin<=lo<=hi<=margin:
        return 'equivalent within margin'
    if lo>margin:
        return 'A better than B (practically meaningful)'
    if hi < -margin:
        return 'B better than A (practically meaningful)'
    if lo>0 or hi<0:
        return 'detectable difference, not shown to exceed the margin'
    return 'no detectable difference (inconclusive at this sample size)'


def holm_adjust(pvalues):
    if len(pvalues)!=6 or any(not np.isfinite(p) or not 0<=p<=1 for p in pvalues.values()):
        raise ValueError('exactly six finite primary SRCC p values required')
    adjusted={}
    previous=0.
    for i,(key,p) in enumerate(sorted(pvalues.items(),key=lambda kv:(kv[1],kv[0]))):
        previous=max(previous,min(1.,(6-i)*p))
        adjusted[key]=float(previous)
    return {key:adjusted[key] for key in pvalues}
