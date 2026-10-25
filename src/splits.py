"""Immutable rank-stratified outer and inner roles."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.cache import atomic_json, atomic_write, csv_bytes
from src.features import sha256

ALGORITHM = 'rank-quantile-5-roundrobin-largest-remainder-v1'


def make_splits(rows, seed=0):
    if seed != 0:
        raise ValueError('frozen split seed must be 0')
    ordered = sorted(rows,key=lambda r:(float(r['mos']),r['video_id']))
    if len(ordered)<25 or len({r['video_id'] for r in rows})!=len(rows):
        raise ValueError('insufficient or duplicate inventory IDs')
    if any(not np.isfinite(float(r['mos'])) or r.get('status','ok')!='ok' for r in rows):
        raise ValueError('invalid inventory MOS/status')
    rng=np.random.default_rng(seed)
    strata=[list(x) for x in np.array_split(np.arange(len(ordered)),5)]
    outer={}
    for stratum in strata:
        rng.shuffle(stratum)
        for position,index in enumerate(stratum):
            outer[ordered[index]['video_id']]=position%5
    result=[]
    for fold in range(5):
        pools=[[ordered[i]['video_id'] for i in s if outer[ordered[i]['video_id']]!=fold] for s in strata]
        sizes=np.array([len(p) for p in pools])
        desired=int(np.floor(sum(sizes)*.15+.5))
        quotas=sizes*.15
        allocation=np.floor(quotas).astype(int)
        for i in sorted(range(5),key=lambda i:(-(quotas[i]-allocation[i]),i))[:desired-sum(allocation)]:
            allocation[i]+=1
        inner=np.random.default_rng(np.random.SeedSequence([seed,fold]))
        validation=set()
        for pool,n in zip(pools,allocation):
            inner.shuffle(pool)
            validation.update(pool[:n])
        for key in sorted(outer):
            result.append({'outer_fold':fold,'video_id':key,
                           'role':'test' if outer[key]==fold else ('val' if key in validation else 'train')})
    return result


def validate_splits(splits, inventory):
    ids={r['video_id'] for r in inventory}
    if len(ids)!=len(inventory) or len(splits)!=5*len(ids):
        raise ValueError('split/inventory coverage mismatch')
    for fold in range(5):
        rows=[r for r in splits if int(r['outer_fold'])==fold]
        if Counter(r['video_id'] for r in rows)!=Counter({key:1 for key in ids}):
            raise ValueError(f'fold {fold}: duplicate/missing IDs')
        if {r['role'] for r in rows}!={'train','val','test'}:
            raise ValueError('invalid/empty split role')
    if Counter(r['video_id'] for r in splits if r['role']=='test')!=Counter({key:1 for key in ids}):
        raise ValueError('each video must be test exactly once')


def freeze_splits(manifest, output, seed=0):
    manifest,output=Path(manifest),Path(output)
    meta_path=output.with_suffix('.meta.json')
    rows=list(csv.DictReader(manifest.open()))
    if output.exists() or meta_path.exists():
        if not output.exists() or not meta_path.exists():
            raise ValueError('incomplete immutable splits; investigate, never regenerate')
        meta=json.loads(meta_path.read_text())
        if (meta['inventory_sha256']!=sha256(manifest) or meta['split_sha256']!=sha256(output)
                or meta['algorithm']!=ALGORITHM or meta['seed']!=seed):
            raise ValueError('stale/changed immutable splits')
        splits=list(csv.DictReader(output.open()))
        validate_splits(splits,rows)
        return meta
    splits=make_splits(rows,seed)
    validate_splits(splits,rows)
    atomic_write(output,csv_bytes(splits,['outer_fold','video_id','role']))
    meta={'schema_version':1,'algorithm':ALGORITHM,'seed':seed,'inventory_sha256':sha256(manifest),
          'split_sha256':sha256(output),'role_rows':len(splits),
          'fold_counts':{str(f):dict(Counter(r['role'] for r in splits if r['outer_fold']==f)) for f in range(5)}}
    atomic_json(meta_path,meta)
    return meta
