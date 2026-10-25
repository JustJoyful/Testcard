"""Regenerate fixed-video CPU fp32 features and compare every cached GPU timestep."""
import argparse
import csv
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cache import BRANCHES, atomic_json, read_branch, read_manifest, validate_freeze
from src.features import extract_features,load_backbone,sha256
from src.preprocess import iter_samples,spatial_tensor


def minimum_cosine(reference,candidate):
    a,b=np.asarray(reference,dtype=np.float64),np.asarray(candidate,dtype=np.float64)
    if a.ndim!=2 or a.shape!=b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('parity requires matching finite nonempty [T,D] arrays')
    norms=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    if np.any(norms==0) or not np.isfinite(norms).all():
        raise ValueError('parity rejects zero/nonfinite feature norms')
    return float(np.clip(np.sum(a*b,axis=1)/norms,-1.,1.).min())


def validate_parity(path,provenance,root):
    path,root=Path(path),Path(root)
    if not path.exists():
        raise ValueError('missing Phase 1 hardware parity')
    record=json.loads(path.read_text())
    if (record.get('status')!='passed' or record.get('reference_device')!='cuda' or record.get('candidate_device')!='cpu'
            or record.get('reference_precision')!='float16' or record.get('candidate_precision')!='float32'
            or record.get('feature_config_sha256')!=provenance.get('config_feature_sha256')
            or record.get('lock_sha256')!=provenance.get('lock_sha256')
            or record.get('weights_sha256')!=provenance.get('weights_sha256')
            or record.get('cache_meta_sha256')!=sha256(root/'meta.json')):
        raise ValueError('failed/stale hardware parity identity')
    sample=yaml.safe_load(Path('configs/phase0.yaml').read_text())['samples']['konvid']
    expected={(key,branch) for key in sample for branch in BRANCHES}
    actual=[(r['video_id'],r['branch']) for r in record['comparisons']]
    if len(sample)!=20 or len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('missing/duplicate parity video or branch')
    sources={r['video_id']:r for r in csv.DictReader(Path('data/manifests/konvid.csv').open())}
    rows=read_manifest(root)
    for r in record['comparisons']:
        key,branch=r['video_id'],r['branch']
        a=read_branch(root,'konvid',key,branch,rows,sources[key])
        if (r['status']!='passed' or not np.isfinite(r['minimum_cosine']) or r['minimum_cosine']<.999
                or r['minimum_cosine']>1 or r['T']!=len(a) or r['source_sha256']!=sources[key]['source_sha256']
                or r['reference_sha256']!=sha256(root/'konvid'/branch/(key+'.npy'))):
            raise ValueError(f'failed/stale parity: {key}/{branch}')
    return {'status':'passed','comparisons':len(actual),'minimum_cosine':min(r['minimum_cosine'] for r in record['comparisons'])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--sample-ids',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    args=p.parse_args()
    frozen=validate_freeze(args.features)
    config=yaml.safe_load(args.features.read_text())
    ids=args.sample_ids.read_text().splitlines()
    expected=yaml.safe_load(Path('configs/phase0.yaml').read_text())['samples']['konvid']
    if ids!=expected or len(ids)!=20:
        raise ValueError('parity requires the original fixed 20 IDs')
    sources={r['video_id']:r for r in csv.DictReader(Path('data/manifests/konvid.csv').open())}
    extractor=importlib.import_module('scripts.02_extract_spatial')
    torch.set_num_threads(4)
    rows=read_manifest(args.cache)
    comparisons=[]
    started=time.perf_counter()
    for backbone in ('dino','resnet50'):
        model=load_backbone(backbone,'cpu',config)
        branches=[b for b in BRANCHES if (b.startswith('dino_') if backbone=='dino' else b=='resnet50')]
        for key in ids:
            try:
                references={b:read_branch(args.cache,'konvid',key,b,rows,sources[key]) for b in branches}
                values={b:[] for b in branches}
                with extractor.source_video(sources[key],config['datasets']['konvid']) as path:
                    for _,frame,_ in iter_samples(path,config):
                        features=extract_features(model,spatial_tensor(frame).unsqueeze(0),backbone,'cpu')
                        for name,value in features.items():
                            branch='dino_'+name if backbone=='dino' else 'resnet50'
                            values[branch].append(value.numpy())
                for branch in branches:
                    candidate=np.concatenate(values[branch]).astype(np.float32,copy=False)
                    score=minimum_cosine(references[branch],candidate)
                    comparisons.append({'video_id':key,'branch':branch,'T':len(candidate),'minimum_cosine':score,
                        'status':'passed' if score>=.999 else 'failed','source_sha256':sources[key]['source_sha256'],
                        'reference_sha256':sha256(args.cache/'konvid'/branch/(key+'.npy')),
                        'candidate_array_sha256':hashlib.sha256(candidate.tobytes()).hexdigest()})
                print(f'parity {key}/{backbone}: {min(r["minimum_cosine"] for r in comparisons if r["video_id"]==key):.9f}',flush=True)
            except Exception as exc:
                comparisons.append({'video_id':key,'branch':backbone,'status':'failed','reason':str(exc)})
    passed=len(comparisons)==160 and all(r['status']=='passed' for r in comparisons)
    record={'status':'passed' if passed else 'failed','comparisons':comparisons,'reference_device':'cuda','candidate_device':'cpu',
            'reference_precision':'float16','candidate_precision':'float32','feature_config_sha256':sha256(args.features),
            'lock_sha256':frozen['lock_sha256'],'weights_sha256':frozen['weights_sha256'],
            'cache_meta_sha256':sha256(args.cache/'meta.json'),'cache_manifest_sha256_at_measurement':sha256(args.cache/'manifest.csv'),
            'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'wall_seconds':time.perf_counter()-started}
    atomic_json('results/phase1/parity.json',record)
    return not passed


if __name__=='__main__':
    raise SystemExit(main())
