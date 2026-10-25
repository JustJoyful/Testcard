"""Extract frozen spatial branches; one manually scheduled writer only."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cache import (BRANCHES, DINO_BRANCHES, atomic_json, atomic_write, coverage, read_manifest,
                       reusable, safe_id, validate_freeze, write_video_features)
from src.features import extract_features, load_backbone, sha256
from src.preprocess import iter_samples, spatial_tensor


@contextmanager
def source_video(row, settings):
    safe_id(row['dataset'],row['video_id'])
    member = PurePosixPath(row['relative_path'])
    if member.is_absolute() or '..' in member.parts or '\\' in str(member) or ':' in str(member):
        raise ValueError(f'{row["video_id"]}: unsafe archive member')
    raw = Path(settings['raw_dir'])/(row['video_id']+'.mp4')
    if raw.exists():
        if raw.stat().st_size != int(row['source_size_bytes']) or sha256(raw) != row['source_sha256']:
            raise ValueError(f'{row["video_id"]}: raw source mismatch; retained')
        yield raw
        return
    with tempfile.TemporaryDirectory(prefix='spatial-') as temporary:
        path = Path(temporary)/member.name
        with zipfile.ZipFile(settings['archive']) as archive:
            info = archive.getinfo(str(member))
            if (info.external_attr >> 16) & 0o170000 == 0o120000 or info.file_size != int(row['source_size_bytes']):
                raise ValueError('unsafe member or source size mismatch')
            with archive.open(info) as source, path.open('wb') as target:
                shutil.copyfileobj(source,target)
        if sha256(path) != row['source_sha256']:
            raise ValueError(f'{row["video_id"]}: source checksum mismatch')
        yield path


def extract_video(path, config, model, backbone, device):
    result = {}
    for _, frame, _ in iter_samples(path,config):
        values = extract_features(model,spatial_tensor(frame).unsqueeze(0),backbone,device)
        for key, value in values.items():
            branch = 'dino_'+key if backbone == 'dino' else 'resnet50'
            result.setdefault(branch,[]).append(value.cpu().numpy().astype(np.float16))
    if not result:
        raise ValueError(f'{path}: no samples')
    return {key: np.concatenate(values) for key, values in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--dataset',choices=['konvid','livevqc'],required=True)
    parser.add_argument('--backbone',choices=['dino','resnet50'],required=True)
    parser.add_argument('--sample-ids',type=Path)
    args = parser.parse_args()
    frozen = validate_freeze(args.config)
    config = yaml.safe_load(args.config.read_text())
    inventories = {d: list(csv.DictReader(Path(f'data/manifests/{d}.csv').open())) for d in config['datasets']}
    if any(r['status'] != 'ok' for rows in inventories.values() for r in rows):
        raise ValueError('failed source inventory')
    sample = '\n'.join(yaml.safe_load(Path('configs/phase0.yaml').read_text())['samples']['konvid'])+'\n'
    sample_path = Path('data/splits/phase0_konvid_ids.txt')
    if sample_path.exists() and sample_path.read_text() != sample:
        raise ValueError('fixed sample list changed')
    if not sample_path.exists():
        atomic_write(sample_path,sample.encode())
    root = Path('cache')/config['cache_version']
    if (root/'meta.json').exists():
        meta = json.loads((root/'meta.json').read_text())
        if meta['feature_config_sha256'] != frozen['feature_config_sha256']:
            raise ValueError('cache config changed')
    else:
        meta = {'schema_version':1,'feature_config_sha256':sha256(args.config),'lock_sha256':frozen['lock_sha256'],
                'weights_sha256':frozen['weights_sha256'],'feature_config':config,
                'expected_ids':{d: sorted(r['video_id'] for r in rows) for d,rows in inventories.items()},
                'inventory_sha256':{d:sha256(f'data/manifests/{d}.csv') for d in inventories},
                'expected_branches':BRANCHES,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'decoder':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    for d, rows in inventories.items():
        if meta['inventory_sha256'][d] != sha256(f'data/manifests/{d}.csv'):
            raise ValueError(f'{d}: source inventory changed')
    rows = inventories[args.dataset]
    if args.sample_ids:
        ids = args.sample_ids.read_text().splitlines()
        if not ids or len(ids) != len(set(ids)) or set(ids)-{r['video_id'] for r in rows}:
            raise ValueError('unknown/duplicate sample IDs')
        rows = [r for r in rows if r['video_id'] in ids]
    if not torch.cuda.is_available():
        raise RuntimeError('offline cache requires CUDA fp16')
    torch.set_num_threads(4)
    model = load_backbone(args.backbone,'cuda',config)
    branches = DINO_BRANCHES if args.backbone=='dino' else ['resnet50']
    start = time.perf_counter()
    failures, completed, skipped = [], [], []
    for row in rows:
        key = row['video_id']
        if reusable(root,args.dataset,key,branches,row):
            skipped.append(key)
            continue
        try:
            with source_video(row,config['datasets'][args.dataset]) as path:
                values = extract_video(path,config,model,args.backbone,'cuda')
            write_video_features(root,args.dataset,key,values,row,meta)
            completed.append(key)
            print(f'{args.dataset}/{args.backbone}: {len(completed)+len(skipped)}/{len(rows)} {key}',flush=True)
        except Exception as exc:
            failures.append({'video_id':key,'reason':str(exc)})
            print(f'FAILED {key}: {exc}',flush=True)
    torch.cuda.synchronize()
    evidence_path = Path('results/phase1/extraction.json')
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {'invocations':[]}
    evidence['invocations'].append({'dataset':args.dataset,'backbone':args.backbone,'subset':bool(args.sample_ids),
        'expected_ids':[r['video_id'] for r in rows],'completed_ids':completed,'skipped_ids':skipped,
        'failures':failures,'wall_seconds':time.perf_counter()-start,'status':'failed' if failures else 'complete'})
    if (root/'meta.json').exists():
        evidence['coverage'] = coverage(root)
        if all(c['complete'] for c in evidence['coverage'].values()):
            atomic_write('data/manifests/phase1_cache.csv',(root/'manifest.csv').read_bytes())
            atomic_write('data/manifests/phase1_cache.meta.json',(root/'meta.json').read_bytes())
    atomic_json(evidence_path,evidence)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
