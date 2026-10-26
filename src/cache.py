"""Single-writer, checksum-verified spatial feature cache."""
import csv
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml

from src.features import sha256

DINO_BRANCHES = ['dino_cls'] + [f'dino_patch{stat}_L{layer}' for layer in (6,9,12) for stat in ('mean','std')]
BRANCHES = DINO_BRANCHES + ['resnet50']
FIELDS = ['video_id','dataset','feature_set','T','mos','mos_scale','source_size_bytes','source_sha256',
          'duration_s','fps','width','height','rotation_applied','feature_sha256','extraction_env','status','failure_reason']


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def atomic_array(path, array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    atomic_write(path, buffer.getvalue())


def csv_bytes(rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def read_manifest(root):
    path = Path(root)/'manifest.csv'
    return list(csv.DictReader(path.open())) if path.exists() else []


def safe_id(dataset, video_id):
    if dataset not in ('konvid','livevqc') or not re.fullmatch(r'[A-Za-z0-9_-]+', video_id):
        raise ValueError(f'unsafe dataset/video ID: {dataset}/{video_id}')


def read_branch(root, dataset, video_id, branch, rows=None, source=None):
    safe_id(dataset, video_id)
    if branch not in BRANCHES:
        raise ValueError(f'{video_id}: unknown branch {branch}')
    records = [r for r in (read_manifest(root) if rows is None else rows)
               if (r['dataset'],r['video_id'],r['feature_set']) == (dataset,video_id,branch)]
    path = Path(root)/dataset/branch/(video_id+'.npy')
    try:
        if len(records) != 1 or records[0]['status'] != 'ok':
            raise ValueError('missing or failed manifest entry')
        record = records[0]
        if source and any(str(record[k]) != str(source[k]) for k in ('source_sha256','source_size_bytes')):
            raise ValueError('source checksum/size mismatch')
        if sha256(path) != record['feature_sha256']:
            raise ValueError('array checksum mismatch')
        array = np.load(path, allow_pickle=False)
        dimension = 2048 if branch == 'resnet50' else 768
        if (array.dtype != np.float16 or array.ndim != 2 or array.shape != (int(record['T']),dimension)
                or not len(array) or not np.isfinite(array).all()):
            raise ValueError('invalid shape/dtype/values')
        return array
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f'{dataset}/{video_id}/{branch}: {exc}') from exc


def reusable(root, dataset, video_id, branches, source):
    rows = read_manifest(root)
    try:
        arrays = [read_branch(root,dataset,video_id,b,rows,source) for b in branches]
        return len({len(x) for x in arrays}) == 1
    except ValueError:
        return False


def write_video_features(root, dataset, video_id, arrays, source_row, meta):
    root = Path(root)
    safe_id(dataset, video_id)
    keys = set(arrays)
    if keys not in (set(DINO_BRANCHES), {'resnet50'}, set(BRANCHES)):
        raise ValueError(f'{video_id}: incomplete branch group')
    if len({len(x) for x in arrays.values()}) != 1:
        raise ValueError(f'{video_id}: inconsistent T')
    for branch, value in arrays.items():
        dimension = 2048 if branch == 'resnet50' else 768
        if value.ndim != 2 or value.shape[1] != dimension or not len(value) or value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError(f'{video_id}: invalid {branch}')
    metadata = root/'meta.json'
    if metadata.exists():
        existing = json.loads(metadata.read_text())
        if existing != meta:
            raise ValueError('incompatible cache metadata')
    else:
        atomic_json(metadata, meta)
    rows = read_manifest(root)
    # Remove the whole group first: interrupted repairs never leave reusable stale rows.
    kept = [r for r in rows if not (r['dataset']==dataset and r['video_id']==video_id and r['feature_set'] in keys)]
    if len(kept) != len(rows):
        atomic_write(root/'manifest.csv', csv_bytes(kept,FIELDS))
    additions = []
    for branch, value in arrays.items():
        path = root/dataset/branch/(video_id+'.npy')
        atomic_array(path,value)
        record = {key: source_row.get(key,'') for key in FIELDS}
        record.update(dataset=dataset,video_id=video_id,feature_set=branch,T=len(value),
                      feature_sha256=sha256(path),extraction_env='local',status='ok',failure_reason='')
        additions.append(record)
    atomic_write(root/'manifest.csv', csv_bytes(kept+additions,FIELDS))
    return additions


def coverage(root):
    meta = json.loads((Path(root)/'meta.json').read_text())
    rows = read_manifest(root)
    result = {}
    for dataset, ids in meta['expected_ids'].items():
        failed = []
        for key in ids:
            try:
                arrays = [read_branch(root,dataset,key,b,rows) for b in meta['expected_branches']]
                if len({len(a) for a in arrays}) != 1:
                    raise ValueError('inconsistent T')
            except ValueError:
                failed.append(key)
        result[dataset] = {'complete': not failed, 'expected': len(ids), 'failed_ids': failed}
    return result


def validate_freeze(feature_config, freeze_record=None):
    config = yaml.safe_load(Path(feature_config).read_text())
    freeze_record = freeze_record or config.get('freeze_record', 'results/phase0/preprocessing_freeze.json')
    frozen = json.loads(Path(freeze_record).read_text())
    if frozen.get('status') != 'frozen' or frozen['cache_version'] != config['cache_version']:
        raise ValueError('feature config/freeze version or status mismatch')
    checks = {str(feature_config): frozen['feature_config_sha256'], 'env/requirements.lock': frozen['lock_sha256'],
              'configs/scope.yaml': frozen['scope_config_sha256'],
              'results/phase0/readiness_at_freeze.json': frozen['readiness_snapshot_sha256'], **frozen['source_sha256']}
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError(f'frozen identity mismatch: {path}')
    if json.loads(Path('results/phase0/readiness_at_freeze.json').read_text())['status'] != 'go':
        raise ValueError('Phase 0 readiness is not go')
    config = yaml.safe_load(Path(feature_config).read_text())
    for name, filename in [('dino','dinov2_vitb14_pretrain.pth'),('resnet50','resnet50-11ad3fa6.pth')]:
        path = Path(config.get('weights_dir','weights'))/'torch/checkpoints'/filename
        if sha256(path) != frozen['weights_sha256'][name]:
            raise ValueError(f'{name}: weight identity mismatch')
    return frozen


def validate_cache(root, feature_config, freeze_record=None):
    frozen = validate_freeze(feature_config,freeze_record)
    root = Path(root)
    meta = json.loads((root/'meta.json').read_text())
    if meta['feature_config_sha256'] != frozen['feature_config_sha256'] or meta['lock_sha256'] != frozen['lock_sha256']:
        raise ValueError('incompatible cache policy/lock')
    for dataset, ids in meta['expected_ids'].items():
        inventory = list(csv.DictReader(Path(f'data/manifests/{dataset}.csv').open()))
        if sorted(ids) != sorted(r['video_id'] for r in inventory) or sha256(f'data/manifests/{dataset}.csv') != meta['inventory_sha256'][dataset]:
            raise ValueError(f'{dataset}: changed source inventory')
    checks = coverage(root)
    if not all(v['complete'] for v in checks.values()):
        raise ValueError(f'incomplete cache: {checks}')
    return {**meta,'coverage': checks,'manifest_sha256': sha256(root/'manifest.csv')}


def load_cached_video(root, dataset, video_id, feature_set):
    if feature_set not in ('dino','resnet50'):
        raise ValueError(f'unknown feature set: {feature_set}')
    rows = read_manifest(root)
    branches = DINO_BRANCHES if feature_set == 'dino' else ['resnet50']
    arrays = {b:read_branch(root,dataset,video_id,b,rows) for b in branches}
    if len({len(a) for a in arrays.values()}) != 1:
        raise ValueError(f'{video_id}: inconsistent timesteps')
    result = np.concatenate([arrays['dino_cls'],arrays['dino_patchmean_L12']],axis=1) if feature_set=='dino' else arrays['resnet50'].copy()
    result.setflags(write=False)
    return result
