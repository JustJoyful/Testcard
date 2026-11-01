"""Fail-closed portable bundle validation shared by preparation and inference."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml

from src.features import sha256

MODEL_SHA256 = '4c9782aefca4b81da4d15da31681f11ca8ded4e890d36d67fb478663583327a3'
WEIGHTS_SHA256 = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
WEIGHTS_ENUM = 'ResNet50_Weights.IMAGENET1K_V2'
FEATURE_CONFIG_SHA256 = '8397ea636c79e69a382e88a7238f798b398dafd91774edd6b50555f7bb62777c'
PREPROCESSING_SHA256 = 'acc2e85da8cce49f23d21d990af382111cd1a9352d2fdeeb4731d06e3fe1678b'
LIMITS = dict(file_bytes=104857600, body_bytes=105906176, receipt_seconds=60,
              duration_seconds=30, source_pixels=2073600, longest_side=1920, fps=60)
ROOT = Path(__file__).resolve().parents[1]


def verify_file(path, expected):
    path = Path(path)
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('mandatory SHA256 missing or invalid')
    if not path.is_file():
        raise ValueError(f'missing artifact: {path.name}')
    with path.open('rb') as handle:
        if handle.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'LFS pointer requires materialization: {path.name}')
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'checksum mismatch: {path.name}')
    return actual


def validate_settings(config):
    if config.get('winner') != 'resnet50/ridge':
        raise ValueError('unsupported serving winner')
    if config.get('model_sha256') != MODEL_SHA256:
        raise ValueError('selected model identity mismatch')
    if config.get('cpu_threads') != 4 or config.get('limits') != LIMITS:
        raise ValueError('unsupported numerical settings or limits')
    return config


def read_settings(path):
    return validate_settings(yaml.safe_load(Path(path).read_text()))


def read_model(path):
    required = {'feature_mean', 'feature_scale', 'coefficient', 'target_mean', 'target_std', 'alpha'}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError('incorrect ridge arrays')
            arrays = {name: archive[name] for name in required}
        for name, value in arrays.items():
            shape = (2048,) if name in ('feature_mean', 'feature_scale', 'coefficient') else ()
            if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError(f'invalid ridge array: {name}')
        if any(np.any(arrays[name] <= 0) for name in ('feature_scale', 'target_std', 'alpha')):
            raise ValueError('nonpositive ridge scale/std/alpha')
        return arrays
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f'invalid ridge artifact: {exc}') from exc


# Committed list of approved decision-log appends, in order; each entry names its record and the field
# that pins it to the previous record. Adding an append means adding a record and one list entry.
DECISION_APPENDS = 'results/decision_appends.json'


def verify_decision_chain(content, expected, root):
    """The log must be the historical prefix plus exactly the recorded appends, nothing else."""
    records, previous = [], None
    for entry in json.loads((Path(root) / DECISION_APPENDS).read_text()):
        relative, pin = entry['record'], entry['pin']
        path = Path(root) / relative
        if not path.is_file():
            break
        record = json.loads(path.read_text())
        if pin:
            verify_file(previous, record[pin])
        records.append(record)
        previous = path
    before = expected
    for index, record in enumerate(records):
        start = record['prefix_bytes']
        end = records[index + 1]['prefix_bytes'] if index + 1 < len(records) else len(content)
        if (not isinstance(start, int) or not isinstance(end, int) or not 0 < start < end <= len(content)
                or record['before_sha256'] != before
                or hashlib.sha256(content[:start]).hexdigest() != before
                or hashlib.sha256(content[start:end]).hexdigest() != record['appended_sha256']
                or hashlib.sha256(content[:end]).hexdigest() != record['after_sha256']):
            raise ValueError('unapproved decision-log change')
        before = record['after_sha256']
    if hashlib.sha256(content).hexdigest() != before:
        raise ValueError('unapproved decision-log change')


def verify_snapshot(snapshot, root=ROOT):
    for name, expected in {**snapshot['source_files'], **snapshot['bindings']}.items():
        try:
            path = Path(root) / name
            if name == 'DECISIONS.md' and (Path(root) / DECISION_APPENDS).is_file():
                verify_decision_chain(path.read_bytes(), expected, root)
            else:
                verify_file(path, expected)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f'historical identity changed: {name}') from exc
    for name, expected in snapshot['outputs_sha256'].items():
        verify_file(Path(root) / 'results/phase3' / name, expected)


def verify_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    validate_settings(manifest)
    if manifest.get('weights_enum') != WEIGHTS_ENUM or manifest.get('weights_sha256') != WEIGHTS_SHA256:
        raise ValueError('incorrect backbone identity')
    if (manifest.get('feature_config_sha256') != FEATURE_CONFIG_SHA256 or
            manifest.get('preprocessing_sha256') != PREPROCESSING_SHA256):
        raise ValueError('unrecognized frozen preprocessing')
    files = manifest['files']
    if set(files) != {'model', 'weights', 'config', 'freeze'}:
        raise ValueError('incomplete bundle')
    resolved = {}
    for name, value in files.items():
        relative = Path(value)
        target = (path.parent / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(path.parent):
            raise ValueError('nonportable bundle path')
        resolved[name] = target
    if files['weights'] != 'weights/torch/checkpoints/resnet50-11ad3fa6.pth':
        raise ValueError('incorrect backbone path')
    verified_model = verify_file(resolved['model'], MODEL_SHA256)
    verified_weights = verify_file(resolved['weights'], WEIGHTS_SHA256)
    verify_file(resolved['config'], manifest['runtime_config_sha256'])
    verified_preprocessing = verify_file(resolved['freeze'], manifest['preprocessing_sha256'])
    freeze = json.loads(resolved['freeze'].read_text())
    if (freeze['feature_config_sha256'] != manifest['feature_config_sha256'] or
            freeze['cache_version'] != 'v2' or freeze['status'] != 'frozen' or
            freeze['weights_sha256']['resnet50'] != WEIGHTS_SHA256):
        raise ValueError('preprocessing identity mismatch')
    if not manifest.get('shared_source_sha256'):
        raise ValueError('missing shared source identities')
    required_sources = {'src/preprocess.py', 'src/features.py', 'src/baselines.py', 'src/metrics.py', 'src/cache.py'}
    if set(manifest['shared_source_sha256']) != required_sources:
        raise ValueError('incomplete shared source identities')
    for name, expected in manifest['shared_source_sha256'].items():
        verify_file(ROOT / name, expected)
        if name in freeze['source_sha256'] and expected != freeze['source_sha256'][name]:
            raise ValueError('frozen source identity mismatch')
    config = yaml.safe_load(resolved['config'].read_text())
    if config['backbones'] != {'resnet50': {'weights': WEIGHTS_ENUM, 'weights_sha256': WEIGHTS_SHA256}}:
        raise ValueError('incorrect runtime backbone settings')
    config['weights_dir'] = str(path.parent / 'weights')
    return dict(manifest=manifest, config=config, artifact=read_model(resolved['model']),
                weights=resolved['weights'], verified=dict(model_sha256=verified_model,
                weights_sha256=verified_weights, weights_enum=WEIGHTS_ENUM,
                preprocessing_sha256=verified_preprocessing, cpu_threads=4))
