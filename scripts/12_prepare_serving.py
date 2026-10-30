"""Verify historical bindings and assemble the selected portable serving bundle."""
import argparse
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import yaml

from src.cache import atomic_json, atomic_write
from src.features import sha256
from src.serving_bundle import (MODEL_SHA256, WEIGHTS_ENUM, WEIGHTS_SHA256,
    read_model, read_settings, verify_bundle, verify_file, verify_snapshot)


def prepare_bundle(config_path: Path, output: Path) -> dict:
    config = read_settings(config_path)
    snapshot = json.loads(Path(config['research_snapshot']).read_text())
    verify_snapshot(snapshot)
    gate = json.loads(Path(config['phase3_gate']).read_text())
    deployment = json.loads(Path(config['deployment']).read_text())
    if (gate['status'] != 'passed with qualification' or gate['control_status'] != 'failed'
            or deployment['candidate_id'] != config['winner'] or deployment['seed'] is not None
            or deployment['checkpoint_path'] != config['model']
            or deployment['checkpoint_sha256'] != MODEL_SHA256):
        raise ValueError('unsupported deployment or upstream qualification')
    verify_file(config['model'], MODEL_SHA256)
    read_model(config['model'])
    feature_config = yaml.safe_load(Path(config['feature_config']).read_text())
    freeze = json.loads(Path(config['preprocessing_freeze']).read_text())
    if (sha256(config['feature_config']) != freeze['feature_config_sha256'] or
            feature_config['backbones']['resnet50']['weights'] != WEIGHTS_ENUM or
            feature_config['backbones']['resnet50']['weights_sha256'] != WEIGHTS_SHA256):
        raise ValueError('frozen configuration mismatch')
    weights = Path(config['weights'])
    if not weights.exists():
        from torchvision.models import ResNet50_Weights
        weights.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=weights.parent) as directory:
            staged = Path(directory) / weights.name
            urllib.request.urlretrieve(ResNet50_Weights.IMAGENET1K_V2.url, staged)
            verify_file(staged, WEIGHTS_SHA256)
            staged.replace(weights)
    verify_file(weights, WEIGHTS_SHA256)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the manifest last so an interrupted preparation cannot advertise partial files.
    (output / 'manifest.json').unlink(missing_ok=True)
    output.mkdir(exist_ok=True)
    files = dict(model='model.npz', weights='weights/torch/checkpoints/resnet50-11ad3fa6.pth',
                 config='runtime.yaml', freeze='preprocessing.json')
    for source, name in ((config['model'], 'model'), (weights, 'weights'),
                         (config['preprocessing_freeze'], 'freeze')):
        destination = output / files[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    runtime_config = {key: feature_config[key] for key in ('decoder', 'preprocessing', 'cache_version')}
    runtime_config['backbones'] = {'resnet50': dict(weights=WEIGHTS_ENUM, weights_sha256=WEIGHTS_SHA256)}
    atomic_write(output / files['config'], yaml.safe_dump(runtime_config).encode())
    manifest = dict(winner=config['winner'], model_sha256=MODEL_SHA256, weights_sha256=WEIGHTS_SHA256,
        weights_enum=WEIGHTS_ENUM, cpu_threads=4, limits=config['limits'], files=files,
        feature_config_sha256=sha256(config['feature_config']),
        runtime_config_sha256=sha256(output / files['config']),
        preprocessing_sha256=sha256(config['preprocessing_freeze']),
        shared_source_sha256={name: snapshot['source_files'][name] for name in
            ('src/preprocess.py', 'src/features.py', 'src/baselines.py', 'src/metrics.py', 'src/cache.py')},
        research_source_sha256=snapshot['source_code_sha256'],
        research_snapshot_sha256=sha256(config['research_snapshot']),
        phase3_gate_sha256=sha256(config['phase3_gate']), phase3_status=gate['status'],
        control_status=gate['control_status'], decoder_version='n9.0.2-37-gae19868753-20261029',
        packaging_amendment_sha256=sha256('results/phase4a/packaging_amendment.json'))
    atomic_json(output / 'manifest.json', manifest)
    verify_bundle(output / 'manifest.json')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/serving.yaml'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/serving'))
    args = parser.parse_args(argv)
    print(json.dumps(prepare_bundle(args.config, args.output), allow_nan=False))


if __name__ == '__main__':
    main()
