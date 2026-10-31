"""Serving bundle checks cover bytes and arrays before any model load."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def bundle_module():
    return importlib.import_module('src.serving_bundle')


def test_bundle_pins_deployment_and_v2_weights():
    module = bundle_module()
    config = module.read_settings(Path('configs/serving.yaml'))
    assert config['winner'] == 'resnet50/ridge'
    assert config['model_sha256'] == '4c9782aefca4b81da4d15da31681f11ca8ded4e890d36d67fb478663583327a3'
    assert module.WEIGHTS_ENUM == 'ResNet50_Weights.IMAGENET1K_V2'
    assert module.WEIGHTS_SHA256 == '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'


def test_pointer_missing_hash_and_tamper_fail_before_load(tmp_path):
    module = bundle_module()
    path = tmp_path / 'model.npz'
    path.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='pointer'):
        module.verify_file(path, 'a' * 64)
    path.write_bytes(b'payload')
    for expected in (None, '', 'a' * 64):
        with pytest.raises(ValueError):
            module.verify_file(path, expected)
    config = module.read_settings(Path('configs/serving.yaml'))
    config['winner'] = 'dino/ridge'
    with pytest.raises(ValueError, match='winner'):
        module.validate_settings(config)


def model_arrays():
    return dict(feature_mean=np.zeros(2048), feature_scale=np.ones(2048),
                coefficient=np.zeros(2048), target_mean=np.array(3.),
                target_std=np.array(1.), alpha=np.array(1.))


@pytest.mark.parametrize('change', ['dimension', 'nan', 'scale', 'std', 'extra', 'missing'])
def test_bundle_validates_arrays(tmp_path, change):
    module = bundle_module()
    arrays = model_arrays()
    if change == 'dimension': arrays['coefficient'] = np.zeros(2047)
    if change == 'nan': arrays['feature_mean'][0] = np.nan
    if change == 'scale': arrays['feature_scale'][0] = 0.
    if change == 'std': arrays['target_std'] = np.array(-1.)
    if change == 'extra': arrays['mos'] = np.array(2.)
    if change == 'missing': del arrays['alpha']
    path = tmp_path / 'model.npz'
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='ridge'):
        module.read_model(path)


def test_runtime_bundle_needs_no_cache_labels_or_dino(tmp_path):
    module = bundle_module()
    prepare = importlib.import_module('scripts.12_prepare_serving')
    output = tmp_path / 'bundle'
    manifest = prepare.prepare_bundle(Path('configs/serving.yaml'), output)
    assert manifest['winner'] == 'resnet50/ridge'
    assert not any(Path(value).is_absolute() for value in manifest['files'].values())
    assert not any(word in json.dumps(manifest) for word in ('mos', 'dino', str(Path.cwd())))
    verified = module.verify_bundle(output / 'manifest.json')
    assert verified['artifact']['coefficient'].shape == (2048,)
    assert verified['verified']['model_sha256'] == module.MODEL_SHA256
    (output / 'model.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError):
        module.verify_bundle(output / 'manifest.json')


def test_historical_snapshot_checks_original_roster(tmp_path):
    module = bundle_module()
    from src.features import sha256
    (tmp_path / 'src').mkdir()
    original = tmp_path / 'src/original.py'
    original.write_text('pass\n')
    snapshot = {'source_files': {'src/original.py': sha256(original)}, 'bindings': {}, 'outputs_sha256': {}}
    module.verify_snapshot(snapshot, tmp_path)
    (tmp_path / 'src/new.py').write_text('pass\n')
    module.verify_snapshot(snapshot, tmp_path)
    original.write_text('changed\n')
    with pytest.raises(ValueError, match='historical'):
        module.verify_snapshot(snapshot, tmp_path)
    original.unlink()
    with pytest.raises(ValueError, match='historical'):
        module.verify_snapshot(snapshot, tmp_path)


def test_manifest_cannot_rebind_preprocessing_or_sources(tmp_path):
    module = bundle_module()
    prepare = importlib.import_module('scripts.12_prepare_serving')
    output = tmp_path / 'bundle'
    manifest = prepare.prepare_bundle(Path('configs/serving.yaml'), output)
    original = json.loads(json.dumps(manifest))
    manifest['preprocessing_sha256'] = 'a' * 64
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.verify_bundle(output / 'manifest.json')
    manifest = original
    manifest['feature_config_sha256'] = 'a' * 64
    freeze = json.loads((output / 'preprocessing.json').read_text())
    freeze['feature_config_sha256'] = 'a' * 64
    (output / 'preprocessing.json').write_text(json.dumps(freeze))
    from src.features import sha256
    manifest['preprocessing_sha256'] = sha256(output / 'preprocessing.json')
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.verify_bundle(output / 'manifest.json')


@pytest.mark.parametrize('change', ['weights_bytes', 'missing_weights_hash', 'missing_model_hash', 'unsupported_winner'])
def test_portable_bundle_rejects_corrupt_or_unbound_artifacts(tmp_path, change):
    module = bundle_module()
    prepare = importlib.import_module('scripts.12_prepare_serving')
    manifest = prepare.prepare_bundle(Path('configs/serving.yaml'), tmp_path)
    if change == 'weights_bytes':
        with (tmp_path / manifest['files']['weights']).open('r+b') as file:
            file.write(b'corrupt')
    elif change == 'missing_weights_hash':
        del manifest['weights_sha256']
    elif change == 'missing_model_hash':
        del manifest['model_sha256']
    else:
        manifest['winner'] = 'resnet50/transformer'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.verify_bundle(tmp_path / 'manifest.json')


def test_authorized_decision_append_preserves_historical_prefix(tmp_path):
    from src.features import sha256
    module = bundle_module()
    decision = tmp_path / 'DECISIONS.md'
    decision.write_bytes(b'historical decision\n')
    before = sha256(decision)
    snapshot = {'source_files': {}, 'bindings': {'DECISIONS.md': before}, 'outputs_sha256': {}}
    suffix = b'\nApproved dated decoder packaging amendment.\n'
    decision.write_bytes(decision.read_bytes() + suffix)
    import hashlib
    amendment = dict(before_sha256=before, prefix_bytes=len(b'historical decision\n'),
        after_sha256=sha256(decision), appended_sha256=hashlib.sha256(suffix).hexdigest())
    record = tmp_path / 'results/phase4a/packaging_amendment.json'
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(amendment))
    module.verify_snapshot(snapshot, tmp_path)
    decision.write_bytes(b'changed historical decision\n' + suffix)
    with pytest.raises(ValueError, match='historical'):
        module.verify_snapshot(snapshot, tmp_path)
    decision.write_bytes(b'historical decision\n' + suffix + b'unapproved append')
    with pytest.raises(ValueError, match='historical'):
        module.verify_snapshot(snapshot, tmp_path)


def test_docker_context_excludes_research_payloads(tmp_path):
    import subprocess
    assert Path('.dockerignore').is_file(), 'missing allowlisted Docker context'
    (tmp_path / '.dockerignore').write_bytes(Path('.dockerignore').read_bytes())
    names = ['src/predict.py', 'serving/app.py', 'env/serving.lock', 'artifacts/serving/model.npz',
             'artifacts/serving/weights/torch/checkpoints/resnet50-11ad3fa6.pth',
             'data/raw/video.mp4', 'videos.zip', 'cache/a.npy', 'weights/dino.pth',
             'results/phase3/refit/dino/best.pt', '.venv/bin/python', '.venv-serving/bin/python']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('sentinel')
    (tmp_path / 'Dockerfile').write_text('FROM scratch\nCOPY . /\n')
    import io, tarfile
    image_file = tmp_path / 'image-id'
    result = subprocess.run(['docker', 'build', '-q', '--iidfile', str(image_file), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    container = subprocess.check_output(['docker', 'create', '--entrypoint', '/unused', image_file.read_text().strip()], text=True).strip()
    try:
        exported = subprocess.check_output(['docker', 'export', container])
        with tarfile.open(fileobj=io.BytesIO(exported)) as archive:
            included = set(archive.getnames())
        assert all(name in included for name in names[:5])
        assert all(name not in included for name in names[5:])
    finally:
        subprocess.run(['docker', 'rm', container], check=True, capture_output=True)



def test_bundle_portable_from_other_working_tree(tmp_path, monkeypatch):
    import shutil
    module = bundle_module()
    prepare = importlib.import_module('scripts.12_prepare_serving')
    source = tmp_path / 'source'
    prepare.prepare_bundle(Path('configs/serving.yaml'), source)
    moved = tmp_path / 'relocated'
    source.rename(moved)
    monkeypatch.chdir(tmp_path)
    assert module.verify_bundle(moved / 'manifest.json')['artifact']['coefficient'].shape == (2048,)


def test_decoder_checksum_mismatch_fails_before_extraction(tmp_path):
    import subprocess
    source = Path('Dockerfile').read_text().split('FROM python:', 2)
    assert len(source) == 3, 'missing decoder builder stage'
    stage = 'FROM python:' + source[1]
    stage = stage.replace('83537b0f6a97905f325080f6f5916e70ec252d1ef3e4ce4746f755b88720cbf5', '0' * 64)
    (tmp_path / 'Dockerfile').write_text(stage)
    result = subprocess.run(['docker', 'build', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'FAILED' in result.stdout + result.stderr and 'sha256sum' in result.stdout + result.stderr


def test_docker_builder_rejects_lfs_pointer_model(tmp_path):
    import shutil, subprocess
    for name in ['Dockerfile', '.dockerignore']:
        shutil.copyfile(name, tmp_path / name)
    for name in ['src', 'serving', 'env', 'artifacts/serving']:
        shutil.copytree(name, tmp_path / name, ignore=shutil.ignore_patterns('__pycache__'))
    model = tmp_path / 'artifacts/serving/model.npz'
    model.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
    result = subprocess.run(['docker', 'build', '--target', 'builder', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'LFS pointer' in result.stdout + result.stderr


def test_image_default_user_can_verify_bundle_offline():
    import subprocess
    code = "from pathlib import Path; import os; from src.serving_bundle import verify_bundle; assert os.getuid() != 0; verify_bundle(Path('artifacts/serving/manifest.json'))"
    result = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,size=256m', '--entrypoint', 'python', 'vqa:phase4a', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
