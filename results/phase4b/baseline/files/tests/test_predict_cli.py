"""CLI and shared inference behavior, using a cheap backbone with known outputs."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def prediction(tmp_path, monkeypatch):
    p = importlib.import_module('src.predict')
    artifact = dict(feature_mean=np.zeros(2048), feature_scale=np.ones(2048),
                    coefficient=np.full(2048, 1 / 2048), target_mean=np.array(3.),
                    target_std=np.array(2.), alpha=np.array(1.))
    class Backbone(torch.nn.Module):
        def forward(self, x):
            assert x.device.type == 'cpu' and x.dtype == torch.float32
            assert x.shape == (1, 3, 448, 448)
            return torch.full((1, 2048), 2.25)
    from src.serving_bundle import LIMITS
    from src.serving_bundle import MODEL_SHA256, WEIGHTS_SHA256, WEIGHTS_ENUM, PREPROCESSING_SHA256
    verified = dict(model_sha256=MODEL_SHA256, weights_sha256=WEIGHTS_SHA256,
                    weights_enum=WEIGHTS_ENUM, preprocessing_sha256=PREPROCESSING_SHA256, cpu_threads=4)
    runtime = p.InferenceRuntime(Backbone(), artifact, {}, verified, .05, LIMITS)
    video = tmp_path / 'clip.mp4'
    video.write_bytes(b'\x00\x00\x00\x18ftypisom' + b'\x00' * 20)
    meta = dict(duration_s=1., fps=30., width=32, height=48, rotation_applied=90)
    def probe(path, *, timestamps):
        assert not timestamps
        return dict(meta)
    monkeypatch.setattr(p, 'probe_video', probe)
    def samples(path, config):
        assert config is runtime.config
        for _ in range(2):
            yield 0., np.zeros((48, 32, 3), dtype=np.uint8), None
    monkeypatch.setattr(p, 'iter_samples', samples)
    return p, runtime, video, meta


def test_predict_matches_shared_ridge_on_known_features(prediction):
    p, runtime, video, _ = prediction
    result = p.predict_video(video, runtime)
    assert result['score'] == 7.5  # 3 + 2 * mean(2.25), deliberately above five.


def test_cpu_fp32_four_threads_and_exact_shared_calls(tmp_path, monkeypatch):
    p = importlib.import_module('src.predict')
    prepare = importlib.import_module('scripts.12_prepare_serving')
    prepare.prepare_bundle(Path('configs/serving.yaml'), tmp_path)
    calls = []
    def load(name, device, config):
        calls.append((name, device))
        assert torch.get_num_threads() == 4
        return torch.nn.Linear(1, 1).eval().requires_grad_(False)
    monkeypatch.setattr(p, 'load_backbone', load)
    runtime = p.load_runtime(tmp_path / 'manifest.json')
    assert calls == [('resnet50', 'cpu')]
    assert next(runtime.model.parameters()).dtype == torch.float32
    assert runtime.verified['weights_sha256'] == '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
    assert p.spatial_tensor is importlib.import_module('src.preprocess').spatial_tensor
    assert p.extract_features is importlib.import_module('src.features').extract_features


def test_cli_json_contains_finite_score_timings_and_verified_hashes(prediction, monkeypatch, capsys):
    p, runtime, video, _ = prediction
    monkeypatch.setattr(p, 'load_runtime', lambda bundle: runtime)
    assert p.main([str(video), '--json']) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result['score'] == 7.5 and output.err == ''
    assert set(result['timings_seconds']) == {'extraction', 'head', 'total'}
    assert all(np.isfinite(x) and x >= 0 for x in result['timings_seconds'].values())
    assert result['verified'] == runtime.verified
    assert result['startup_seconds'] == .05


def test_cli_failure_is_nonzero_without_score(prediction, monkeypatch, capsys):
    p, runtime, video, _ = prediction
    monkeypatch.setattr(p, 'load_runtime', lambda bundle: runtime)
    video.write_bytes(b'garbage')
    assert p.main([str(video)]) != 0
    output = capsys.readouterr()
    assert not output.out and output.err
    def corrupt(bundle):
        raise ValueError('checksum mismatch: model.npz')
    monkeypatch.setattr(p, 'load_runtime', corrupt)
    assert p.main([str(video)]) != 0
    assert not capsys.readouterr().out


@pytest.mark.parametrize('key,value', [('duration_s', 30.001), ('duration_s', float('nan')),
    ('width', 1921), ('width', 0), ('fps', 60.01), ('fps', float('inf')), ('height', 1920)])
def test_input_limits_and_rotation_vfr_use_shared_preprocess(prediction, key, value):
    p, runtime, video, meta = prediction
    meta[key] = value
    if key == 'height' and value == 1920: meta['width'] = 1920
    with pytest.raises(p.PredictionError) as caught:
        p.predict_video(video, runtime)
    assert caught.value.status_code == 422


def test_input_file_and_signature_limits(prediction):
    p, runtime, video, _ = prediction
    video.write_bytes(b'')
    with pytest.raises(p.PredictionError): p.predict_video(video, runtime)
    video.write_bytes(b'#EXTM3U\nhttps://example.org/video')
    with pytest.raises(p.PredictionError) as caught: p.predict_video(video, runtime)
    assert caught.value.status_code == 415
    with video.open('wb') as handle: handle.truncate(104857601)
    with pytest.raises(p.PredictionError) as caught: p.predict_video(video, runtime)
    assert caught.value.status_code == 413
    with pytest.raises(p.PredictionError): p.predict_video(Path('https://example.org/video.mp4'), runtime)


def test_cli_parity_export_is_raw_fp32(prediction, tmp_path):
    p, runtime, video, _ = prediction
    output = tmp_path / 'features.npy'
    result = p.predict_video(video, runtime, features_out=output)
    features = np.load(output)
    assert features.shape == (2, 2048) and features.dtype == np.float32
    assert np.all(features == 2.25) and result['score'] == 7.5


def test_generator_finalized_on_feature_failure(prediction, monkeypatch):
    p, runtime, video, _ = prediction
    closed = []
    def samples(path, config):
        try:
            yield 0., np.zeros((32, 32, 3), dtype=np.uint8), None
        finally:
            closed.append(True)
    monkeypatch.setattr(p, 'iter_samples', samples)
    def fail(*args): raise ValueError('bad feature')
    monkeypatch.setattr(p, 'extract_features', fail)
    with pytest.raises(p.PredictionError): p.predict_video(video, runtime)
    assert closed == [True]


def test_real_rotation_vfr_decode_uses_frozen_path(tmp_path):
    import subprocess
    p = importlib.import_module('src.predict')
    from src.serving_bundle import LIMITS
    video = tmp_path / 'vfr.mp4'
    rotated = tmp_path / 'portrait.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=64x32:rate=30:duration=0.6', '-vf',
        "settb=1/1000,setpts='if(eq(N,13),401,N*1000/30)'", '-fps_mode', 'passthrough',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-display_rotation', '90', '-i', str(video),
                    '-c', 'copy', str(rotated)], check=True)
    class Backbone(torch.nn.Module):
        def forward(self, x):
            return x.mean().expand(1, 2048)
    artifact = dict(feature_mean=np.zeros(2048), feature_scale=np.ones(2048),
                    coefficient=np.zeros(2048), target_mean=np.array(3.), target_std=np.array(1.))
    runtime = p.InferenceRuntime(Backbone(), artifact, {}, {}, 0., LIMITS)
    assert p.validate_media(rotated, LIMITS)['rotation_applied'] == 90
    assert p.predict_video(rotated, runtime)['score'] == 3.


@pytest.mark.parametrize('alias', ['same_path', 'symlink', 'hardlink'])
def test_feature_export_rejects_input_alias_and_preserves_video(prediction, tmp_path, alias):
    p, runtime, video, _ = prediction
    original = video.read_bytes()
    output = video
    if alias != 'same_path':
        output = tmp_path / 'alias.npy'
        if alias == 'symlink': output.symlink_to(video)
        else: output.hardlink_to(video)
    with pytest.raises(p.PredictionError, match='different file'):
        p.predict_video(video, runtime, features_out=output)
    assert video.read_bytes() == original


def test_terminal_cli_prints_readable_unclipped_score(prediction, monkeypatch, capsys):
    p, runtime, video, _ = prediction
    monkeypatch.setattr(p, 'load_runtime', lambda bundle: runtime)
    monkeypatch.setattr(p.sys.stdout, 'isatty', lambda: True)
    assert p.main([str(video)]) == 0
    output = capsys.readouterr()
    assert 'Predicted quality: 7.50 (KoNViD 1–5 scale)' in output.out
    assert 'Higher is better' in output.out and 'unclipped' in output.out
    assert 'Analysis time:' in output.out and 'Model startup:' in output.out
    assert 'Analysis time:' in output.out and 'for a 1.00 s clip' in output.out
    assert 'Feature extraction (incl. backbone):' in output.out and 'Ridge head:' in output.out
    assert 'CPU threads: 4' in output.out
    assert 'Model 4c9782ae… · Weights 11ad3fa6… (verified)' in output.out
    assert runtime.verified['preprocessing_sha256'] in output.out
    assert 'timings_seconds' not in output.out and output.err == ''


def test_json_flag_preserves_machine_contract_in_terminal(prediction, monkeypatch, capsys):
    p, runtime, video, _ = prediction
    monkeypatch.setattr(p, 'load_runtime', lambda bundle: runtime)
    monkeypatch.setattr(p.sys.stdout, 'isatty', lambda: True)
    assert p.main([str(video), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['score'] == 7.5


def test_cli_filters_only_known_readonly_frame_warning(prediction, monkeypatch, capsys):
    import warnings
    p, runtime, video, _ = prediction
    monkeypatch.setattr(p, 'load_runtime', lambda bundle: runtime)
    original = p.predict_video
    def predict(*args, **kwargs):
        warnings.warn_explicit('The given NumPy array is not writable, and PyTorch does not support non-writable tensors.',
                              UserWarning, 'src/preprocess.py', 130, module='src.preprocess')
        warnings.warn('Unexpected inference warning', UserWarning)
        return original(*args, **kwargs)
    monkeypatch.setattr(p, 'predict_video', predict)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        assert p.main([str(video), '--json']) == 0
    assert [str(w.message) for w in captured] == ['Unexpected inference warning']
    assert json.loads(capsys.readouterr().out)['score'] == 7.5


def test_default_output_is_human_even_when_redirected(prediction, monkeypatch, capsys):
    p, runtime, video, _ = prediction
    monkeypatch.setattr(p, 'load_runtime', lambda bundle: runtime)
    monkeypatch.setattr(p.sys.stdout, 'isatty', lambda: False)
    assert p.main([str(video)]) == 0
    assert 'Predicted quality: 7.50' in capsys.readouterr().out


def test_json_schema_and_score_precision_are_unchanged(prediction, monkeypatch, capsys):
    p, runtime, video, _ = prediction
    runtime.artifact['target_mean'] = np.array(0.12345678912345678)
    runtime.artifact['coefficient'][:] = 0.
    monkeypatch.setattr(p, 'load_runtime', lambda bundle: runtime)
    assert p.main([str(video), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {'score', 'timings_seconds', 'verified', 'startup_seconds'}
    assert set(result['timings_seconds']) == {'extraction', 'head', 'total'}
    assert result['verified'] == runtime.verified
    assert result['score'] == 0.12345678912345678
