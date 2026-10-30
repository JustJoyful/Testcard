"""Evidence gates must reject an individual failure, stale identity or incomplete roster."""
import copy
import importlib
from pathlib import Path

import numpy as np
import pytest


def tool():
    return importlib.import_module('scripts.14_serving_check')


def response(score=0.):
    from src.serving_bundle import MODEL_SHA256, WEIGHTS_SHA256, WEIGHTS_ENUM, PREPROCESSING_SHA256
    return dict(score=score, startup_seconds=.1, timings_seconds=dict(extraction=.2, head=.01, total=.22),
                verified=dict(model_sha256=MODEL_SHA256, weights_sha256=WEIGHTS_SHA256,
                weights_enum=WEIGHTS_ENUM, preprocessing_sha256=PREPROCESSING_SHA256, cpu_threads=4))


def test_reference_uses_full_refit_ridge_without_mos(tmp_path, monkeypatch):
    m = tool()
    from src.serving_bundle import read_settings
    config = read_settings(Path('configs/serving.yaml'))
    monkeypatch.setattr(m, 'OUT', tmp_path)
    original = m.load_sources
    class NoMOS(dict):
        def __getitem__(self, key):
            assert key != 'mos', 'serving reference accessed MOS'
            return super().__getitem__(key)
    monkeypatch.setattr(m, 'load_sources', lambda path: {k: NoMOS(v) for k, v in original(path).items()})
    record = m.make_reference(config)
    assert len(record['rows']) == 20
    assert record['model_sha256'] == '4c9782aefca4b81da4d15da31681f11ca8ded4e890d36d67fb478663583327a3'
    assert all(np.isfinite(r['score']) for r in record['rows'])
    assert 'mos' not in (tmp_path / 'reference_predictions.csv').read_text().splitlines()[0]
    config['model'] = 'results/phase2/resnet50/ridge/fold0/model.npz'
    with pytest.raises(ValueError): m.make_reference(config)


def test_sample_rejects_duplicates_missing_or_reordered_ids():
    m = tool()
    for ids in [['a', 'a'], ['a'], ['b', 'a']]:
        with pytest.raises(ValueError): m.validate_ids(ids, ['a', 'b'])


def test_parity_rejects_one_bad_timestep_or_score():
    m = tool()
    assert m.passes_parity(.999, .01, 1e-6)
    assert not m.passes_parity(np.nextafter(.999, 0), .01, 1e-6)
    assert not m.passes_parity(.999, np.nextafter(.01, np.inf), 1e-6)
    assert not m.passes_parity(.999, .01, np.nextafter(1e-6, np.inf))
    reference = np.ones((3, 2048), dtype=np.float16)
    raw = reference.astype(np.float32)
    raw[1] *= -1
    row = m.compare_features(reference, raw, 0., response(), response())
    assert row['minimum_cosine'] == pytest.approx(-1.) and row['status'] == 'failed'
    with pytest.raises(ValueError): m.compare_features(reference, raw[:2], 0., response(), response())


def test_real_front_doors_required():
    m = tool()
    assert m.validate_front_door({'exit_code': 0, 'response': response()}, 'cli')['score'] == 0.
    assert m.validate_front_door({'http_status': 200, 'response': response()}, 'api')['score'] == 0.
    for door, envelope in [('cli', {'response': response()}), ('api', {'http_status': 503, 'response': response()})]:
        with pytest.raises(ValueError): m.validate_front_door(envelope, door)
    broken = response()
    del broken['verified']['model_sha256']
    with pytest.raises(ValueError): m.validate_front_door({'exit_code': 0, 'response': broken}, 'cli')
    broken = response(); broken['score'] = float('nan')
    with pytest.raises(ValueError): m.validate_front_door({'http_status': 200, 'response': broken}, 'api')


def latency_rows():
    return [dict(video_id=key, repeat=repeat, role='warm', duration_seconds=5.,
        request_started=100., request_finished=100. + repeat,
        request_roundtrip_seconds=float(repeat), extraction=.2, head=.01, internal_total=.22,
        http_status=200, response=response()) for key in ['a', 'b'] for repeat in [1, 2, 3]]


def test_measurements_separate_cold_warm_and_request_total():
    m = tool()
    rows = latency_rows()
    summary = m.summarize_latency(rows, ['a', 'b'])
    assert summary['maximum_seconds_per_ten_seconds'] == 6.
    assert summary['median_seconds_per_ten_seconds'] == 4.
    assert summary['budget_passed']
    for changed in [rows[:-1], rows + rows[:1], [{**r, 'role': 'cold'} for r in rows]]:
        with pytest.raises(ValueError): m.summarize_latency(changed, ['a', 'b'])
    changed = copy.deepcopy(rows); changed[0]['request_roundtrip_seconds'] = .22
    with pytest.raises(ValueError): m.summarize_latency(changed, ['a', 'b'])
    changed = copy.deepcopy(rows); changed[0].update(request_finished=161., request_roundtrip_seconds=61.)
    assert not m.summarize_latency(changed, ['a', 'b'])['budget_passed']


def test_historical_evidence_never_rewritten(tmp_path, monkeypatch):
    m = tool()
    from src.serving_bundle import read_settings
    from src.features import sha256
    config = read_settings(Path('configs/serving.yaml'))
    paths = ['results/phase3/phase_gate.json', 'results/phase3/deployment.json',
             'results/phase4a/research_snapshot.json', 'DECISIONS.md', 'src/preprocess.py',
             'src/features.py', 'src/baselines.py', 'env/requirements.lock']
    before = {p: sha256(p) for p in paths}
    monkeypatch.setattr(m, 'OUT', tmp_path)
    m.make_reference(config)
    assert {p: sha256(p) for p in paths} == before


def test_gate_requires_complete_evidence_and_budget(tmp_path, monkeypatch):
    m = tool()
    from src.serving_bundle import read_settings
    monkeypatch.setattr(m, 'OUT', tmp_path)
    with pytest.raises(ValueError): m.collect_evidence(read_settings(Path('configs/serving.yaml')))


def test_report_publishes_gate_last(tmp_path, monkeypatch):
    m = tool()
    monkeypatch.setattr(m, 'OUT', tmp_path)
    from src.cache import atomic_json
    atomic_json(tmp_path / 'phase_gate.json', {'status': 'passed'})
    monkeypatch.setattr(m, 'collect_evidence', lambda config: {'status': 'passed'})
    def fail(*args): raise OSError('injected report write failure')
    monkeypatch.setattr(m, 'report_text', fail)
    with pytest.raises(OSError): m.write_report({})
    import json
    assert json.loads((tmp_path / 'phase_gate.json').read_text())['status'] == 'failed'


def test_notes_and_operational_limits_required(tmp_path, monkeypatch):
    m = tool()
    (tmp_path / 'NOTES.md').write_text('no serving evidence yet')
    with pytest.raises(ValueError): m.documentation_bindings(tmp_path)
    (tmp_path / 'NOTES.md').write_text('Phase 4a results/phase4a/ report')
    ops = tmp_path / 'docs/phase4a-operations.md'
    ops.parent.mkdir()
    ops.write_text('No limits documented')
    with pytest.raises(ValueError): m.documentation_bindings(tmp_path)


def test_failed_control_remains_disclosed():
    m = tool()
    record = m.qualified_provenance()
    assert record['phase3_status'] == 'passed with qualification'
    assert record['control_status'] == 'failed'


def test_cli_export_handles_private_source_and_host_readable_features(tmp_path):
    import subprocess
    m = tool()
    video = tmp_path / 'private.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'testsrc2=size=64x32:rate=4:duration=0.5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(video)], check=True)
    video.chmod(0o600)
    original = video.read_bytes()
    destination = tmp_path / 'features.npy'
    result = m.cli_request('vqa:phase4a', video, destination)
    assert result['exit_code'] == 0, result['stderr']
    assert m.validate_front_door(result, 'cli')['verified']['cpu_threads'] == 4
    features = np.load(destination)
    assert features.shape == (2, 2048) and features.dtype == np.float32
    assert video.read_bytes() == original and video.stat().st_mode & 0o777 == 0o600


def test_parity_failure_stops_and_preserves_failing_video(tmp_path, monkeypatch):
    import json
    m = tool()
    from src.serving_bundle import read_settings
    config = read_settings(Path('configs/serving.yaml'))
    monkeypatch.setattr(m, 'OUT', tmp_path)
    reference = m.make_reference(config)
    monkeypatch.setattr(m, 'require_build', lambda *args: ({'image_id': 'test-image'}, {}))
    (tmp_path / 'build.json').write_text('{}')
    calls = []
    def failed_cli(image, video, destination):
        calls.append(destination.stem)
        return {'exit_code': 1, 'response': None, 'stderr': 'decode failed'}
    monkeypatch.setattr(m, 'cli_request', failed_cli)
    with pytest.raises(ValueError, match='unsuccessful actual cli'):
        m.check_parity(config, 'test-image', 'http://127.0.0.1:8000')
    record = json.loads((tmp_path / 'parity.json').read_text())
    first = reference['rows'][0]['video_id']
    assert record['status'] == 'failed' and record['failed_video_id'] == first
    assert calls == [first]


@pytest.mark.parametrize('contents', ['winner: unsupported\n', 'winner: [\n'])
def test_report_entrypoint_invalid_configuration_clears_passed_gate(tmp_path, monkeypatch, contents):
    import json
    m = tool()
    monkeypatch.setattr(m, 'OUT', tmp_path)
    config = tmp_path / 'invalid.yaml'
    config.write_text(contents)
    monkeypatch.setattr(m, 'CONFIG', config)
    (tmp_path / 'phase_gate.json').write_text(json.dumps({'status': 'passed'}))
    assert m.main(['--mode', 'report']) == 1
    assert json.loads((tmp_path / 'phase_gate.json').read_text())['status'] == 'failed'


def test_runtime_source_edit_after_build_is_rejected(tmp_path):
    m = tool()
    from src.features import sha256
    hashes = {}
    for name in m.RUNTIME_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original runtime bytes\n')
        hashes[name] = sha256(path)
    proof = {'runtime_source_sha256': hashes}
    m.verify_runtime_sources(proof, tmp_path)
    (tmp_path / 'src/predict.py').write_text('edited after image build\n')
    with pytest.raises(ValueError): m.verify_runtime_sources(proof, tmp_path)
    with pytest.raises(ValueError): m.verify_runtime_sources({'runtime_source_sha256': {}}, tmp_path)


def test_evidence_cli_requests_explicit_json(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    m = tool()
    video = tmp_path / 'input.mp4'
    video.write_bytes(b'fixture')
    commands = []
    def run(command):
        commands.append(command)
        output_mount = command[command.index('--mount', command.index('--mount') + 1) + 1]
        directory = Path(output_mount.split('src=', 1)[1].split(',', 1)[0])
        np.save(directory / 'features.npy', np.ones((2, 2048), dtype=np.float32))
        return SimpleNamespace(returncode=0, stdout=json.dumps(response()), stderr='')
    monkeypatch.setattr(m, 'run', run)
    result = m.cli_request('vqa:phase4a', video, tmp_path / 'features.npy')
    assert result['exit_code'] == 0
    assert '--json' in commands[0]
