"""Phase 4b anchors retain verified Phase 4a bytes after current source changes."""
import copy
import gzip
import importlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from src.features import sha256


@pytest.fixture
def baseline_tree(tmp_path):
    root = tmp_path / 'repo'
    def write(name, payload):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) if isinstance(payload, dict) else payload)
        return sha256(path)
    sources = {'src/original.py': write('src/original.py', 'pass\n')}
    for name in ('model.npz', 'weights.pth'):
        write(name, 'artifact bytes')
    freeze_hash = write('freeze.json', {'weights_sha256': {'resnet50': sha256(root / 'weights.pth')}})
    snapshot = dict(source_files=sources, bindings={'freeze.json': freeze_hash}, outputs_sha256={})
    research = {'results/phase4a/research_snapshot.json': write('results/phase4a/research_snapshot.json', snapshot)}
    config = dict(model='model.npz', weights='weights.pth', preprocessing_freeze='freeze.json',
                  model_sha256=sha256(root / 'model.npz'))
    research['configs/serving.yaml'] = write('configs/serving.yaml', config)
    gate = dict(status='passed', image_id='sha256:' + '1' * 64,
                evidence_sha256={'results/phase4a/parity.json': write('results/phase4a/parity.json', {'status': 'passed'})},
                documentation_sha256={'NOTES.md': write('NOTES.md', 'notes')},
                serving_source_sha256=sources, research_bindings=research,
                report_sha256=write('results/phase4a/report.md', 'report'))
    write('results/phase4a/phase_gate.json', gate)
    return root, gate


@pytest.mark.parametrize('name', ['src/original.py', 'NOTES.md', 'results/phase4a/report.md', 'results/phase4a/parity.json'])
def test_baseline_rejects_stale_bound_file(baseline_tree, tmp_path, name):
    root, _ = baseline_tree
    (root / name).write_text('stale')
    with pytest.raises(ValueError):
        importlib.import_module('scripts.15_demo_check').capture_baseline(root, tmp_path / 'baseline')


def test_baseline_copies_verified_identities(baseline_tree, tmp_path):
    root, gate = baseline_tree
    module = importlib.import_module('scripts.15_demo_check')
    destination = tmp_path / 'baseline'
    manifest = module.capture_baseline(root, destination)
    assert manifest['image_id'] == gate['image_id']
    for name, expected in manifest['files_sha256'].items():
        copied = destination / 'files' / name
        assert copied.read_bytes() == (root / name).read_bytes()
        assert sha256(copied) == expected
    assert (destination / 'files/results/phase4a/phase_gate.json').read_bytes() == (root / 'results/phase4a/phase_gate.json').read_bytes()
    assert not (destination / 'files/weights.pth').exists()
    (root / 'src/original.py').write_text('current implementation')
    assert module.capture_baseline(root, destination) == manifest
    (destination / 'files/src/original.py').write_text('tampered archive')
    with pytest.raises(ValueError):
        module.capture_baseline(root, destination)


def test_baseline_rejects_conflicting_destination(baseline_tree, tmp_path):
    root, _ = baseline_tree
    destination = tmp_path / 'baseline'
    destination.mkdir()
    (destination / 'unrelated').write_text('preserve me')
    with pytest.raises(ValueError):
        importlib.import_module('scripts.15_demo_check').capture_baseline(root, destination)
    assert (destination / 'unrelated').read_text() == 'preserve me'


@pytest.mark.parametrize('change', ['source_roster', 'reference_roster', 'different_gate', 'rehashed_source'])
def test_existing_baseline_rederives_required_bindings(baseline_tree, tmp_path, change):
    root, _ = baseline_tree
    module = importlib.import_module('scripts.15_demo_check')
    destination = tmp_path / 'baseline'
    module.capture_baseline(root, destination)
    record = destination / 'manifest.json'
    manifest = json.loads(record.read_text())
    if change == 'source_roster':
        del manifest['files_sha256']['src/original.py']
        (destination / 'files/src/original.py').unlink()
    if change == 'reference_roster':
        manifest['references_sha256'] = {}
    if change == 'rehashed_source':
        source = destination / 'files/src/original.py'
        source.write_text('altered source')
        manifest['files_sha256']['src/original.py'] = sha256(source)
    if change == 'different_gate':
        gate = json.loads((root / 'results/phase4a/phase_gate.json').read_text())
        gate['image_id'] = 'sha256:' + '2' * 64
        (root / 'results/phase4a/phase_gate.json').write_text(json.dumps(gate))
    record.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.capture_baseline(root, destination)


def test_initial_baseline_rejects_unapproved_weight_bytes(baseline_tree, tmp_path):
    root, _ = baseline_tree
    (root / 'weights.pth').write_bytes(b'corrupt backbone')
    with pytest.raises(ValueError):
        importlib.import_module('scripts.15_demo_check').capture_baseline(root, tmp_path / 'baseline')


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'serving/web'
CONTEXT = {'Dockerfile', '.dockerignore', 'env/serving.lock',
           *(f'src/{name}.py' for name in ('__init__', 'preprocess', 'features', 'baselines', 'metrics', 'cache',
                                           'serving_bundle', 'predict')),
           'serving/__init__.py', 'serving/app.py', 'serving/jobs.py', 'serving/index.html',
           *(f'artifacts/serving/{name}' for name in ('manifest.json', 'model.npz', 'runtime.yaml', 'preprocessing.json',
                                                     'weights/torch/checkpoints/resnet50-11ad3fa6.pth')),
           *(f'serving/web/{name}' for name in ('.env', 'index.html', 'package.json', 'package-lock.json', 'vite.config.js',
                                               'src/styles.css')),
           *(f'serving/web/src/{name}.js' for name in ('main', 'state', 'api', 'mock', 'format', 'result', 'json-view', 'tv'))}


def test_build_context_and_asset_roster():
    lines = [line for line in (ROOT / '.dockerignore').read_text().splitlines() if line.strip()]
    # Docker re-includes a whole subtree for a directory exception, so only exact file exceptions are allowed.
    assert lines[0] == '**' and all(line.startswith('!') and '*' not in line for line in lines[1:])
    included = {line[1:] for line in lines[1:]}
    assert included == CONTEXT and all((ROOT / name).is_file() for name in included)
    banned = {'node_modules', 'dist', 'tests', 'test-results', 'data', 'cache'}
    assert not any(banned & set(Path(name).parts) for name in included)
    dockerfile = (ROOT / 'Dockerfile').read_text()
    assert 'COPY serving/ ' not in dockerfile and 'COPY --from=web /web/dist/ /app/serving/web/dist/' in dockerfile
    node = next(line for line in dockerfile.splitlines() if line.startswith('FROM node:'))
    assert '@sha256:' in node and node.endswith(' AS web')


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    if not (WEB / 'node_modules/.bin/vite').exists():
        pytest.skip('frontend dependencies not installed; run npm ci in serving/web')
    out = tmp_path_factory.mktemp('dist')
    subprocess.run(['npx', 'vite', 'build', '--outDir', str(out), '--emptyOutDir'], cwd=WEB, check=True,
                   capture_output=True)
    return out


def test_production_mock_not_shipped(built):
    files = [path for path in built.rglob('*') if path.is_file()]
    assert not any('mock' in path.name.lower() for path in files)
    scripts = [path for path in files if path.suffix == '.js']
    assert scripts
    for path in scripts:
        text = path.read_text()
        assert 'analyzeMock' not in text and '4c9782aefca4' not in text and "has('mock')" not in text
        assert 'has("mock")' not in text and 'Mock backend' not in text
    # The inert badge markup stays hidden; only the development branch reveals it.
    assert '<span id="mock-badge" class="mock-badge" hidden>' in (built / 'index.html').read_text()
    assert sum(len(gzip.compress(path.read_bytes(), 9)) for path in scripts) < 30000
    assert not any('://' in ref for ref in re.findall(r'(?:src|href)="([^"]+)"', (built / 'index.html').read_text())
                   if not ref.startswith('https://github.com/'))


def demo_tool():
    return importlib.import_module('scripts.15_demo_check')


def prediction(score=3.):
    from src.serving_bundle import MODEL_SHA256, WEIGHTS_SHA256, WEIGHTS_ENUM, PREPROCESSING_SHA256
    return dict(score=score, startup_seconds=.1, timings_seconds=dict(extraction=2., head=.01, total=2.01),
                verified=dict(model_sha256=MODEL_SHA256, weights_sha256=WEIGHTS_SHA256, weights_enum=WEIGHTS_ENUM,
                              preprocessing_sha256=PREPROCESSING_SHA256, cpu_threads=4))


def status_envelope(score=3., name='a.mp4', started=100., wall=5.):
    result = dict(task_id='t', filename=name, clip_seconds=8., created_at='2026-10-31T00:00:00Z', prediction=prediction(score))
    return dict(http_status=202, accepted=dict(task_id='t'), filename=name, request_started=started, upload_seconds=1.,
                request_finished=started + wall, wall_seconds=(started + wall) - started,
                polls=[dict(http_status=200, state='processing'), dict(http_status=200, state='done')],
                final=dict(task_id='t', state='done', stage=None, result=result, error=None))


@pytest.fixture
def image_tree(tmp_path):
    m = demo_tool()
    root = tmp_path / 'repo'
    def write(name, text):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return sha256(path)
    runtime = {name: write(name, name) for name in m.RUNTIME_SOURCES}
    bindings = {name: write(name, 'bound ' + name) for name in m.BUILD_BINDINGS}
    dist = {'index.html': 'a' * 64, 'assets/index.js': 'b' * 64}
    build = dict(status='passed', clean_build=True, image='vqa:phase4b', image_id='sha256:' + '4' * 64, bindings=bindings,
                 environment=dict(m.EXPECTED_ENVIRONMENT))
    sources = dict(image_id=build['image_id'], runtime_source_sha256=runtime, dist_sha256=dist,
                   bytecode=['src/__pycache__/features.cpython-311.pyc'], js_gzip_bytes={'assets/index.js': 9000})
    observed = dict(copy.deepcopy(sources), environment=dict(m.EXPECTED_ENVIRONMENT), fresh_dist=dict(dist))
    return m, root, build, sources, observed


@pytest.mark.parametrize('change', ['image_id', 'runtime_source', 'dist_asset', 'fresh_dist', 'checkpoint', 'lock',
                                    'node_in_image', 'stray_bytecode', 'budget'])
def test_gate_rejects_changed_image_asset_or_source(image_tree, change):
    m, root, build, sources, observed = image_tree
    m.verify_image(build, sources, observed, root)
    if change == 'image_id':
        observed['image_id'] = 'sha256:' + '5' * 64
    if change == 'runtime_source':
        (root / 'serving/jobs.py').write_text('edited after build')
    if change == 'dist_asset':
        observed['dist_sha256']['assets/index.js'] = 'c' * 64
    if change == 'fresh_dist':
        observed['fresh_dist']['assets/index.js'] = 'c' * 64
    if change == 'checkpoint':
        (root / 'results/phase4b/frontend_checkpoint.json').write_text('{}')
    if change == 'lock':
        (root / 'env/serving.lock').write_text('changed lock')
    if change == 'node_in_image':
        observed['environment']['node'] = '/usr/bin/node'
    if change == 'stray_bytecode':
        observed['bytecode'] = sources['bytecode'] = ['src/__pycache__/training.cpython-311.pyc']
    if change == 'budget':
        observed['js_gzip_bytes'] = sources['js_gzip_bytes'] = {'assets/index.js': 30000}
    with pytest.raises(ValueError):
        m.verify_image(build, sources, observed, root)


def test_gate_rederives_all_comparisons():
    import numpy as np
    m = demo_tool()
    cached = np.ones((3, 2048), dtype=np.float16)
    raw = cached.astype(np.float32)
    def row(reference=3., cli=3., api=3., status=3., features=raw):
        return m.parity_row(cached, features, reference, prediction(cli), prediction(api),
                            m.validate_status_door(status_envelope(status), 'a.mp4'))
    assert row()['status'] == 'passed'
    assert row(reference=.01, cli=0., api=0., status=0.)['status'] == 'passed'
    assert row(reference=0., cli=0., api=0., status=1e-6)['status'] == 'passed'
    assert row(reference=.010001, cli=0., api=0., status=0.)['status'] == 'failed'
    assert row(reference=0., cli=0., api=0., status=.0000011)['status'] == 'failed'
    assert row(reference=0., cli=0., api=.0000011, status=0.)['status'] == 'failed'
    assert m.passes_parity(.999, .01, 1e-6) and not m.passes_parity(.998999, .01, 1e-6)
    for features in (raw[:2], np.full_like(raw, np.nan)):
        with pytest.raises(ValueError):
            row(features=features)
    broken = status_envelope()
    broken['final']['result']['prediction']['score'] = float('nan')
    for envelope in (broken, {**status_envelope(), 'final': None}, {**status_envelope(), 'http_status': 503},
                     {**status_envelope(), 'filename': 'other.mp4'}):
        with pytest.raises(ValueError):
            m.validate_status_door(envelope, 'a.mp4')
    unknown = status_envelope()
    unknown['final']['result']['extra'] = 1
    with pytest.raises(ValueError):
        m.validate_status_door(unknown, 'a.mp4')
    rows = [dict(video_id=key, **row()) for key in ('a', 'b')]
    summary = m.summarize_parity(rows, ['a', 'b'])
    assert summary['minimum_cosine'] == pytest.approx(1.) and summary['status'] == 'passed'
    for changed in (rows[:1], rows + rows[:1], [rows[1], rows[0]]):
        with pytest.raises(ValueError):
            m.summarize_parity(changed, ['a', 'b'])
    with pytest.raises(ValueError):
        m.check_summary(dict(summary, minimum_cosine=1.), dict(summary, maximum_door_difference=0.5))


def latency_rows(m, ids, image='sha256:' + '4' * 64):
    return [dict(video_id=key, repeat=repeat, role='warm', image_id=image, container_id='c', duration_seconds=8.,
                 **m.latency_fields(status_envelope(name=key + '.mp4', wall=float(repeat))), envelope=status_envelope(name=key + '.mp4', wall=float(repeat)))
            for key in ids for repeat in (1, 2, 3)]


def test_latency_requires_sixty_current_image_rows():
    m = demo_tool()
    ids = [f'v{index}' for index in range(20)]
    durations = dict.fromkeys(ids, 8.)
    image = 'sha256:' + '4' * 64
    rows = latency_rows(m, ids)
    summary = m.summarize_demo_latency(rows, durations, image, 'c')
    assert len(rows) == 60 and summary['budget_passed'] and summary['maximum_seconds_per_ten_seconds'] == 3.75
    changed = [rows[:-1], rows + rows[:1], rows[1:] + rows[:1], [{**r, 'role': 'warmup'} if i == 0 else r for i, r in enumerate(rows)],
               latency_rows(m, ids, 'sha256:' + '7' * 64), [{**r, 'wall_seconds': 9.} if i == 0 else r for i, r in enumerate(rows)],
               [{**r, 'container_id': 'old'} if i == 0 else r for i, r in enumerate(rows)]]
    for bad in changed:
        with pytest.raises(ValueError):
            m.summarize_demo_latency(bad, durations, image, 'c')
    slow = copy.deepcopy(rows)
    slow[5]['envelope'] = status_envelope(name=slow[5]['video_id'] + '.mp4', wall=96.1)
    slow[5].update(m.latency_fields(slow[5]['envelope']))
    assert not m.summarize_demo_latency(slow, durations, image, 'c')['budget_passed']
    edge = copy.deepcopy(rows)
    edge[5]['envelope'] = status_envelope(name=edge[5]['video_id'] + '.mp4', wall=96.)
    edge[5].update(m.latency_fields(edge[5]['envelope']))
    assert m.summarize_demo_latency(edge, durations, image, 'c')['budget_passed']


def test_report_failure_invalidates_existing_gate(tmp_path, monkeypatch):
    m = demo_tool()
    monkeypatch.setattr(m, 'OUT', tmp_path)
    (tmp_path / 'phase_gate.json').write_text(json.dumps({'status': 'passed'}))
    def missing(config, url):
        raise ValueError('missing evidence: parity.json')
    monkeypatch.setattr(m, 'collect_demo_evidence', missing)
    assert m.main(['--mode', 'report']) == 1
    gate = json.loads((tmp_path / 'phase_gate.json').read_text())
    assert gate['status'] == 'failed' and 'parity.json' in gate['reason']
    (tmp_path / 'phase_gate.json').write_text(json.dumps({'status': 'passed'}))
    monkeypatch.setattr(m, 'check_demo_parity', lambda *args: (_ for _ in ()).throw(ValueError('stop')))
    assert m.main(['--mode', 'parity']) == 1
    assert json.loads((tmp_path / 'phase_gate.json').read_text())['status'] == 'failed'


def test_qualified_research_status_survives(tmp_path, monkeypatch):
    m = demo_tool()
    monkeypatch.setattr(m, 'OUT', tmp_path)
    m.invalidate_gate('test')
    gate = json.loads((tmp_path / 'phase_gate.json').read_text())
    assert gate['phase3_status'] == 'passed with qualification' and gate['control_status'] == 'failed'
    assert 'control: **failed**' in m.report_text(dict(fixed_videos=20, image_id='i', parity=dict(minimum_cosine=1.,
        maximum_cache_difference=0., maximum_door_difference=0.), latency=dict(maximum_seconds_per_ten_seconds=1.,
        median_seconds_per_ten_seconds=1., budget_seconds_per_ten_seconds=120, median_polling_overhead_seconds=.5,
        maximum_polling_overhead_seconds=1.), js_gzip_total=9000, accessibility=1.,
        baseline_image_id='b', phase3_status='passed with qualification', control_status='failed'))
