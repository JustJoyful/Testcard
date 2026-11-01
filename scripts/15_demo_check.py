"""Independent Phase 4b evidence; preserve the verified Phase 4a baseline first."""
import argparse
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter, sleep
from urllib.parse import urlparse

import httpx
import numpy as np
import yaml

from src.cache import atomic_json, atomic_write, csv_bytes, read_branch
from src.features import sha256
from src.serving_bundle import read_settings, verify_file, verify_snapshot

OUT = Path('results/phase4b')
GATE = 'results/phase4a/phase_gate.json'
HASH_MAPS = ('evidence_sha256', 'documentation_sha256', 'research_bindings', 'serving_source_sha256')


def baseline_bindings(root: Path, reference_root: Path):
    gate = json.loads((root / GATE).read_text())
    if gate['status'] != 'passed':
        raise ValueError('Phase 4a must have passed')
    files = {GATE: sha256(root / GATE), 'results/phase4a/report.md': gate['report_sha256']}
    for field in HASH_MAPS:
        for name, expected in gate[field].items():
            if name in files and files[name] != expected:
                raise ValueError(f'conflicting baseline binding: {name}')
            files[name] = expected
    for name, expected in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('baseline path must be relative')
        verify_file(root / name, expected)
    snapshot = json.loads((root / 'results/phase4a/research_snapshot.json').read_text())
    verify_snapshot(dict(snapshot, outputs_sha256={}), root)
    for name in {**snapshot['source_files'], **snapshot['bindings']}:
        # The decision log includes the verified packaging append, not only the research prefix.
        files[name] = sha256(root / name)
    references = {'results/phase3/' + name: value for name, value in snapshot['outputs_sha256'].items()}
    config = yaml.safe_load((root / 'configs/serving.yaml').read_text())
    freeze = json.loads((root / config['preprocessing_freeze']).read_text())
    references.update({config['model']: config['model_sha256'],
                       config['weights']: freeze['weights_sha256']['resnet50'],
                       config['preprocessing_freeze']: files[config['preprocessing_freeze']]})
    for name, expected in references.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('reference path must be relative')
        verify_file(reference_root / name, expected)
    # Keep the entire small evidence packet, including raw responses and test/build logs.
    for path in (root / 'results/phase4a').rglob('*'):
        if path.is_file() and 'features' not in path.relative_to(root / 'results/phase4a').parts:
            files[str(path.relative_to(root))] = sha256(path)
    return dict(image_id=gate['image_id'], files_sha256=files, references_sha256=references)


def capture_baseline(root: Path, destination: Path) -> dict:
    root, destination = Path(root), Path(destination)
    if destination.exists():
        try:
            manifest = json.loads((destination / 'manifest.json').read_text())
            archive = destination / 'files'
            verify_file(archive / GATE, sha256(root / GATE))
            required = baseline_bindings(archive, root)
            if manifest != required:
                raise ValueError('baseline roster or identity conflicts with captured gate')
            for name, expected in manifest['files_sha256'].items():
                verify_file(archive / name, expected)
            return manifest
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError('existing baseline is incomplete or conflicts') from exc

    manifest = baseline_bindings(root, root)
    files = manifest['files_sha256']
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='baseline-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'baseline'
        for name, expected in files.items():
            target = staged / 'files' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
            verify_file(target, expected)
        atomic_json(staged / 'manifest.json', manifest)
        staged.rename(destination)
    return manifest


# ---------- Phase 4b final-image evidence; Phase 4a outputs are only read, never written ----------

S14 = importlib.import_module('scripts.14_serving_check')
CONFIG = Path('configs/serving.yaml')
WEB = Path('serving/web')
RUNTIME_SOURCES = (*S14.RUNTIME_SOURCES, 'serving/jobs.py')
FRONTEND_INPUTS = tuple('serving/web/' + name for name in (
    '.env', 'index.html', 'package.json', 'package-lock.json', 'vite.config.js', 'src/styles.css',
    *(f'src/{module}.js' for module in ('main', 'state', 'api', 'mock', 'format', 'result', 'json-view', 'tv'))))
BUILD_BINDINGS = ('Dockerfile', '.dockerignore', 'env/serving.lock', 'configs/serving.yaml',
                  'artifacts/serving/manifest.json', 'DESIGN.md', 'DECISIONS.md', 'results/phase4b/decision_amendment.json',
                  'results/phase4b/frontend_checkpoint.json', *FRONTEND_INPUTS)
EXPECTED_ENVIRONMENT = dict(python='3.11.15', torch='2.14.1+cpu', torchvision='0.29.1+cpu', cuda=None,
                            cuda_available=False, uid=10001, node=None, npm=None)
OUTER = {'task_id', 'filename', 'clip_seconds', 'created_at', 'prediction'}
BUILD_LOG = OUT / 'package/build.log'
JS_BUDGET = 30000
INSIDE = r'''
import gzip, hashlib, json, os, shutil, sys, torch, torchvision
from pathlib import Path
files = {str(p.relative_to('/app')): p.read_bytes() for root in ('/app/src', '/app/serving')
         for p in sorted(Path(root).rglob('*')) if p.is_file()}
print(json.dumps(dict(files={k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    js_gzip_bytes={k[len('serving/web/dist/'):]: len(gzip.compress(v, 9)) for k, v in files.items()
                   if k.startswith('serving/web/dist/') and k.endswith('.js')},
    environment=dict(python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__,
        cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(), uid=os.getuid(),
        node=shutil.which('node'), npm=shutil.which('npm')))))
'''


def poll_cadence(path=WEB / 'src/api.js'):
    """The browser contract's real timings, read from the shipped transport source."""
    text = Path(path).read_text()
    def constant(name):
        found = re.search(rf'export const {name} = ([0-9 *]+);', text)
        if not found:
            raise ValueError(f'polling constant {name} missing from api.js')
        return math.prod(int(part) for part in found.group(1).split('*')) / 1000
    return dict(fast_seconds=constant('FAST_POLL_MS'), slow_seconds=constant('SLOW_POLL_MS'),
                fast_window_seconds=constant('FAST_POLL_WINDOW_MS'), deadline_seconds=constant('TIMEOUT_MS'))


def invalidate_gate(reason):
    try:
        provenance = S14.qualified_provenance()
    except (ValueError, KeyError):
        provenance = {}
    atomic_json(OUT / 'phase_gate.json', dict(status='failed', reason=reason, **provenance))


def observe_image(image):
    """Read the runtime roster, built assets and environment from inside the actual image."""
    image_id = S14.image_identity(image)
    inside = json.loads(S14.checked(['docker', 'run', '--rm', '--network', 'none', *S14.CAPS,
                                     '--entrypoint', 'python', image_id, '-c', INSIDE]))
    files, prefix = inside['files'], 'serving/web/dist/'
    return dict(image_id=image_id, environment=inside['environment'], js_gzip_bytes=inside['js_gzip_bytes'],
                dist_sha256={k[len(prefix):]: v for k, v in files.items() if k.startswith(prefix)},
                bytecode=sorted(k for k in files if '__pycache__' in Path(k).parts),
                runtime_source_sha256={k: v for k, v in files.items()
                                       if not k.startswith(prefix) and '__pycache__' not in Path(k).parts})


def fresh_dist():
    """Rebuild the current frontend source; the image must ship exactly these bytes."""
    with tempfile.TemporaryDirectory(prefix='vqa-dist-') as temporary:
        S14.checked(['bash', '-c', f'cd {WEB} && npx vite build --outDir {temporary} --emptyOutDir'])
        root = Path(temporary)
        return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_image(build, sources, observed, root=Path('.')):
    if build.get('status') != 'passed' or not build.get('clean_build'):
        raise ValueError('unverified clean image build')
    if not observed['image_id'] == build['image_id'] == sources['image_id']:
        raise ValueError('image identity changed since evidence capture')
    if observed['environment'] != EXPECTED_ENVIRONMENT or build['environment'] != EXPECTED_ENVIRONMENT:
        raise ValueError('image is not the CPU-only, Node-free, UID 10001 runtime')
    for key in ('runtime_source_sha256', 'dist_sha256', 'bytecode', 'js_gzip_bytes'):
        if observed[key] != sources[key]:
            raise ValueError(f'image {key} differs from recorded roster')
    if set(sources['runtime_source_sha256']) != set(RUNTIME_SOURCES):
        raise ValueError('incomplete or extra runtime source in image')
    for name, digest in sources['runtime_source_sha256'].items():
        verify_file(root / name, digest)
    allowed = {f'src/__pycache__/{Path(n).stem}.cpython-311.pyc' for n in RUNTIME_SOURCES if n.startswith('src/')}
    if not set(sources['bytecode']) <= allowed:
        raise ValueError('unexpected bytecode in image')
    dist = sources['dist_sha256']
    if 'index.html' not in dist or observed['fresh_dist'] != dist:
        raise ValueError('shipped assets differ from a fresh build of the current frontend source')
    if set(build['bindings']) != set(BUILD_BINDINGS):
        raise ValueError('incomplete build binding roster')
    for name, digest in build['bindings'].items():
        verify_file(root / name, digest)
    scripts = sources['js_gzip_bytes']
    if not scripts or set(scripts) != {n for n in dist if n.endswith('.js')} or sum(scripts.values()) >= JS_BUDGET:
        raise ValueError('production JavaScript missing or over the 30 KB gzip budget')


def verify_build_log(build):
    text = Path(build['build_log']).read_text()
    verify_file(build['build_log'], build['build_log_sha256'])
    if (f"Successfully built {build['image_id'][7:19]}" not in text or 'exit=0' not in text or 'Using cache' in text
            or 'docker build --no-cache' not in ' '.join(build['command'])):
        raise ValueError('build log is not the clean build of this image')


def approved_archive():
    approved = S14.read_json('results/phase4a/build.json')
    dockerfile = Path('Dockerfile').read_text()
    if approved['archive_url'] not in dockerfile or f"echo '{approved['archive_sha256']}  /tmp/ffmpeg.tar.xz'" not in dockerfile:
        raise ValueError('Dockerfile decoder URL/SHA256 differ from the approved immutable archive')
    return approved['archive_url'], approved['archive_sha256']


def verify_decoders(versions, build):
    if versions['image_id'] != build['image_id'] or (versions['archive_url'], versions['archive_sha256']) != approved_archive():
        raise ValueError('decoder versions belong to another image or archive')
    for binary in ('ffmpeg', 'ffprobe'):
        pair = versions['commands'][binary]
        if (pair['host']['exit_code'] or pair['image']['exit_code'] or
                f'{binary} version n9.0.2 ' not in pair['host']['stdout'].splitlines()[0] or
                f'{binary} version n9.0.2-37-gae19868753-20261029 ' not in pair['image']['stdout'].splitlines()[0]):
            raise ValueError('unapproved host/image decoder versions')


def active_container(image_id, url):
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or not parsed.port:
        raise ValueError('measure over local loopback HTTP with an explicit port')
    binding = [{'HostIp': '127.0.0.1', 'HostPort': str(parsed.port)}]
    found = [c for cid in S14.checked(['docker', 'ps', '-q', '--no-trunc']).split()
             for c in json.loads(S14.checked(['docker', 'inspect', cid]))
             if (c['HostConfig'].get('PortBindings') or {}).get('8000/tcp') == binding]
    if len(found) != 1 or found[0]['Image'] != image_id:
        raise ValueError('exactly one container of the measured image must serve the URL')
    host = found[0]['HostConfig']
    caps = dict(nano_cpus=host['NanoCpus'], memory=host['Memory'], pids_limit=host['PidsLimit'],
                read_only=host['ReadonlyRootfs'], tmpfs=host['Tmpfs'])
    if caps != dict(nano_cpus=4000000000, memory=4294967296, pids_limit=128, read_only=True,
                    tmpfs={'/tmp': 'rw,nosuid,size=256m'}) or found[0]['Config']['User'] != '10001:10001':
        raise ValueError('wrong resource caps or user')
    health = httpx.get(url + '/health', timeout=10)
    if health.status_code != 200 or health.json().get('verified') != S14.EXPECTED_VERIFIED:
        raise ValueError('API is not the verified runtime')
    return dict(container_id=found[0]['Id'], name=found[0]['Name'].lstrip('/'), image_id=image_id, url=url, caps=caps)


def capture_image(image, url):
    observed = observe_image(image)
    observed['fresh_dist'] = fresh_dist()
    archive_url, archive_sha256 = approved_archive()
    node = next(line for line in Path('Dockerfile').read_text().splitlines() if line.startswith('FROM node:'))
    build = dict(status='passed', clean_build=True, command=['docker', 'build', '--no-cache', '-t', image, '.'],
                 image=image, image_id=observed['image_id'], build_log=str(BUILD_LOG), build_log_sha256=sha256(BUILD_LOG),
                 node_image=node.split()[1], archive_url=archive_url, archive_sha256=archive_sha256,
                 environment=observed['environment'], resource_caps=S14.CAPS,
                 bindings={name: sha256(name) for name in BUILD_BINDINGS})
    sources = {k: observed[k] for k in ('image_id', 'runtime_source_sha256', 'dist_sha256', 'bytecode', 'js_gzip_bytes')}
    verify_image(build, sources, observed)
    verify_build_log(build)
    commands = {}
    for binary in ('ffmpeg', 'ffprobe'):
        pair = {}
        for side, command in (('host', [binary, '-version']), ('image', ['docker', 'run', '--rm', '--network', 'none',
                              *S14.CAPS, '--entrypoint', binary, observed['image_id'], '-version'])):
            result = S14.run(command)
            pair[side] = dict(command=command, stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode)
        commands[binary] = pair
    versions = dict(image_id=observed['image_id'], archive_url=archive_url, archive_sha256=archive_sha256, commands=commands)
    verify_decoders(versions, build)
    active = active_container(observed['image_id'], url)
    for name, record in (('build', build), ('image_sources', sources), ('decoder_versions', versions), ('active_container', active)):
        atomic_json(OUT / f'{name}.json', record)
    return build, active


def status_request(client, url, video, filename, cadence):
    """Upload once and poll exactly like the page: immediately, then fast for the window, then slow."""
    started = perf_counter()
    with Path(video).open('rb') as handle:
        response = client.post(url + '/upload', files={'file': (filename, handle, 'video/mp4')})
    accepted_at = perf_counter()
    polls, final = [], None
    if response.status_code == 202:
        task_id = response.json()['task_id']
        while perf_counter() - started < cadence['deadline_seconds']:
            reply = client.get(f'{url}/status/{task_id}')
            final = reply.json()
            polls.append(dict(at_seconds=perf_counter() - started, http_status=reply.status_code,
                              state=final.get('state'), stage=final.get('stage')))
            if reply.status_code != 200 or final.get('state') in ('done', 'error'):
                break
            fast = perf_counter() - accepted_at < cadence['fast_window_seconds']
            sleep(cadence['fast_seconds'] if fast else cadence['slow_seconds'])
    finished = perf_counter()
    return dict(http_status=response.status_code, accepted=response.json(), filename=filename, polls=polls, final=final,
                request_started=started, request_finished=finished, wall_seconds=finished - started,
                upload_seconds=accepted_at - started)


def validate_status_door(envelope, filename):
    try:
        if envelope['http_status'] != 202 or envelope['filename'] != filename:
            raise ValueError('upload was not accepted for this clip')
        task_id, final, polls = envelope['accepted']['task_id'], envelope['final'], envelope['polls']
        if final['state'] != 'done' or final['task_id'] != task_id or final['error'] is not None:
            raise ValueError('status did not finish successfully')
        result = final['result']
        if set(result) != OUTER or result['task_id'] != task_id or result['filename'] != filename:
            raise ValueError('status wrapper fields changed')
        clip = result['clip_seconds']
        if isinstance(clip, bool) or not math.isfinite(clip) or clip <= 0 or not re.fullmatch(
                r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z', result['created_at']):
            raise ValueError('invalid status metadata')
        if not polls or any(p['http_status'] != 200 for p in polls) or polls[-1]['state'] != 'done':
            raise ValueError('incomplete polling record')
        wall, upload = envelope['wall_seconds'], envelope['upload_seconds']
        if wall != envelope['request_finished'] - envelope['request_started'] or not 0 <= upload <= wall or wall <= 0:
            raise ValueError('invalid client wall timing')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('incomplete actual status front-door response') from exc
    return S14.validate_front_door(dict(http_status=200, response=result['prediction']), 'api')


def parity_row(cached, raw, reference, cli, api, status):
    row = S14.compare_features(cached, raw, reference, cli, api)
    scores = (cli['score'], api['score'], status['score'])
    row.update(cached_to_status_absolute=abs(status['score'] - reference), status_cli_absolute=abs(status['score'] - cli['score']),
               status_api_absolute=abs(status['score'] - api['score']), maximum_door_difference=max(scores) - min(scores))
    cache = max(row['cached_to_cli_absolute'], row['cached_to_api_absolute'], row['cached_to_status_absolute'])
    row['status'] = 'passed' if S14.passes_parity(row['minimum_cosine'], cache, row['maximum_door_difference']) else 'failed'
    return row


def passes_parity(cosine, cache_delta, door_delta):
    return S14.passes_parity(cosine, cache_delta, door_delta)


def summarize_parity(rows, ids):
    S14.validate_ids([r['video_id'] for r in rows], ids)
    return dict(status='passed' if all(r['status'] == 'passed' for r in rows) else 'failed', fixed_videos=len(rows),
                minimum_cosine=min(r['minimum_cosine'] for r in rows),
                maximum_cache_difference=max(max(r['cached_to_cli_absolute'], r['cached_to_api_absolute'],
                                                 r['cached_to_status_absolute']) for r in rows),
                maximum_door_difference=max(r['maximum_door_difference'] for r in rows))


def check_summary(saved, derived):
    if saved != derived:
        raise ValueError('saved summary does not reproduce from raw evidence')


DOORS = ('cli', 'api', 'status')


def check_demo_parity(config: dict, image: str, url: str) -> dict:
    context = S14.research_inputs(config)
    references = S14.reference_for(context)
    build, active = capture_image(image, url)
    cadence, settings = poll_cadence(), context['video_config']['datasets']['konvid']
    rows, key = [], None
    record = dict(status='failed', rows=rows, image_id=build['image_id'], container_id=active['container_id'],
                  bindings=context['bindings'], build_sha256=sha256(OUT / 'build.json'),
                  image_sources_sha256=sha256(OUT / 'image_sources.json'), cadence=cadence,
                  reference_sha256=sha256(S14.OUT / 'reference.meta.json'), measurement_tool_sha256=sha256(__file__))
    try:
        with httpx.Client(timeout=300) as client:
            for key in context['ids']:
                source = context['sources'][key]
                paths = {door: OUT / door / (key + '.json') for door in DOORS}
                features = OUT / 'features' / (key + '.npy')
                with S14.source_video(source, settings) as video:
                    cli = S14.cli_request(build['image_id'], video, features)
                    atomic_json(paths['cli'], cli)
                    cli_result = S14.validate_front_door(cli, 'cli')
                    api = S14.api_request(client, url, video)
                    atomic_json(paths['api'], api)
                    api_result = S14.validate_front_door(api, 'api')
                    status = status_request(client, url, video, key + '.mp4', cadence)
                    atomic_json(paths['status'], status)
                    status_result = validate_status_door(status, key + '.mp4')
                cached = read_branch(context['cache'], 'konvid', key, 'resnet50', context['records'], source)
                row = dict(video_id=key, source_sha256=source['source_sha256'], cache_sha256=references[key]['cache_sha256'],
                           feature_path=str(features), feature_sha256=sha256(features),
                           **{f'{door}_path': str(path) for door, path in paths.items()},
                           **{f'{door}_sha256': sha256(path) for door, path in paths.items()},
                           **parity_row(cached, np.load(features, allow_pickle=False), references[key]['score'],
                                        cli_result, api_result, status_result))
                rows.append(row)
                atomic_json(OUT / 'parity.json', record)
                print(json.dumps({k: row[k] for k in ('video_id', 'status', 'minimum_cosine', 'maximum_door_difference')}), flush=True)
                if row['status'] != 'passed':
                    raise ValueError(f'parity failed for {key}; thresholds unchanged; stop and report')
        record['summary'] = summarize_parity(rows, context['ids'])
        record['status'] = record['summary']['status']
    except Exception as exc:
        record.update(failed_video_id=key, error=str(exc))
        atomic_json(OUT / 'parity.json', record)
        raise
    atomic_json(OUT / 'parity.json', record)
    atomic_write(OUT / 'parity.csv', csv_bytes([{k: v for k, v in r.items()} for r in rows], list(rows[0])))
    return record


def validate_demo_parity(context, references, build):
    record = S14.read_json(OUT / 'parity.json')
    if record['status'] != 'passed' or record['bindings'] != context['bindings'] or record['image_id'] != build['image_id']:
        raise ValueError('missing/failed/stale Phase 4b parity')
    verify_file(OUT / 'build.json', record['build_sha256'])
    verify_file(OUT / 'image_sources.json', record['image_sources_sha256'])
    verify_file(S14.OUT / 'reference.meta.json', record['reference_sha256'])
    rows = record['rows']
    for row in rows:
        key = row['video_id']
        for kind in ('feature', *DOORS):
            verify_file(row[kind + '_path'], row[kind + '_sha256'])
        cli = S14.validate_front_door(S14.read_json(row['cli_path']), 'cli')
        api = S14.validate_front_door(S14.read_json(row['api_path']), 'api')
        status = validate_status_door(S14.read_json(row['status_path']), key + '.mp4')
        cached = read_branch(context['cache'], 'konvid', key, 'resnet50', context['records'], context['sources'][key])
        actual = parity_row(cached, np.load(row['feature_path'], allow_pickle=False), references[key]['score'], cli, api, status)
        if actual['status'] != 'passed' or any(row[k] != v for k, v in actual.items()):
            raise ValueError(f'parity evidence for {key} does not reproduce')
    summary = summarize_parity(rows, context['ids'])
    check_summary(record['summary'], summary)
    return record


LATENCY_FIELDS = ('wall_seconds', 'upload_seconds', 'extraction', 'head', 'internal_total', 'polling_overhead_seconds',
                  'poll_count', 'request_started', 'request_finished')


def latency_fields(envelope):
    timing = envelope['final']['result']['prediction']['timings_seconds']
    return dict(wall_seconds=envelope['wall_seconds'], upload_seconds=envelope['upload_seconds'],
                extraction=timing['extraction'], head=timing['head'], internal_total=timing['total'],
                polling_overhead_seconds=envelope['wall_seconds'] - envelope['upload_seconds'] - timing['total'],
                poll_count=len(envelope['polls']), request_started=envelope['request_started'],
                request_finished=envelope['request_finished'])


def summarize_demo_latency(rows, durations, image_id, container_id):
    ids = list(durations)
    if [(r['video_id'], r['repeat']) for r in rows] != [(key, repeat) for key in ids for repeat in (1, 2, 3)]:
        raise ValueError('latency requires exactly three ordered warm repeats for every fixed ID')
    for row in rows:
        if (row['role'] != 'warm' or row['image_id'] != image_id or row['container_id'] != container_id
                or row['duration_seconds'] != durations[row['video_id']] or row['duration_seconds'] <= 0):
            raise ValueError('cold, warmup or other-image rows cannot enter the warm summary')
        validate_status_door(row['envelope'], row['video_id'] + '.mp4')
        if {k: row[k] for k in LATENCY_FIELDS} != latency_fields(row['envelope']):
            raise ValueError('latency row does not match its raw status envelope')
    normalized = [r['wall_seconds'] * 10 / r['duration_seconds'] for r in rows]
    overhead = [r['polling_overhead_seconds'] for r in rows]
    per_video = {key: max(n for r, n in zip(rows, normalized) if r['video_id'] == key) for key in ids}
    return dict(rows=len(rows), maximum_seconds_per_ten_seconds=max(normalized), median_seconds_per_ten_seconds=median(normalized),
                budget_seconds_per_ten_seconds=120, budget_passed=max(normalized) <= 120,
                median_polling_overhead_seconds=median(overhead), maximum_polling_overhead_seconds=max(overhead),
                per_video_maximum_seconds_per_ten_seconds=per_video)


def wait_ready(url, started, limit=180):
    while perf_counter() - started < limit:
        try:
            if httpx.get(url + '/health', timeout=5).json().get('status') == 'ready':
                return perf_counter() - started
        except (httpx.HTTPError, ValueError):
            pass
        sleep(.25)
    raise ValueError('restarted container never became ready')


def measure_demo_latency(config: dict, image: str, url: str) -> dict:
    context = S14.research_inputs(config)
    references = S14.reference_for(context)
    build, sources = S14.read_json(OUT / 'build.json'), S14.read_json(OUT / 'image_sources.json')
    observed = observe_image(image)
    observed['fresh_dist'] = fresh_dist()
    verify_image(build, sources, observed)
    validate_demo_parity(context, references, build)
    previous = active_container(build['image_id'], url)
    # Cold measurement restarts only the Phase 4b test container.
    S14.checked(['docker', 'rm', '-f', previous['container_id']])
    started = perf_counter()
    S14.checked(['docker', 'run', '-d', '--name', previous['name'], *S14.CAPS, '-p',
                 f'127.0.0.1:{urlparse(url).port}:8000', build['image_id']])
    ready = wait_ready(url, started)
    active = active_container(build['image_id'], url)
    atomic_json(OUT / 'active_container.json', active)
    cadence, settings = poll_cadence(), context['video_config']['datasets']['konvid']
    ids = context['ids']
    durations = {key: float(context['sources'][key]['duration_s']) for key in ids}
    rows = []
    with httpx.Client(timeout=300) as client:
        with S14.source_video(context['sources'][ids[0]], settings) as video:
            first = status_request(client, url, video, ids[0] + '.mp4', cadence)
            validate_status_door(first, ids[0] + '.mp4')
            atomic_json(OUT / 'cold.json', dict(image_id=build['image_id'], container_id=active['container_id'], video_id=ids[0],
                container_startup_seconds=ready, startup_and_first_task_seconds=first['request_finished'] - started,
                first_task=first, **latency_fields(first)))
            warmup = status_request(client, url, video, ids[0] + '.mp4', cadence)
            validate_status_door(warmup, ids[0] + '.mp4')
            atomic_json(OUT / 'warmup.json', dict(excluded_from_warm_summary=True, image_id=build['image_id'],
                container_id=active['container_id'], video_id=ids[0], envelope=warmup, **latency_fields(warmup)))
        for key in ids:
            with S14.source_video(context['sources'][key], settings) as video:
                for repeat in (1, 2, 3):
                    envelope = status_request(client, url, video, key + '.mp4', cadence)
                    validate_status_door(envelope, key + '.mp4')
                    rows.append(dict(video_id=key, repeat=repeat, role='warm', image_id=build['image_id'],
                                     container_id=active['container_id'], duration_seconds=durations[key],
                                     **latency_fields(envelope), envelope=envelope))
                    atomic_json(OUT / 'latency.raw.json', rows)
                    print(json.dumps(dict(video_id=key, repeat=repeat, wall=envelope['wall_seconds'])), flush=True)
    summary = summarize_demo_latency(rows, durations, build['image_id'], active['container_id'])
    record = dict(status='passed' if summary['budget_passed'] else 'failed', summary=summary, cadence=cadence,
                  image_id=build['image_id'], container_id=active['container_id'], bindings=context['bindings'],
                  parity_sha256=sha256(OUT / 'parity.json'), raw_sha256=sha256(OUT / 'latency.raw.json'),
                  cold_sha256=sha256(OUT / 'cold.json'), warmup_sha256=sha256(OUT / 'warmup.json'),
                  measurement_tool_sha256=sha256(__file__), platform=platform.platform(), cpu=S14.checked(['lscpu']),
                  resource_caps=build['resource_caps'], cpu_threads=4)
    atomic_json(OUT / 'latency.json', record)
    fields = ['video_id', 'repeat', 'role', 'duration_seconds', *LATENCY_FIELDS]
    atomic_write(OUT / 'latency.csv', csv_bytes([{k: r[k] for k in fields} for r in rows], fields))
    if record['status'] != 'passed':
        raise ValueError('warm loopback budget failed; stop and report')
    return record


def validate_demo_latency(context, build, active):
    latency = S14.read_json(OUT / 'latency.json')
    if (latency['status'] != 'passed' or latency['bindings'] != context['bindings'] or latency['image_id'] != build['image_id']
            or latency['container_id'] != active['container_id'] or latency['cadence'] != poll_cadence()):
        raise ValueError('failed/stale latency identities')
    for name in ('parity', 'cold', 'warmup'):
        verify_file(OUT / (name + '.json'), latency[name + '_sha256'])
    verify_file(OUT / 'latency.raw.json', latency['raw_sha256'])
    durations = {key: float(context['sources'][key]['duration_s']) for key in context['ids']}
    summary = summarize_demo_latency(S14.read_json(OUT / 'latency.raw.json'), durations, build['image_id'], active['container_id'])
    check_summary(latency['summary'], summary)
    if not summary['budget_passed']:
        raise ValueError('warm loopback budget failed')
    first = context['ids'][0] + '.mp4'
    cold, warmup = S14.read_json(OUT / 'cold.json'), S14.read_json(OUT / 'warmup.json')
    for record, envelope in ((cold, cold['first_task']), (warmup, warmup['envelope'])):
        if record['image_id'] != build['image_id'] or record['container_id'] != active['container_id']:
            raise ValueError('cold/warmup request belongs to another image or container')
        validate_status_door(envelope, first)
    if not warmup['excluded_from_warm_summary']:
        raise ValueError('warmup must be excluded')
    return dict(latency, summary=summary)


TEST_SOURCES = ('tests/test_predict_cli.py', 'tests/test_serving_bundle.py', 'tests/test_serving_api.py',
                'tests/test_serving_evidence.py', 'tests/test_demo_jobs.py', 'tests/test_demo_api.py', 'tests/test_demo_evidence.py')


def tested_sources():
    names = [*RUNTIME_SOURCES, *TEST_SOURCES, *BUILD_BINDINGS, 'scripts/14_serving_check.py', 'scripts/15_demo_check.py',
             'serving/web/playwright.config.js',
             *sorted(str(p) for p in (WEB / 'tests').rglob('*') if p.is_file())]
    return {name: sha256(name) for name in dict.fromkeys(names)}


def run_tests(config: dict, url: str) -> dict:
    context = S14.research_inputs(config)
    chrome = sorted(Path.home().glob('.cache/ms-playwright/chromium-*/chrome-linux*/chrome'))
    if not chrome:
        raise ValueError('Playwright Chromium is required for browser checks')
    (OUT / 'tests').mkdir(parents=True, exist_ok=True)
    runs = {}
    with S14.source_video(context['sources'][context['ids'][0]], context['video_config']['datasets']['konvid']) as video:
        commands = dict(
            serving=(['.venv-serving/bin/python', '-m', 'pytest', *TEST_SOURCES, '-q'], '.', {}),
            node=(['npm', 'test'], str(WEB), {}),
            browser=(['npx', 'playwright', 'test'], str(WEB), {}),
            container_browser=(['npx', 'playwright', 'test'], str(WEB),
                               dict(VQA_CONTAINER_URL=url, VQA_REAL_VIDEO=str(Path(video).resolve()),
                                    VQA_ACCEPTANCE_OUT=str((OUT / 'tests').resolve()))),
            lighthouse=(['npx', 'lighthouse', url + '/', '--only-categories=accessibility', '--preset=desktop', '--output=json',
                         f'--output-path={(OUT / "lighthouse.json").resolve()}', '--chrome-flags=--headless=new', '--quiet'],
                        str(WEB), dict(CHROME_PATH=str(chrome[-1]))))
        for name, (command, cwd, extra) in commands.items():
            log = OUT / 'tests' / f'{name}.log'
            result = subprocess.run(command, cwd=cwd, env={**os.environ, **extra}, capture_output=True, text=True, timeout=1800)
            atomic_write(log, (result.stdout + result.stderr).encode())
            runs[name] = dict(command=command, cwd=cwd, environment=sorted(extra), exit_code=result.returncode, log=str(log),
                              log_sha256=sha256(log), summary=result.stdout.strip().splitlines()[-3:])
    browser = S14.checked(['bash', '-c', f'cd {WEB} && npx playwright --version']).strip()
    record = dict(runs=runs, playwright=browser, chromium=str(chrome[-1]), tested_sha256=tested_sources())
    atomic_json(OUT / 'tests.json', record)
    if any(run['exit_code'] for run in runs.values()):
        raise ValueError('required tests failed: ' + ', '.join(n for n, r in runs.items() if r['exit_code']))
    return dict(record, status='passed')


def validate_tests(url):
    tests = S14.read_json(OUT / 'tests.json')
    if set(tests['runs']) != {'serving', 'node', 'browser', 'container_browser', 'lighthouse'}:
        raise ValueError('incomplete test roster')
    for run in tests['runs'].values():
        if run['exit_code']:
            raise ValueError('required tests have not passed')
        verify_file(run['log'], run['log_sha256'])
    if tests['tested_sha256'] != tested_sources():
        raise ValueError('sources changed after tests ran; rerun tests mode')
    if 'passed' not in Path(tests['runs']['container_browser']['log']).read_text() or ' skipped' in Path(
            tests['runs']['container_browser']['log']).read_text():
        raise ValueError('container browser checks must all run')
    lighthouse = S14.read_json(OUT / 'lighthouse.json')
    score = lighthouse['categories']['accessibility']['score']
    if lighthouse['finalDisplayedUrl'] != url + '/' or score is None or score < .95:
        raise ValueError('Lighthouse accessibility on the built container is below 95')
    return tests, score


def documentation_bindings(root=Path('.')):
    docs = {name: root / name for name in ('NOTES.md', 'PHASES.md', 'docs/phase4b-operations.md')}
    if any(not path.is_file() for path in docs.values()):
        raise ValueError('NOTES, PHASES and operations are required before gate publication')
    if 'results/phase4b/report.md' not in docs['NOTES.md'].read_text() or 'Phase 4b' not in docs['PHASES.md'].read_text():
        raise ValueError('NOTES/PHASES lack Phase 4b evidence links')
    text = docs['docs/phase4b-operations.md'].read_text()
    terms = ['100 MiB', '60 seconds', 'zero video retention', '900 seconds', '10 minutes', 'unclipped', 'uncalibrated']
    if any(term not in text for term in terms):
        raise ValueError('operational limits/semantics missing')
    return {name: sha256(path) for name, path in docs.items()}


EVIDENCE = ('build.json', 'image_sources.json', 'decoder_versions.json', 'active_container.json', 'parity.json', 'parity.csv',
            'latency.json', 'latency.raw.json', 'latency.csv', 'cold.json', 'warmup.json', 'tests.json', 'lighthouse.json',
            'decision_amendment.json', 'frontend_checkpoint.json', 'baseline/manifest.json', 'package/build.log')


def collect_demo_evidence(config: dict, url: str) -> dict:
    baseline = capture_baseline(Path('.'), OUT / 'baseline')
    context = S14.research_inputs(config)
    references = S14.reference_for(context)
    build, sources = S14.read_json(OUT / 'build.json'), S14.read_json(OUT / 'image_sources.json')
    observed = observe_image(build['image'])
    observed['fresh_dist'] = fresh_dist()
    verify_image(build, sources, observed)
    verify_build_log(build)
    verify_decoders(S14.read_json(OUT / 'decoder_versions.json'), build)
    active = active_container(build['image_id'], url)
    if S14.read_json(OUT / 'active_container.json') != active:
        raise ValueError('measured container is no longer the active one')
    checkpoint = S14.read_json(OUT / 'frontend_checkpoint.json')
    if checkpoint['design_md_sha256'] != sha256('DESIGN.md'):
        raise ValueError('DESIGN.md changed after the frontend checkpoint')
    parity = validate_demo_parity(context, references, build)
    latency = validate_demo_latency(context, build, active)
    tests, accessibility = validate_tests(url)
    browser = {str(p): sha256(p) for p in sorted((OUT / 'browser').rglob('*')) if p.is_file()}
    if not any(name.endswith('.png') for name in browser):
        raise ValueError('browser screenshots are required')
    return dict(status='passed', fixed_videos=len(context['ids']), image_id=build['image_id'], container_id=active['container_id'],
                baseline_image_id=baseline['image_id'], parity=parity['summary'], latency=latency['summary'],
                cadence=latency['cadence'], js_gzip_total=sum(sources['js_gzip_bytes'].values()), js_gzip_bytes=sources['js_gzip_bytes'],
                accessibility=accessibility, playwright=tests['playwright'],
                evidence_sha256={str(OUT / name): sha256(OUT / name) for name in EVIDENCE}, browser_sha256=browser,
                documentation_sha256=documentation_bindings(), runtime_source_sha256=sources['runtime_source_sha256'],
                dist_sha256=sources['dist_sha256'], build_bindings=build['bindings'], research_bindings=context['bindings'],
                **S14.qualified_provenance())


def report_text(record):
    parity, latency = record['parity'], record['latency']
    return ('# Phase 4b local demo\n\n'
        f"Status: **passed**. Phase 3: **{record['phase3_status']}**; control: **{record['control_status']}**.\n\n"
        f"Fixed videos: {record['fixed_videos']}, each through the CLI `--json`, `/predict` and `/upload`→`/status`. "
        f"Minimum per-timestep cosine: {parity['minimum_cosine']:.12g}. Maximum cache/raw score difference: "
        f"{parity['maximum_cache_difference']:.12g} MOS. Maximum front-door score difference: {parity['maximum_door_difference']:.12g}.\n\n"
        f"Warm upload-to-result wall seconds per ten seconds of video (browser polling contract): maximum "
        f"{latency['maximum_seconds_per_ten_seconds']:.6g}; median {latency['median_seconds_per_ten_seconds']:.6g}; "
        f"budget {latency['budget_seconds_per_ten_seconds']}. Polling overhead beyond upload and internal inference: median "
        f"{latency['median_polling_overhead_seconds']:.3g} s, maximum {latency['maximum_polling_overhead_seconds']:.3g} s.\n\n"
        f"Production JavaScript: {record['js_gzip_total']} bytes gzip (budget 30000, fonts excluded). "
        f"Lighthouse accessibility on the built container: {record['accessibility']:.2f}.\n\n"
        'Cold startup/first task and the excluded warmup are in [cold](cold.json) and [warmup](warmup.json); every warm row with '
        'upload, internal extraction/head/total and polling overhead is in [raw timings](latency.raw.json) and [CSV](latency.csv). '
        'Per-clip raw CLI, API and status responses are under `cli/`, `api/` and `status/`.\n\n'
        f"CPU image: `{record['image_id']}`. Phase 4a anchor image: `{record['baseline_image_id']}` ([baseline](baseline/manifest.json)). "
        '[Build](build.json), [image sources](image_sources.json), [decoder versions](decoder_versions.json), [parity](parity.json), '
        '[tests](tests.json), [Lighthouse](lighthouse.json), [operations](../../docs/phase4b-operations.md), [NOTES](../../NOTES.md).\n\n'
        'Measurements are local loopback on one CPU host under the recorded caps, not hosted latency. Scores are raw, unclipped and '
        'uncalibrated KoNViD MOS-scale ridge outputs. Job metadata is in memory only and video bytes are deleted after each job. '
        'Hosting, final README and publication remain outside this phase.\n')


def write_demo_report(config: dict, url: str) -> dict:
    record = collect_demo_evidence(config, url)
    atomic_write(OUT / 'report.md', report_text(record).encode())
    record['report_sha256'] = sha256(OUT / 'report.md')
    atomic_json(OUT / 'phase_gate.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'parity', 'latency', 'tests', 'report'], required=True)
    parser.add_argument('--image', default='vqa:phase4b')
    parser.add_argument('--url', default='http://127.0.0.1:8001')
    args = parser.parse_args(argv)
    if args.mode == 'baseline':
        record = capture_baseline(Path('.'), OUT / 'baseline')
        print(json.dumps(dict(mode=args.mode, image_id=record['image_id'], files=len(record['files_sha256']))))
        return 0
    invalidate_gate(f'{args.mode} mode in progress')
    try:
        config = read_settings(CONFIG)
        if args.mode == 'parity': result = check_demo_parity(config, args.image, args.url)
        elif args.mode == 'latency': result = measure_demo_latency(config, args.image, args.url)
        elif args.mode == 'tests': result = run_tests(config, args.url)
        else: result = write_demo_report(config, args.url)
        print(json.dumps({'mode': args.mode, 'status': result.get('status', 'saved')}), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        invalidate_gate(f'{args.mode}: {exc}')
        print(json.dumps({'mode': args.mode, 'status': 'failed', 'error': str(exc)}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
