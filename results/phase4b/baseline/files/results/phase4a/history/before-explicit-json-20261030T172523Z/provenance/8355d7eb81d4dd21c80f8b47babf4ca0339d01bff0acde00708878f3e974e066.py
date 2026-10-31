"""Fixed-sample serving references, real Docker front doors and gate-last reporting."""
import argparse
import csv
import importlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.parse import urlparse

import httpx
import numpy as np
import yaml

from src.baselines import predict_ridge
from src.cache import atomic_json, atomic_write, csv_bytes, read_branch, read_manifest
from src.features import sha256
from src.serving_bundle import (MODEL_SHA256, WEIGHTS_SHA256, WEIGHTS_ENUM, PREPROCESSING_SHA256,
    FEATURE_CONFIG_SHA256, read_model, read_settings, verify_file, verify_snapshot, verify_bundle)

OUT = Path('results/phase4a')
CONFIG = Path('configs/serving.yaml')
CAPS = ['--cpus=4', '--memory=4g', '--pids-limit=128', '--read-only', '--tmpfs', '/tmp:rw,nosuid,size=256m']
EXPECTED_VERIFIED = dict(model_sha256=MODEL_SHA256, weights_sha256=WEIGHTS_SHA256,
    weights_enum=WEIGHTS_ENUM, preprocessing_sha256=PREPROCESSING_SHA256, cpu_threads=4)
minimum_cosine = importlib.import_module('scripts.04_parity_check').minimum_cosine
source_video = importlib.import_module('scripts.02_extract_spatial').source_video


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'missing/invalid evidence: {Path(path).name}') from exc


def run(command):
    return subprocess.run(command, capture_output=True, text=True, timeout=300)


def checked(command):
    result = run(command)
    if result.returncode:
        raise ValueError(f'command failed ({result.returncode}): {command[:3]}: {result.stderr[-2000:]}')
    return result.stdout


def validate_ids(ids, expected):
    if ids != expected or len(ids) != len(set(ids)):
        raise ValueError('missing, duplicate or reordered fixed video IDs')


def load_sources(path):
    columns = ('video_id', 'dataset', 'relative_path', 'source_size_bytes', 'source_sha256',
               'duration_s', 'fps', 'width', 'height', 'rotation_applied', 'status')
    with Path(path).open() as handle:
        rows = [{key: row[key] for key in columns} for row in csv.DictReader(handle)]
    if len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate inventory IDs')
    return {r['video_id']: r for r in rows}


def qualified_provenance():
    gate = read_json('results/phase3/phase_gate.json')
    if gate['status'] != 'passed with qualification' or gate['control_status'] != 'failed':
        raise ValueError('changed upstream control qualification')
    return dict(phase3_status=gate['status'], control_status=gate['control_status'])


def research_inputs(config):
    if (config['parity'] != dict(minimum_cosine=.999, cache_score_absolute=.01, cli_api_absolute=1e-6)
            or config['latency'] != dict(warm_repeats=3, seconds_per_ten_seconds=120)):
        raise ValueError('serving thresholds cannot be overridden')
    snapshot = read_json(config['research_snapshot'])
    verify_snapshot(snapshot)
    deployment = read_json(config['deployment'])
    if (config['winner'] != 'resnet50/ridge' or config['model'] != deployment['checkpoint_path']
            or config['model_sha256'] != MODEL_SHA256 or deployment['checkpoint_sha256'] != MODEL_SHA256
            or deployment['seed'] is not None):
        raise ValueError('full-refit selected ridge artifact required')
    verify_file(config['model'], MODEL_SHA256)
    ids = Path(config['sample_ids']).read_text().splitlines()
    expected = yaml.safe_load(Path(config['sample_config']).read_text())['samples']['konvid']
    validate_ids(ids, expected)
    if len(ids) != 20:
        raise ValueError('original twenty-video sample required')
    sources = load_sources('data/manifests/konvid.csv')
    if any(key not in sources or sources[key]['status'] != 'ok' for key in ids):
        raise ValueError('missing or invalid fixed source')
    cache = Path('cache/v2')
    meta = read_json(cache / 'meta.json')
    if (meta['feature_config_sha256'] != FEATURE_CONFIG_SHA256 or
            meta['weights_sha256']['resnet50'] != WEIGHTS_SHA256 or
            meta['lock_sha256'] != snapshot['bindings']['env/requirements.lock'] or
            meta['inventory_sha256']['konvid'] != sha256('data/manifests/konvid.csv')):
        raise ValueError('cache identity mismatch')
    files = [str(CONFIG), config['sample_ids'], config['sample_config'], config['research_snapshot'],
             config['phase3_gate'], config['deployment'], config['feature_config'],
             'cache/v2/meta.json', 'cache/v2/manifest.csv', 'data/manifests/konvid.csv',
             'env/serving.lock', 'results/phase4a/packaging_amendment.json']
    bindings = {name: sha256(name) for name in files}
    return dict(ids=ids, sources=sources, cache=cache, records=read_manifest(cache),
                artifact=read_model(config['model']), bindings=bindings,
                video_config=yaml.safe_load(Path(config['feature_config']).read_text()))


def make_reference(config: dict) -> dict:
    context = research_inputs(config)
    rows = []
    for key in context['ids']:
        source = context['sources'][key]
        array = read_branch(context['cache'], 'konvid', key, 'resnet50', context['records'], source)
        score = float(predict_ridge(context['artifact'], array.mean(0, dtype=np.float64)[None, :])[0])
        rows.append(dict(video_id=key, score=score, T=len(array), duration_seconds=float(source['duration_s']),
                         source_sha256=source['source_sha256'], cache_sha256=sha256(context['cache'] / 'konvid/resnet50' / (key+'.npy'))))
    atomic_write(OUT / 'reference_predictions.csv', csv_bytes(rows, list(rows[0])))
    record = dict(rows=rows, model_sha256=MODEL_SHA256, bindings=context['bindings'],
                  csv_sha256=sha256(OUT / 'reference_predictions.csv'), measurement_tool_sha256=sha256(__file__),
                  **qualified_provenance())
    atomic_json(OUT / 'reference.meta.json', record)
    return record


def validate_front_door(envelope, door):
    key, expected = ('exit_code', 0) if door == 'cli' else ('http_status', 200)
    if envelope.get(key) != expected:
        raise ValueError(f'missing or unsuccessful actual {door} result')
    result = envelope.get('response', {})
    try:
        numbers = [result['score'], result['startup_seconds'], *result['timings_seconds'].values()]
        if (result['verified'] != EXPECTED_VERIFIED or set(result['timings_seconds']) != {'extraction', 'head', 'total'}
                or any(isinstance(x, bool) or not math.isfinite(x) for x in numbers)
                or result['startup_seconds'] < 0 or min(result['timings_seconds'].values()) < 0):
            raise ValueError('invalid numerical response or verified identities')
    except (KeyError, TypeError) as exc:
        raise ValueError('incomplete actual front-door response') from exc
    return result


def passes_parity(cosine, cache_delta, cli_api_delta):
    return (all(math.isfinite(x) for x in (cosine, cache_delta, cli_api_delta)) and
            .999 <= cosine <= 1. and 0 <= cache_delta <= .01 and 0 <= cli_api_delta <= 1e-6)


def compare_features(cached, raw, reference, cli, api):
    if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 2048:
        raise ValueError('raw CLI feature export must be fp32 [T,2048]')
    cosine = minimum_cosine(cached, raw)
    cache_delta = abs(cli['score'] - reference)
    api_cache_delta = abs(api['score'] - reference)
    cli_api_delta = abs(cli['score'] - api['score'])
    passed = passes_parity(cosine, cache_delta, cli_api_delta) and api_cache_delta <= .01
    return dict(status='passed' if passed else 'failed', T=len(raw), minimum_cosine=cosine,
                cached_to_cli_absolute=cache_delta, cached_to_api_absolute=api_cache_delta,
                cli_api_absolute=cli_api_delta)


def image_identity(image):
    return json.loads(checked(['docker', 'image', 'inspect', image]))[0]['Id']


def require_build(image, url):
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('measure request timing over local loopback HTTP')
    build = read_json(OUT / 'build.json')
    if (build['status'] != 'passed' or not build['clean_build'] or build['image_id'] != image_identity(image)
            or build['environment']['cuda'] is not None or build['environment']['cuda_available']
            or build['environment']['uid'] == 0 or build['environment']['python'] != '3.11.15'
            or build['environment']['torch'] != '2.14.1+cpu'
            or build['environment']['torchvision'] != '0.29.1+cpu'):
        raise ValueError('unverified clean CPU image')
    for field, file in [('runtime_lock_sha256', 'env/serving.lock'), ('dockerfile_sha256', 'Dockerfile'),
            ('dockerignore_sha256', '.dockerignore'), ('manifest_sha256', 'artifacts/serving/manifest.json'),
            ('amendment_sha256', 'results/phase4a/packaging_amendment.json'),
            ('decoder_versions_sha256', 'results/phase4a/decoder_versions.json'), ('build_log_sha256', 'results/phase4a/docker_build.log'),
            ('cli_smoke_sha256', 'results/phase4a/docker_cli_smoke.json'), ('api_smoke_sha256', 'results/phase4a/docker_api_smoke.json'),
            ('page_sha256', 'results/phase4a/docker_page.html')]:
        verify_file(file, build[field])
    versions = read_json(OUT / 'decoder_versions.json')
    for binary, pair in versions['commands'].items():
        if (f'{binary} version n9.0.2 ' not in pair['host']['stdout'].splitlines()[0] or
                f'{binary} version n9.0.2-37-gae19868753-20261029 ' not in pair['image']['stdout'].splitlines()[0]):
            raise ValueError('unapproved host/image decoder versions')
    active = read_json(OUT / 'active_container.json')
    container = json.loads(checked(['docker', 'inspect', active['container_id']]))[0]
    host = container['HostConfig']
    if (container['Image'] != build['image_id'] or host['NanoCpus'] != 4000000000 or
            host['Memory'] != 4294967296 or host['PidsLimit'] != 128 or not host['ReadonlyRootfs'] or
            host['Tmpfs'].get('/tmp') != 'rw,nosuid,size=256m' or active['url'] != url):
        raise ValueError('wrong active image/resource caps/URL')
    health = httpx.get(url + '/health', timeout=10)
    if health.status_code != 200 or health.json().get('verified') != EXPECTED_VERIFIED:
        raise ValueError('API is not the verified runtime')
    return build, active


def reference_for(context):
    record = read_json(OUT / 'reference.meta.json')
    if record['bindings'] != context['bindings'] or record['model_sha256'] != MODEL_SHA256:
        raise ValueError('stale or wrong reference identities')
    verify_file(OUT / 'reference_predictions.csv', record['csv_sha256'])
    validate_ids([r['video_id'] for r in record['rows']], context['ids'])
    return {r['video_id']: r for r in record['rows']}


def api_request(client, url, video):
    start = perf_counter()
    with Path(video).open('rb') as handle:
        response = client.post(url + '/predict', files={'file': ('video.mp4', handle, 'video/mp4')})
    finish = perf_counter()
    return dict(http_status=response.status_code, response=response.json(), request_started=start,
                request_finished=finish, request_roundtrip_seconds=finish-start)


def cli_request(image, video, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vqa-export-') as temporary:
        directory = Path(temporary)
        directory.chmod(0o777)
        command = ['docker', 'run', '--rm', '--network', 'none', *CAPS, '--user',
            f'{os.getuid()}:{os.getgid()}', '--mount',
            f'type=bind,src={Path(video).resolve()},dst=/input/video.mp4,readonly', '--mount',
            f'type=bind,src={directory.resolve()},dst=/output', '--entrypoint', 'python', image,
            '-m', 'src.predict', '/input/video.mp4', '--features-out', '/output/features.npy']
        process = run(command)
        record = dict(command=command, exit_code=process.returncode, stderr=process.stderr,
                      response=json.loads(process.stdout) if process.returncode == 0 else None)
        if process.returncode == 0:
            atomic_write(destination, (directory / 'features.npy').read_bytes())
        return record


def check_parity(config: dict, image: str, url: str) -> dict:
    context = research_inputs(config)
    references = reference_for(context)
    build, active = require_build(image, url)
    rows = []
    record = dict(status='failed', rows=rows, bindings=context['bindings'], image_id=build['image_id'],
                  build_sha256=sha256(OUT/'build.json'), reference_sha256=sha256(OUT/'reference.meta.json'),
                  measurement_tool_sha256=sha256(__file__))
    try:
        with httpx.Client(timeout=300) as client:
            for key in context['ids']:
                source = context['sources'][key]
                raw_path = OUT / 'features' / (key + '.npy')
                cli_path, api_path = OUT / 'cli' / (key+'.json'), OUT / 'api' / (key+'.json')
                with source_video(source, context['video_config']['datasets']['konvid']) as video:
                    cli = cli_request(image, video, raw_path)
                    atomic_json(cli_path, cli)
                    cli_result = validate_front_door(cli, 'cli')
                    api = api_request(client, url, video)
                    atomic_json(api_path, api)
                    api_result = validate_front_door(api, 'api')
                cached = read_branch(context['cache'], 'konvid', key, 'resnet50', context['records'], source)
                raw = np.load(raw_path, allow_pickle=False)
                row = dict(video_id=key, source_sha256=source['source_sha256'],
                    cache_sha256=references[key]['cache_sha256'], feature_path=str(raw_path),
                    feature_sha256=sha256(raw_path), cli_path=str(cli_path), cli_sha256=sha256(cli_path),
                    api_path=str(api_path), api_sha256=sha256(api_path),
                    **compare_features(cached, raw, references[key]['score'], cli_result, api_result))
                rows.append(row)
                atomic_json(OUT/'parity.json', record)
                print(json.dumps(dict(video_id=key, **{k: row[k] for k in ['status', 'minimum_cosine', 'cached_to_cli_absolute', 'cli_api_absolute']})), flush=True)
                if row['status'] != 'passed':
                    raise ValueError(f'parity failed for {key}; thresholds unchanged; stop and report')
        validate_ids([r['video_id'] for r in rows], context['ids'])
        record['status'] = 'passed'
    except Exception as exc:
        record['failed_video_id'] = key
        record['error'] = str(exc)
        atomic_json(OUT/'parity.json', record)
        raise
    atomic_json(OUT/'parity.json', record)
    atomic_write(OUT/'parity.csv', csv_bytes(rows, list(rows[0])))
    return record


def summarize_latency(rows, ids):
    if [(r['video_id'], r['repeat']) for r in rows] != [(key, repeat) for key in ids for repeat in (1, 2, 3)]:
        raise ValueError('latency requires three ordered warm repeats for every fixed ID')
    for row in rows:
        if row['role'] != 'warm' or row['duration_seconds'] <= 0:
            raise ValueError('cold/warmup requests cannot enter warm summary')
        validate_front_door(dict(http_status=row['http_status'], response=row['response']), 'api')
        expected = row['request_finished'] - row['request_started']
        if (row['request_roundtrip_seconds'] != expected or expected <= 0 or
                any(not math.isfinite(row[k]) or row[k] < 0 for k in
                    ('duration_seconds', 'extraction', 'head', 'internal_total', 'request_started', 'request_finished'))):
            raise ValueError('invalid actual client request timing')
        timing = row['response']['timings_seconds']
        if (row['extraction'], row['head'], row['internal_total']) != (timing['extraction'], timing['head'], timing['total']):
            raise ValueError('internal times do not match the HTTP response')
    normalized = [r['request_roundtrip_seconds'] * 10 / r['duration_seconds'] for r in rows]
    per_video = {key: dict(maximum=max(v := [n for r,n in zip(rows,normalized) if r['video_id']==key]), median=median(v)) for key in ids}
    return dict(maximum_seconds_per_ten_seconds=max(normalized), median_seconds_per_ten_seconds=median(normalized),
                budget_seconds_per_ten_seconds=120, budget_passed=max(normalized) <= 120, per_video=per_video)


def validate_parity(context, build):
    record = read_json(OUT/'parity.json')
    if record['status'] != 'passed' or record['bindings'] != context['bindings'] or record['image_id'] != build['image_id']:
        raise ValueError('missing/failed/stale Docker parity')
    verify_file(OUT/'build.json', record['build_sha256'])
    verify_file(OUT/'reference.meta.json', record['reference_sha256'])
    references = reference_for(context)
    validate_ids([r['video_id'] for r in record['rows']], context['ids'])
    for row in record['rows']:
        key = row['video_id']
        for kind in ('feature', 'cli', 'api'):
            verify_file(row[kind+'_path'], row[kind+'_sha256'])
        cli = validate_front_door(read_json(row['cli_path']), 'cli')
        api = validate_front_door(read_json(row['api_path']), 'api')
        cached = read_branch(context['cache'], 'konvid', key, 'resnet50', context['records'], context['sources'][key])
        actual = compare_features(cached, np.load(row['feature_path'], allow_pickle=False), references[key]['score'], cli, api)
        if actual['status'] != 'passed' or any(row[k] != v for k,v in actual.items()):
            raise ValueError('per-video parity evidence does not reproduce')
    return record


def measure_latency(config: dict, image: str, url: str) -> dict:
    context = research_inputs(config)
    build, active = require_build(image, url)
    validate_parity(context, build)
    rows = []
    with httpx.Client(timeout=300) as client:
        for index, key in enumerate(context['ids']):
            source = context['sources'][key]
            with source_video(source, context['video_config']['datasets']['konvid']) as video:
                if index == 0:
                    warmup = api_request(client, url, video)
                    validate_front_door(warmup, 'api')
                    atomic_json(OUT/'warmup.json', dict(excluded_from_warm_summary=True, video_id=key, **warmup))
                for repeat in (1, 2, 3):
                    envelope = api_request(client, url, video)
                    result = validate_front_door(envelope, 'api')
                    timing = result['timings_seconds']
                    rows.append(dict(video_id=key, repeat=repeat, role='warm', duration_seconds=float(source['duration_s']),
                        extraction=timing['extraction'], head=timing['head'], internal_total=timing['total'], **envelope))
                    atomic_json(OUT/'latency.raw.json', rows)
                    print(json.dumps(dict(video_id=key, repeat=repeat, roundtrip=envelope['request_roundtrip_seconds'])), flush=True)
    summary = summarize_latency(rows, context['ids'])
    cold = read_json(OUT/'cold.json')
    if cold['image_id'] != build['image_id'] or cold['container_id'] != active['container_id']:
        raise ValueError('cold request does not belong to measured image/container')
    record = dict(status='passed' if summary['budget_passed'] else 'failed', summary=summary,
        bindings=context['bindings'], image_id=build['image_id'], container_id=active['container_id'],
        parity_sha256=sha256(OUT/'parity.json'), raw_sha256=sha256(OUT/'latency.raw.json'),
        cold_sha256=sha256(OUT/'cold.json'), warmup_sha256=sha256(OUT/'warmup.json'),
        measurement_tool_sha256=sha256(__file__), platform=platform.platform(),
        cpu=checked(['lscpu']), resource_caps=build['resource_caps'], cpu_threads=4)
    atomic_json(OUT/'latency.json', record)
    fields=['video_id','repeat','role','duration_seconds','extraction','head','internal_total','request_roundtrip_seconds','request_started','request_finished','http_status']
    atomic_write(OUT/'latency.csv', csv_bytes([{k:r[k] for k in fields} for r in rows], fields))
    if record['status'] != 'passed':
        raise ValueError('warm loopback request budget failed; stop and report')
    return record


def documentation_bindings(root=Path('.')):
    notes, operations = root/'NOTES.md', root/'docs/phase4a-operations.md'
    if not notes.is_file() or not operations.is_file():
        raise ValueError('NOTES and operations are required before gate publication')
    if 'results/phase4a/' not in notes.read_text() or 'Phase 4a' not in notes.read_text():
        raise ValueError('NOTES lacks serving evidence')
    text = operations.read_text()
    if any(term not in text for term in ['100 MiB', '60 seconds', 'zero retention', '1e-6', '120 seconds', 'unclipped']):
        raise ValueError('operational limits/semantics missing')
    return {'NOTES.md': sha256(notes), 'docs/phase4a-operations.md': sha256(operations)}


def collect_evidence(config):
    context = research_inputs(config)
    build = read_json(OUT/'build.json')
    if build['status'] != 'passed' or build['image_id'] != image_identity(build['image']):
        raise ValueError('missing/stale clean image')
    # Recheck active resource and response identity, then rederive every saved comparison.
    require_build(build['image'], read_json(OUT/'active_container.json')['url'])
    parity = validate_parity(context, build)
    latency = read_json(OUT/'latency.json')
    if (latency['status'] != 'passed' or latency['bindings'] != context['bindings'] or latency['image_id'] != build['image_id']
            or latency['cpu_threads'] != 4 or latency['resource_caps'] != build['resource_caps']):
        raise ValueError('failed/stale latency identities')
    for name in ('parity', 'cold', 'warmup'):
        verify_file(OUT/(name+'.json'), latency[name+'_sha256'])
    verify_file(OUT/'latency.raw.json', latency['raw_sha256'])
    summary = summarize_latency(read_json(OUT/'latency.raw.json'), context['ids'])
    if not summary['budget_passed'] or summary != latency['summary']:
        raise ValueError('latency summary/budget does not reproduce')
    tests = read_json(OUT/'tests.json')
    if not tests['runs'] or any(row['exit_code'] != 0 for row in tests['runs'].values()):
        raise ValueError('required tests have not passed')
    for row in tests['runs'].values():
        verify_file(row['log'], row['log_sha256'])
    if not tests.get('serving_source_sha256'):
        raise ValueError('tests must bind current serving sources')
    for path, expected in tests['serving_source_sha256'].items():
        verify_file(path, expected)
    docs = documentation_bindings()
    paths = ['build.json','decoder_versions.json','reference.meta.json','reference_predictions.csv','parity.json','parity.csv',
             'latency.json','latency.raw.json','latency.csv','cold.json','warmup.json','tests.json','packaging_amendment.json']
    bindings = {str(OUT/name): sha256(OUT/name) for name in paths}
    return dict(status='passed', fixed_videos=len(context['ids']), image_id=build['image_id'],
        minimum_cosine=min(r['minimum_cosine'] for r in parity['rows']),
        maximum_cached_score_difference=max(max(r['cached_to_cli_absolute'], r['cached_to_api_absolute']) for r in parity['rows']),
        maximum_cli_api_difference=max(r['cli_api_absolute'] for r in parity['rows']), latency=summary,
        evidence_sha256=bindings, documentation_sha256=docs, research_bindings=context['bindings'],
        serving_source_sha256=tests['serving_source_sha256'], **qualified_provenance())


def report_text(record):
    return ('# Phase 4a serving core\n\nStatus: **passed**. Phase 3: **passed with qualification**; control: **failed**.\n\n'
        f"Fixed videos: {record['fixed_videos']}. Minimum per-timestep cosine: {record['minimum_cosine']:.12g}. "
        f"Maximum cache/raw score difference: {record['maximum_cached_score_difference']:.12g} MOS. "
        f"Maximum CLI/API score difference: {record['maximum_cli_api_difference']:.12g}.\n\n"
        f"Warm loopback request seconds per ten seconds: maximum {record['latency']['maximum_seconds_per_ten_seconds']:.6g}; "
        f"median {record['latency']['median_seconds_per_ten_seconds']:.6g}; budget {record['latency']['budget_seconds_per_ten_seconds']}.\n\n"
        'Cold startup/first request and excluded warmup are separate in [cold](cold.json) and [warmup](warmup.json). '
        'Extraction/head/internal total and actual client round-trip are saved in [raw timings](latency.raw.json) and [CSV](latency.csv).\n\n'
        'The [approved decoder amendment](packaging_amendment.json) uses the same release branch plus 37 maintenance commits; '
        '[host/image full versions](decoder_versions.json) are recorded side by side. All predeclared thresholds are unchanged.\n\n'
        f"CPU image: `{record['image_id']}`. [Build](build.json), [parity](parity.json), [tests](tests.json), "
        '[operations](../../docs/phase4a-operations.md), [NOTES](../../NOTES.md).\n\n'
        'Measurements are local loopback under the recorded caps, not hosted/network-independent latency. '
        'Pathological decode may occupy the single slot until decoder exit; forced process death may leave local temporary files. '
        'Public-internet hardening, presentation, hosting, final README and publication remain outside scope.\n')


def write_report(config: dict) -> dict:
    atomic_json(OUT/'phase_gate.json', {'status': 'failed', 'reason': 'publication in progress'})
    try:
        record = collect_evidence(config)
        text = report_text(record)
        atomic_write(OUT/'report.md', text.encode())
        record['report_sha256'] = sha256(OUT/'report.md')
        atomic_json(OUT/'phase_gate.json', record)
        return record
    except Exception as exc:
        atomic_json(OUT/'phase_gate.json', {'status': 'failed', 'reason': str(exc), **qualified_provenance()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reference','parity','latency','report'], required=True)
    parser.add_argument('--image', default='vqa:phase4a')
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args(argv)
    try:
        config = read_settings(CONFIG)
        if args.mode == 'reference': result = make_reference(config)
        elif args.mode == 'parity': result = check_parity(config, args.image, args.url)
        elif args.mode == 'latency': result = measure_latency(config, args.image, args.url)
        else: result = write_report(config)
        print(json.dumps({'mode': args.mode, 'status': result.get('status', 'saved')}, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        print(json.dumps({'mode': args.mode, 'status': 'failed', 'error': str(exc)}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
