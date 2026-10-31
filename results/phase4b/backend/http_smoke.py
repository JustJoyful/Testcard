"""Run from the repository root against only the temporary loopback backend."""
import csv
import importlib
import json
import tempfile
import time
from pathlib import Path

import httpx
import yaml

from src.features import sha256

OUT = Path('results/phase4b/backend')
url = 'http://127.0.0.1:8001'
video_id = Path('data/splits/phase0_konvid_ids.txt').read_text().splitlines()[0]
rows = {row['video_id']: row for row in csv.DictReader(Path('data/manifests/konvid.csv').open())}
settings = yaml.safe_load(Path('configs/features_v2.yaml').read_text())['datasets']['konvid']
source_video = importlib.import_module('scripts.02_extract_spatial').source_video
record = dict(url=url, video_id=video_id, validation='host backend checkpoint; not rebuilt-image parity or latency', poll_seconds=.05)
with httpx.Client(timeout=120) as client, source_video(rows[video_id], settings) as video:
    assert client.get(url + '/health').json()['status'] == 'ready'
    before = set(Path(tempfile.gettempdir()).glob('vqa-upload-*'))
    start = time.perf_counter()
    with video.open('rb') as handle:
        response = client.post(url + '/upload', files={'file': ('東京.mp4', handle, '')})
    record['accepted'] = dict(status=response.status_code, body=response.json())
    assert response.status_code == 202
    task_id = response.json()['task_id']
    active = set(Path(tempfile.gettempdir()).glob('vqa-upload-*')) - before
    assert active and all((directory / 'input.video').is_file() for directory in active)
    busy = {}
    for route in ('/predict', '/upload'):
        rejected = client.post(url + route, files={'file': ('busy.mp4', b'body not admitted', 'video/mp4')})
        busy[route] = dict(status=rejected.status_code, body=rejected.json())
        assert rejected.status_code == 503
    record['busy'] = busy
    record['health_during_job'] = client.get(url + '/health').json()
    observed = []
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        response = client.get(url + '/status/' + task_id)
        response.raise_for_status()
        status = response.json()
        if not observed or observed[-1] != [status['state'], status['stage']]:
            observed.append([status['state'], status['stage']])
        if status['state'] in ('done', 'error'):
            break
        time.sleep(.05)
    assert status['state'] == 'done', status
    record['states'] = observed
    record['status'] = status
    record['upload_to_terminal_seconds'] = time.perf_counter() - start
    result = status['result']
    assert result['clip_seconds'] > 0 and result['filename'] == '東京.mp4'
    assert all(not directory.exists() for directory in active)
    record['private_video_cleanup'] = True
    with video.open('rb') as handle:
        response = client.post(url + '/predict', files={'file': (video.name, handle, 'video/mp4')})
    assert response.status_code == 200
    record['predict'] = response.json()
    record['absolute_score_difference'] = abs(result['prediction']['score'] - record['predict']['score'])
    assert record['absolute_score_difference'] <= 1e-6
    assert result['prediction']['verified'] == record['predict']['verified']
    assert set(result['prediction']) == set(record['predict'])
    record['source_sha256'] = {name: sha256(name) for name in
        ('serving/app.py', 'serving/jobs.py', 'src/predict.py', 'src/serving_bundle.py')}
    record['status_check'] = 'passed'
(OUT / 'http_smoke.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: record[key] for key in ('status_check', 'video_id', 'states', 'absolute_score_difference', 'private_video_cleanup')}))
