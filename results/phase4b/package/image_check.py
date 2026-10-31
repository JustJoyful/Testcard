"""Task 7 packaging check of the built image; run from the repository root. Not Task 8 parity or latency evidence."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import httpx

OUT = Path('results/phase4b/package')
IMAGE, NAME, URL = 'vqa:phase4b', 'vqa-phase4b-check', 'http://127.0.0.1:8001'
CAPS = ['--cpus=4', '--memory=4g', '--pids-limit=128', '--read-only', '--tmpfs', '/tmp:rw,nosuid,size=256m']
VIDEO = Path('data/raw/konvid') / (Path('data/splits/phase0_konvid_ids.txt').read_text().split()[0] + '.mp4')
INSIDE = r'''
import hashlib, json, shutil, os, torch
from pathlib import Path
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for root in ('/app/src', '/app/serving')
         for p in sorted(Path(root).rglob('*')) if p.is_file()}
print(json.dumps(dict(uid=os.getuid(), gid=os.getgid(), torch=torch.__version__, cuda=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), node=shutil.which('node'), npm=shutil.which('npm'),
                      files_sha256=files)))
'''


def docker(*args):
    return subprocess.run(['docker', *args], check=True, capture_output=True, text=True).stdout


record = dict(image=IMAGE, url=URL, video=VIDEO.name, scope='Task 7 packaging smoke; formal parity/latency is Task 8')
record['image_id'] = docker('image', 'inspect', IMAGE, '--format', '{{.Id}}').strip()
config = json.loads(docker('image', 'inspect', IMAGE, '--format', '{{json .Config}}'))
record['config'] = dict(user=config['User'], cmd=config['Cmd'], exposed=sorted(config['ExposedPorts']))
inside = json.loads(docker('run', '--rm', '--network', 'none', *CAPS, '--entrypoint', 'python', IMAGE, '-c', INSIDE))
assert inside['uid'] == 10001 and inside['cuda'] is None and not inside['cuda_available'], inside
assert inside['node'] is None and inside['npm'] is None, 'Node must not ship in the runtime image'
record['inside'] = inside
prefix = '/app/serving/web/dist/'
dist = {name[len(prefix):]: digest for name, digest in inside['files_sha256'].items() if name.startswith(prefix)}
assert 'index.html' in dist and not any('mock' in name for name in dist)

subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True)
docker('run', '-d', '--name', NAME, *CAPS, '-p', '127.0.0.1:8001:8000', IMAGE)
host = json.loads(docker('inspect', NAME, '--format', '{{json .HostConfig}}'))
record['caps'] = dict(nano_cpus=host['NanoCpus'], memory=host['Memory'], pids_limit=host['PidsLimit'],
                      read_only=host['ReadonlyRootfs'], tmpfs=host['Tmpfs'], ports=host['PortBindings'])
assert host['NanoCpus'] == 4_000_000_000 and host['Memory'] == 4 * 1024 ** 3
assert host['PidsLimit'] == 128 and host['ReadonlyRootfs']
assert host['PortBindings'] == {'8000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '8001'}]}

with httpx.Client(base_url=URL, timeout=300) as client:
    start = time.monotonic()
    while True:
        try:
            if client.get('/health').json()['status'] == 'ready':
                break
        except httpx.TransportError:
            pass
        assert time.monotonic() - start < 180, 'health never ready'
        time.sleep(1)
    record['ready_seconds'] = round(time.monotonic() - start, 1)
    page = client.get('/')
    assert page.status_code == 200 and 'id="set"' in page.text and '/src/main.js' not in page.text
    assets = {}
    for ref in re.findall(r'(?:src|href)="(/assets/[^"]+)"', page.text):
        response = client.get(ref)
        assert response.status_code == 200, ref
        assets[ref] = response.headers['content-type']
    css = next(ref for ref in assets if ref.endswith('.css'))
    for font in re.findall(r'url\((/assets/[^)]+\.woff2)\)', client.get(css).text):
        response = client.get(font)
        assert response.status_code == 200 and response.headers['content-type'] == 'font/woff2', font
        assets[font] = response.headers['content-type']
    kinds = {value.split(';')[0] for value in assets.values()}
    assert {'text/javascript', 'text/css', 'font/woff2'} <= kinds, kinds
    record['served_assets'] = assets

    predict = client.post('/predict', files={'file': (VIDEO.name, VIDEO.read_bytes(), 'video/mp4')})
    assert predict.status_code == 200, predict.text
    accepted = client.post('/upload', files={'file': (VIDEO.name, VIDEO.read_bytes(), 'video/mp4')})
    assert accepted.status_code == 202, accepted.text
    task_id = accepted.json()['task_id']
    while (status := client.get(f'/status/{task_id}').json())['state'] not in ('done', 'error'):
        time.sleep(1)
    assert status['state'] == 'done', status
cli = docker('run', '--rm', '--network', 'none', *CAPS, '--user', f'{os.getuid()}:{os.getgid()}',
             '--entrypoint', 'python', '--mount', f'type=bind,src={VIDEO.resolve()},dst=/input/video.mp4,readonly',
             IMAGE, '-m', 'src.predict', '/input/video.mp4', '--json')
doors = dict(predict=predict.json(), status=status['result']['prediction'], cli=json.loads(cli))
assert set(doors['predict']) == set(doors['status']) == set(doors['cli'])
scores = {door: body['score'] for door, body in doors.items()}
assert max(scores.values()) - min(scores.values()) <= 1e-6, scores
record.update(front_doors=doors, status_outer={k: v for k, v in status['result'].items() if k != 'prediction'},
              scores=scores, dist_sha256=dist)
(OUT / 'image_check.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(dict(image_id=record['image_id'], scores=scores, ready_seconds=record['ready_seconds'],
                      dist_files=len(dist), caps=record['caps'])))
