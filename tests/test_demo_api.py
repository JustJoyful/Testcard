"""Real ASGI lifetime and trust-boundary coverage for the polling adapter."""
import asyncio
import importlib
import json
import threading
from pathlib import Path

import pytest

from test_serving_api import api, asgi_request, scope, status, multipart, HEADERS

VIDEO = b'\x00\x00\x00\x18ftypisom' + b'\x00' * 20
META = dict(duration_s=19.42, fps=30., width=640, height=480, rotation_applied=0)
NOT_FOUND = {'error': {'code': 'task_not_found', 'message': 'This result is no longer available. Upload the clip again.'}}


@pytest.fixture
def demo(api, monkeypatch):
    module, app, runtime, expected, paths = api
    jobs = importlib.import_module('serving.jobs')
    monkeypatch.setattr(jobs, 'predict_video', module.predict_video)
    monkeypatch.setattr(importlib.import_module('src.predict'), 'probe_video', lambda *args, **kwargs: dict(META))
    from starlette.formparsers import MultiPartParser
    spools, directories = [], []
    original_headers = MultiPartParser.on_headers_finished
    original_directory = module.tempfile.TemporaryDirectory
    def headers(parser):
        original_headers(parser)
        if parser._current_part.file:
            spools.append(parser._current_part.file.file)
    def directory(*args, **kwargs):
        result = original_directory(*args, **kwargs)
        directories.append(Path(result.name))
        return result
    monkeypatch.setattr(MultiPartParser, 'on_headers_finished', headers)
    monkeypatch.setattr(module.tempfile, 'TemporaryDirectory', directory)
    yield api
    assert all(spool.closed for spool in spools)
    assert all(not directory.exists() for directory in directories)
    assert not app.state.admission.locked()


def body(messages):
    return json.loads(b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body'))


def once(payload):
    async def receive():
        return {'type': 'http.request', 'body': payload, 'more_body': False}
    return receive


def test_unknown_id_is_404(demo):
    from fastapi.testclient import TestClient
    with TestClient(demo[1]) as client:
        response = client.get('/status/not-a-task')
        assert response.status_code == 404 and response.json() == NOT_FOUND


@pytest.mark.parametrize('cancel', [False, True], ids=['202_precedes_finish', 'cancel_after_202'])
def test_202_precedes_finish_and_both_posts_share_slot(demo, monkeypatch, cancel):
    module, app, _, expected, paths = demo
    started, release = threading.Event(), threading.Event()
    def predict(path, runtime):
        paths.append(Path(path))
        started.set()
        assert release.wait(5)
        assert Path(path).exists()
        return expected
    monkeypatch.setattr(importlib.import_module('serving.jobs'), 'predict_video', predict)
    async def run():
        async with app.router.lifespan_context(app):
            messages, accepted = [], asyncio.Event()
            async def send(message):
                messages.append(message)
                if message['type'] == 'http.response.body': accepted.set()
            task = asyncio.create_task(app(scope(HEADERS, '/upload'), once(multipart(VIDEO)), send))
            try:
                await asyncio.wait_for(accepted.wait(), 3)
                assert status(messages) == 202 and not task.done()
                task_id = body(messages)['task_id']
                assert await asyncio.to_thread(started.wait, 3)
                if cancel:
                    task.cancel()
                    await asyncio.sleep(.02)
                    task.cancel()  # Repeated ASGI cancellation cannot abandon an owned thread.
                    await asyncio.sleep(.02)
                response = await asgi_request(app, once(b''), path='/status/' + task_id, method='GET')
                assert body(response)['state'] == 'processing' and body(response)['stage'] == 'extract'
                assert status(await asgi_request(app, once(b''), path='/health', method='GET')) == 200
                assert app.state.admission.locked() and paths[0].exists()
                for route in ('/predict', '/upload'):
                    async def unread(): pytest.fail('busy body read')
                    rejected = await asgi_request(app, unread, HEADERS, path=route)
                    assert status(rejected) == 503
                    if route == '/upload': assert body(rejected)['error']['code'] == 'busy'
            finally:
                release.set()
                if cancel:
                    with pytest.raises(asyncio.CancelledError): await task
                else: await task
            result = body(await asgi_request(app, once(b''), path='/status/' + task_id, method='GET'))
            assert result['state'] == 'done' and result['result']['prediction'] == expected
            assert result['result']['clip_seconds'] == 19.42
            assert not app.state.admission.locked() and not paths[0].exists()
            next_response = await asgi_request(app, once(multipart(VIDEO)), HEADERS, path='/upload')
            assert status(next_response) == 202
    asyncio.run(run())


def test_metadata_cancellation_retains_file_and_admission_without_record(demo, monkeypatch):
    module, app, _, _, paths = demo
    started, release = threading.Event(), threading.Event()
    def validate(path, limits):
        paths.append(Path(path)); started.set()
        assert release.wait(5) and Path(path).exists()
        return dict(META)
    monkeypatch.setattr(module, 'validate_media', validate)
    async def run():
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(asgi_request(app, once(multipart(VIDEO)), HEADERS, path='/upload'))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                task.cancel(); await asyncio.sleep(.02)
                assert app.state.admission.locked() and paths[0].exists()
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError): await task
            assert not paths[0].exists() and not app.state.admission.locked()
            assert not app.state.jobs._records
    asyncio.run(run())


@pytest.mark.parametrize('case,http,code', [
    ('empty', 422, 'invalid_video'), ('spoofed', 415, 'unsupported_type'),
    ('metadata', 422, 'invalid_video'), ('fields', 422, 'invalid_upload'),
    ('files', 422, 'invalid_upload'), ('wrong_field', 422, 'invalid_upload'),
    ('file_size', 413, 'file_too_large'), ('not_multipart', 422, 'invalid_upload'),
])
def test_upload_rejects_invalid_media_and_multipart(demo, monkeypatch, case, http, code):
    from fastapi.testclient import TestClient
    module, app, runtime, _, paths = demo
    files = {'file': ('../../clip.mp4', VIDEO, 'application/spoofed')}
    kwargs = dict(files=files)
    if case == 'empty': files['file'] = ('v.mp4', b'', '')
    if case == 'spoofed': files['file'] = ('v.mp4', b'not a container', 'video/mp4')
    if case == 'metadata':
        monkeypatch.setattr(importlib.import_module('src.predict'), 'probe_video', lambda *a, **k: dict(META, duration_s=float('nan')))
    if case == 'fields': kwargs['data'] = {'extra': 'bad'}
    if case == 'files': kwargs['files'] = [('file', ('a.mp4', VIDEO)), ('file', ('b.mp4', VIDEO))]
    if case == 'wrong_field': kwargs['files'] = {'other': ('v.mp4', VIDEO)}
    if case == 'file_size': runtime.limits['file_bytes'] = len(VIDEO) - 1
    if case == 'not_multipart': kwargs = dict(json={'file': 'url'})
    with TestClient(app) as client:
        response = client.post('/upload', **kwargs)
        assert response.status_code == http and response.json()['error']['code'] == code
        assert set(response.json()['error']) == {'code', 'message'}
        assert not app.state.admission.locked() and not app.state.jobs._records
    assert not paths


@pytest.mark.parametrize('case,http,code', [('body', 413, 'file_too_large'), ('timeout', 408, 'upload_timeout'), ('disconnect', 422, 'invalid_upload')])
def test_actual_byte_limit_timeout_and_disconnect(demo, case, http, code):
    _, app, runtime, _, _ = demo
    async def run():
        async with app.router.lifespan_context(app):
            runtime.limits['body_bytes'] = 100
            runtime.limits['receipt_seconds'] = .02
            async def receive():
                if case == 'timeout': await asyncio.Event().wait()
                if case == 'disconnect': return {'type': 'http.disconnect'}
                return {'type': 'http.request', 'body': multipart(VIDEO), 'more_body': False}
            response = await asgi_request(app, receive, HEADERS + [(b'content-length', b'1')], path='/upload')
            assert status(response) == http and body(response)['error']['code'] == code
            assert not app.state.admission.locked() and not app.state.jobs._records
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['decode', 'unexpected'])
def test_terminal_failure_is_sanitized_and_cleans_video(demo, monkeypatch, failure):
    from fastapi.testclient import TestClient
    module, app, _, _, paths = demo
    def fail(path, runtime):
        paths.append(Path(path))
        if failure == 'decode': raise module.PredictionError('Cannot decode video.')
        raise RuntimeError('/private/path traceback')
    monkeypatch.setattr(importlib.import_module('serving.jobs'), 'predict_video', fail)
    with TestClient(app) as client:
        response = client.post('/upload', files={'file': ('v.mp4', VIDEO, '')})
        assert response.status_code == 202
        result = client.get('/status/' + response.json()['task_id']).json()
        assert result['state'] == 'error' and result['stage'] is result['result'] is None
        assert result['error']['code'] == ('invalid_video' if failure == 'decode' else 'processing_failed')
        assert '/private' not in json.dumps(result) and 'Traceback' not in json.dumps(result)
    assert paths and all(not p.exists() for p in paths)
    assert not app.state.admission.locked()


@pytest.mark.parametrize('filename,mime', [('東京.mp4', ''), ('../../clip.mp4', 'text/plain'), (r'C:\private\clip.mp4', 'video/mp4')])
def test_status_uses_checked_bytes_and_private_disk_names(demo, filename, mime):
    from fastapi.testclient import TestClient
    _, app, _, expected, paths = demo
    with TestClient(app) as client:
        response = client.post('/upload', files={'file': (filename, VIDEO, mime)})
        assert response.status_code == 202
        result = client.get('/status/' + response.json()['task_id']).json()['result']
        assert result['prediction'] == expected
        assert result['filename'] in ('東京.mp4', 'clip.mp4')
    assert len(paths) == 1 and paths[0].name == 'input.video' and not paths[0].exists()


def test_restart_loses_jobs(demo):
    from fastapi.testclient import TestClient
    module, app, _, _, _ = demo
    with TestClient(app) as client:
        task_id = client.post('/upload', files={'file': ('v.mp4', VIDEO)}).json()['task_id']
    with TestClient(module.create_app()) as client:
        response = client.get('/status/' + task_id)
        assert response.status_code == 404 and response.json() == NOT_FOUND


def test_cancel_while_sending_202_still_finishes_owned_job(demo, monkeypatch):
    _, app, _, expected, paths = demo
    started, release = threading.Event(), threading.Event()
    def predict(path, runtime):
        paths.append(Path(path)); started.set()
        assert release.wait(5) and Path(path).exists()
        return expected
    monkeypatch.setattr(importlib.import_module('serving.jobs'), 'predict_video', predict)
    async def run():
        async with app.router.lifespan_context(app):
            messages = []
            async def send(message):
                messages.append(message)
                if message['type'] == 'http.response.body':
                    raise asyncio.CancelledError
            task = asyncio.create_task(app(scope(HEADERS, '/upload'), once(multipart(VIDEO)), send))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                assert app.state.admission.locked() and paths[0].exists() and not task.done()
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError): await task
            task_id = body(messages)['task_id']
            result = body(await asgi_request(app, once(b''), path='/status/' + task_id, method='GET'))
            assert result['state'] == 'done' and result['result']['prediction'] == expected
            assert not app.state.admission.locked() and not paths[0].exists()
    asyncio.run(run())


def test_graceful_shutdown_waits_for_accepted_work(demo, monkeypatch):
    _, app, _, expected, paths = demo
    started, release = threading.Event(), threading.Event()
    def predict(path, runtime):
        paths.append(Path(path)); started.set()
        assert release.wait(5) and Path(path).exists()
        return expected
    monkeypatch.setattr(importlib.import_module('serving.jobs'), 'predict_video', predict)
    async def run():
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        request = asyncio.create_task(asgi_request(app, once(multipart(VIDEO)), HEADERS, path='/upload'))
        shutdown = None
        try:
            assert await asyncio.to_thread(started.wait, 3)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(.02)
            assert not shutdown.done() and paths[0].exists() and app.state.admission.locked()
        finally:
            release.set()
            response = await request
            if shutdown is not None: await shutdown
            else: await lifespan.__aexit__(None, None, None)
        assert app.state.jobs.get(body(response)['task_id'])['state'] == 'done'
        assert not paths[0].exists() and not app.state.work and not app.state.admission.locked()
    asyncio.run(run())


def test_static_mount_keeps_api_routes(demo, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    module, fallback, _, expected, _ = demo
    dist = tmp_path / 'dist'
    (dist / 'assets').mkdir(parents=True)
    (dist / 'index.html').write_text('<!doctype html><title>built</title>')
    types = {'app.js': 'text/javascript', 'app.css': 'text/css', 'font.woff2': 'font/woff2', 'font.woff': 'font/woff'}
    for name in types:
        (dist / 'assets' / name).write_bytes(b'asset')
    import mimetypes
    # The slim image has no /etc/mime.types, so only CPython's built-in table applies there.
    monkeypatch.setattr(mimetypes, '_db', mimetypes.MimeTypes())
    app = module.create_app(web=dist)
    with TestClient(app) as client:
        for name, media in types.items():
            response = client.get('/assets/' + name)
            assert response.status_code == 200 and response.headers['content-type'].split(';')[0] == media
        assert client.get('/').text == '<!doctype html><title>built</title>'
        assert client.get('/health').json()['status'] == 'ready'
        assert client.get('/status/unknown').json() == NOT_FOUND
        assert client.post('/predict', files={'file': ('v.mp4', VIDEO)}).json() == expected
        task_id = client.post('/upload', files={'file': ('v.mp4', VIDEO)}).json()['task_id']
        assert client.get('/status/' + task_id).json()['result']['prediction'] == expected
        assert client.get('/assets/missing.js').status_code == 404
    with TestClient(fallback) as client:
        assert 'action="/predict"' in client.get('/').text
