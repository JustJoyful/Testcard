"""ASGI admission, receipt bounds, cancellation and upload lifetime contracts."""
import asyncio
import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def api(monkeypatch, tmp_path):
    module = importlib.import_module('serving.app')
    from src.serving_bundle import LIMITS
    runtime = SimpleNamespace(limits=dict(LIMITS))
    result = dict(score=5.25, startup_seconds=.1, timings_seconds=dict(extraction=.2, head=.01, total=.21),
                  verified=dict(model_sha256='a' * 64, weights_sha256='b' * 64,
                                weights_enum='ResNet50_Weights.IMAGENET1K_V2',
                                preprocessing_sha256='c' * 64, cpu_threads=4))
    runtime.verified = result['verified']
    seen = []
    def predict(path, loaded):
        assert loaded is runtime
        assert Path(path).is_file()
        seen.append(Path(path))
        return dict(result)
    monkeypatch.setattr(module, 'load_runtime', lambda bundle: runtime)
    monkeypatch.setattr(module, 'predict_video', predict)
    # No built frontend: the plain page fallback used for backend development.
    app = module.create_app(web=tmp_path / 'no-dist')
    return module, app, runtime, result, seen


@pytest.mark.parametrize('score', [-.25, 5.25])
def test_upload_calls_shared_function_and_returns_same_contract(api, score):
    from fastapi.testclient import TestClient
    _, app, _, expected, seen = api
    expected['score'] = score
    with TestClient(app) as client:
        response = client.post('/predict', files={'file': ('video.mp4', b'video', 'video/mp4')})
        assert response.status_code == 200 and response.json() == expected
    assert len(seen) == 1 and not seen[0].exists()


def test_form_posts_one_file_to_predict(api):
    from fastapi.testclient import TestClient
    _, app, _, _, _ = api
    with TestClient(app) as client:
        page = client.get('/').text
    assert 'action="/predict"' in page and 'multipart/form-data' in page
    assert 'name="file"' in page and 'textContent' in page


def test_form_explains_unclipped_konvid_mos(api):
    from fastapi.testclient import TestClient
    _, app, _, _, _ = api
    with TestClient(app) as client:
        page = client.get('/').text
    assert 'Predicted MOS (KoNViD 1–5 scale)' in page
    assert 'Predictions are unclipped and may fall slightly outside 1–5.' in page


def scope(headers=(), path='/predict', method='POST'):
    return dict(type='http', asgi={'version': '3.0'}, method=method, path=path,
                raw_path=path.encode(), root_path='', scheme='http', query_string=b'',
                headers=list(headers), client=('127.0.0.1', 1), server=('test', 80), http_version='1.1')


async def asgi_request(app, receive, headers=(), path='/predict', method='POST'):
    messages = []
    async def send(message): messages.append(message)
    await app(scope(headers, path, method), receive, send)
    return messages


def status(messages):
    return next(m['status'] for m in messages if m['type'] == 'http.response.start')


def multipart(data=b'video', name='file', filename='video.mp4'):
    return (f'--x\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            'Content-Type: video/mp4\r\n\r\n').encode() + data + b'\r\n--x--\r\n'


HEADERS = [(b'content-type', b'multipart/form-data; boundary=x')]


def test_busy_rejected_before_receive(api):
    _, app, _, _, seen = api
    async def run():
        async with app.router.lifespan_context(app):
            assert app.state.admission.acquire(False)
            async def receive(): pytest.fail('busy body was read')
            try:
                assert status(await asgi_request(app, receive, HEADERS)) == 503
            finally:
                app.state.admission.release()
    asyncio.run(run())
    assert seen == []


@pytest.mark.parametrize('length', [None, b'1'])
def test_chunked_or_false_content_length_hits_actual_body_limit(api, length):
    _, app, runtime, _, seen = api
    runtime.limits['body_bytes'] = 200
    async def run():
        async with app.router.lifespan_context(app):
            chunks = iter([multipart(b'v' * 30)[:130], b'z' * 100])
            async def receive():
                return {'type': 'http.request', 'body': next(chunks), 'more_body': True}
            headers = HEADERS + ([(b'content-length', length)] if length else [])
            assert status(await asgi_request(app, receive, headers)) == 413
            assert not app.state.admission.locked()
    asyncio.run(run())
    assert not seen


def test_one_file_no_fields_and_receipt_timeout(api):
    from fastapi.testclient import TestClient
    _, app, runtime, _, seen = api
    with TestClient(app) as client:
        for kwargs in [dict(files=[('file', ('a.mp4', b'a')), ('file', ('b.mp4', b'b'))]),
                       dict(data={'extra': 'no'}, files={'file': ('a.mp4', b'a')}),
                       dict(files={'other': ('a.mp4', b'a')})]:
            assert client.post('/predict', **kwargs).status_code == 422
    runtime.limits['receipt_seconds'] = .02
    async def run():
        async with app.router.lifespan_context(app):
            async def receive(): await asyncio.Event().wait()
            assert status(await asgi_request(app, receive, HEADERS)) == 408
            assert not app.state.admission.locked()
    asyncio.run(run())
    assert not seen


def test_disconnect_keeps_slot_until_worker_finishes(api, monkeypatch):
    module, app, _, expected, _ = api
    started, release = threading.Event(), threading.Event()
    paths = []
    def predict(path, runtime):
        paths.append(Path(path))
        started.set()
        assert release.wait(5)
        assert Path(path).exists()
        return expected
    monkeypatch.setattr(module, 'predict_video', predict)
    async def run():
        async with app.router.lifespan_context(app):
            sent = False
            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {'type': 'http.request', 'body': multipart(), 'more_body': False}
                return {'type': 'http.disconnect'}
            task = asyncio.create_task(asgi_request(app, receive, HEADERS))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                task.cancel()
                await asyncio.sleep(.02)
                assert app.state.admission.locked() and paths[0].exists()
                async def unread(): pytest.fail('second body read')
                assert status(await asgi_request(app, unread, HEADERS)) == 503
            finally:
                release.set()
                with pytest.raises(asyncio.CancelledError): await task
            assert not app.state.admission.locked() and not paths[0].exists()
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['success', 'decode', 'worker', 'parser'])
def test_cleanup_on_success_decode_error_parser_error_and_worker_failure(api, monkeypatch, failure):
    from fastapi.testclient import TestClient
    from starlette.formparsers import MultiPartParser
    module, app, _, expected, seen = api
    spools = []
    original = MultiPartParser.on_headers_finished
    def headers(self):
        original(self)
        if self._current_part.file: spools.append(self._current_part.file.file)
    monkeypatch.setattr(MultiPartParser, 'on_headers_finished', headers)
    def predict(path, runtime):
        seen.append(Path(path))
        if failure == 'decode': raise module.PredictionError('Bad video metadata.')
        if failure == 'worker': raise RuntimeError('/private/model traceback must not escape')
        return expected
    monkeypatch.setattr(module, 'predict_video', predict)
    with TestClient(app) as client:
        files = [('file', ('a.mp4', b'a' * 1100000))]
        if failure == 'parser': files.append(('file', ('b.mp4', b'b')))
        response = client.post('/predict', files=files)
        assert response.status_code == {'success': 200, 'decode': 422, 'worker': 500, 'parser': 422}[failure]
        assert '/private' not in response.text and 'Traceback' not in response.text
    assert spools and all(f.closed for f in spools)
    assert all(not p.exists() for p in seen)


def test_health_responsive_during_inference(api, monkeypatch):
    from fastapi.testclient import TestClient
    module, app, _, expected, _ = api
    started, release = threading.Event(), threading.Event()
    def predict(*args):
        started.set()
        assert release.wait(5)
        return expected
    monkeypatch.setattr(module, 'predict_video', predict)
    responses = []
    with TestClient(app) as client:
        thread = threading.Thread(target=lambda: responses.append(client.post('/predict', files={'file': ('v.mp4', b'v')})))
        thread.start()
        try:
            assert started.wait(3)
            assert client.get('/health').json()['status'] == 'ready'
        finally:
            release.set()
            thread.join(5)
    assert responses[0].status_code == 200


def test_untrusted_filename_and_useful_errors(api):
    from fastapi.testclient import TestClient
    _, app, _, _, seen = api
    with TestClient(app) as client:
        assert client.post('/predict', json={'file': 'https://example.org/secret'}).status_code == 415
        response = client.post('/predict', files={'file': ('../../untrusted.mp4', b'v')})
        assert response.status_code == 200
    assert seen and 'untrusted' not in str(seen[0])
