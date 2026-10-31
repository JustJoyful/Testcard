"""One-process upload API, with admission and byte limits before parsing."""
import asyncio
import logging
import mimetypes
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect
from starlette.staticfiles import StaticFiles

from src.predict import PredictionError, load_runtime, predict_video, validate_media
from serving.jobs import JobStore, execute_job, run_thread, wait_owned

logger = logging.getLogger(__name__)


class ReceiptError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class AdmissionBoundary:
    def __init__(self, app, *, state):
        self.app, self.state = app, state

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] not in ('/predict', '/upload') or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        if not self.state.admission.acquire(False):
            return await error_response(scope['path'], 503, 'busy', 'Busy processing another video; retry later.')(scope, receive, send)
        try:
            limits = self.state.runtime.limits
            received, deadline = 0, monotonic() + limits['receipt_seconds']
            async def bounded_receive():
                nonlocal received
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ReceiptError('Upload receipt exceeded 60 seconds.', 408)
                try:
                    message = await asyncio.wait_for(receive(), remaining)
                except TimeoutError as exc:
                    raise ReceiptError('Upload receipt exceeded 60 seconds.', 408) from exc
                if message['type'] == 'http.request':
                    received += len(message.get('body', b''))
                    if received > limits['body_bytes']:
                        raise ReceiptError('Multipart body exceeds 101 MiB.', 413)
                return message
            await self.app(scope, bounded_receive, send)
        finally:
            try:
                # Sending the 202 itself can be cancelled before BackgroundTask starts.
                if worker := scope.get('vqa_job'):
                    await wait_owned(worker)
            finally:
                self.state.admission.release()


def error_response(path, status, code, message):
    error = dict(code=code, message=message) if path == '/upload' else message
    return JSONResponse({'error': error}, status)


async def run_prediction(path, runtime):
    return await run_thread(predict_video, path, runtime)


def create_app(bundle: Path = Path('artifacts/serving/manifest.json'),
               web: Path = Path(__file__).with_name('web') / 'dist') -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = load_runtime(bundle)
        try:
            yield
        finally:
            if app.state.work:
                await wait_owned(asyncio.gather(*app.state.work))

    app = FastAPI(lifespan=lifespan)
    app.state.admission = threading.Lock()
    app.state.jobs = JobStore()
    app.state.work = set()
    app.add_middleware(AdmissionBoundary, state=app.state)

    @app.get('/health')
    async def health():
        return {'status': 'ready', 'verified': app.state.runtime.verified}

    @app.get('/status/{task_id}')
    async def job_status(task_id: str):
        record = app.state.jobs.get(task_id)
        if record is None:
            return JSONResponse({'error': dict(code='task_not_found', message=
                'This result is no longer available. Upload the clip again.')}, 404)
        return record

    @app.post('/upload')
    @app.post('/predict')
    async def predict(request: Request):
        asynchronous = request.url.path == '/upload'
        def error(status, code, message):
            return error_response(request.url.path, status, code, message)
        if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'multipart/form-data':
            return error(422 if asynchronous else 415, 'invalid_upload', 'Upload one multipart file named file.')
        parser = MultiPartParser(request.headers, request.stream(), max_files=1, max_fields=0)
        directory = None
        transferred = False
        try:
            form = await parser.parse()
            items = form.multi_items()
            if len(items) != 1 or items[0][0] != 'file' or not isinstance(items[0][1], UploadFile):
                return error(422, 'invalid_upload', 'Upload exactly one file named file and no other fields.')
            upload = items[0][1]
            directory = tempfile.TemporaryDirectory(prefix='vqa-upload-')
            video = Path(directory.name) / 'input.video'
            size = 0
            with video.open('wb') as target:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > app.state.runtime.limits['file_bytes']:
                        raise ReceiptError('Video exceeds the 100 MiB file limit.', 413)
                    target.write(chunk)
            if not asynchronous:
                return await run_prediction(video, app.state.runtime)
            metadata = await run_thread(validate_media, video, app.state.runtime.limits)
            task_id = app.state.jobs.create(upload.filename or '', metadata['duration_s'])
            worker = asyncio.create_task(execute_job(task_id, video, app.state.runtime, app.state.jobs, directory))
            request.scope['vqa_job'] = worker
            app.state.work.add(worker)
            worker.add_done_callback(app.state.work.discard)
            transferred = True
            return JSONResponse({'task_id': task_id}, 202, background=BackgroundTask(wait_owned, worker))
        except (ReceiptError, PredictionError) as exc:
            code = {408: 'upload_timeout', 413: 'file_too_large', 415: 'unsupported_type'}.get(
                exc.status_code, 'invalid_video')
            return error(exc.status_code, code, str(exc))
        except MultiPartException:
            return error(422, 'invalid_upload', 'Invalid multipart upload; send one file and no other fields.')
        except ClientDisconnect:
            return error(422, 'invalid_upload', 'Upload disconnected before completion.')
        except Exception:
            logger.exception('Video request failed')
            return error(500, 'processing_failed', 'Video processing failed.')
        finally:
            if directory is not None and not transferred:
                directory.cleanup()
            # Pinned Starlette owns this roster; close it also on cancellation and incomplete parses.
            for spool in parser._files_to_close_on_error:
                spool.close()

    # Mounted last so API routes match first; the plain page remains for backend-only development.
    if (web / 'index.html').is_file():
        # The slim image has no /etc/mime.types; CPython 3.11's built-in table lacks fonts.
        for suffix, media in (('.js', 'text/javascript'), ('.woff2', 'font/woff2'), ('.woff', 'font/woff')):
            mimetypes.add_type(media, suffix)
        app.mount('/', StaticFiles(directory=web, html=True))
    else:
        @app.get('/', response_class=HTMLResponse)
        async def index():
            return Path(__file__).with_name('index.html').read_text()

    return app
