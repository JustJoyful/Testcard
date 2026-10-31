"""One-process upload API, with admission and byte limits before parsing."""
import asyncio
import logging
import tempfile
import threading
from contextlib import asynccontextmanager, suppress
from functools import partial
from pathlib import Path
from time import monotonic

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from src.predict import PredictionError, load_runtime, predict_video

logger = logging.getLogger(__name__)


class ReceiptError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class AdmissionBoundary:
    def __init__(self, app, *, state):
        self.app, self.state = app, state

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/predict' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        if not self.state.admission.acquire(False):
            return await JSONResponse({'error': 'Busy processing another video; retry later.'}, 503)(scope, receive, send)
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
            self.state.admission.release()


async def run_prediction(path, runtime):
    worker = asyncio.create_task(anyio.to_thread.run_sync(partial(predict_video, path, runtime),
                                                        abandon_on_cancel=False))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # ASGI task cancellation must not outlive the file or admission slot used by its thread.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        with suppress(Exception):
            worker.result()
        raise


def create_app(bundle: Path = Path('artifacts/serving/manifest.json')) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = load_runtime(bundle)
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.admission = threading.Lock()
    app.add_middleware(AdmissionBoundary, state=app.state)

    @app.get('/', response_class=HTMLResponse)
    async def index():
        return Path(__file__).with_name('index.html').read_text()

    @app.get('/health')
    async def health():
        return {'status': 'ready', 'verified': app.state.runtime.verified}

    @app.post('/predict')
    async def predict(request: Request):
        if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'multipart/form-data':
            return JSONResponse({'error': 'Upload one multipart file named file.'}, 415)
        parser = MultiPartParser(request.headers, request.stream(), max_files=1, max_fields=0)
        try:
            form = await parser.parse()
            items = form.multi_items()
            if len(items) != 1 or items[0][0] != 'file' or not isinstance(items[0][1], UploadFile):
                return JSONResponse({'error': 'Upload exactly one file named file and no other fields.'}, 422)
            upload = items[0][1]
            with tempfile.TemporaryDirectory(prefix='vqa-upload-') as directory:
                video = Path(directory) / 'input.video'
                size = 0
                with video.open('wb') as target:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > app.state.runtime.limits['file_bytes']:
                            raise ReceiptError('Video exceeds the 100 MiB file limit.', 413)
                        target.write(chunk)
                return await run_prediction(video, app.state.runtime)
        except (ReceiptError, PredictionError) as exc:
            return JSONResponse({'error': str(exc)}, exc.status_code)
        except MultiPartException:
            return JSONResponse({'error': 'Invalid multipart upload; send one file and no other fields.'}, 422)
        except ClientDisconnect:
            return JSONResponse({'error': 'Upload disconnected before completion.'}, 422)
        except Exception:
            logger.exception('Video request failed')
            return JSONResponse({'error': 'Video processing failed.'}, 500)
        finally:
            # Pinned Starlette owns this roster; close it also on cancellation and incomplete parses.
            for spool in parser._files_to_close_on_error:
                spool.close()

    return app
