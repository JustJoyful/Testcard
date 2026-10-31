"""Event-loop-owned, bounded in-memory results and cancellation-safe execution."""
import asyncio
import logging
import unicodedata
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from time import monotonic
from uuid import uuid4

import anyio

from src.predict import PredictionError, predict_video

logger = logging.getLogger(__name__)


class JobStore:
    def __init__(self, *, max_terminal=32, ttl_seconds=900, clock=monotonic):
        self.max_terminal, self.ttl_seconds, self.clock = max_terminal, ttl_seconds, clock
        self._records, self._terminal = {}, {}

    def _prune(self):
        now = self.clock()
        for task_id, completed in list(self._terminal.items()):
            if now - completed >= self.ttl_seconds:
                del self._terminal[task_id]
                del self._records[task_id]
        while len(self._terminal) > self.max_terminal:
            task_id = next(iter(self._terminal))
            del self._terminal[task_id]
            del self._records[task_id]

    def create(self, filename: str, clip_seconds: float) -> str:
        self._prune()
        task_id = uuid4().hex
        label = filename.replace('\\', '/').rsplit('/', 1)[-1]
        label = ''.join(c for c in label if not unicodedata.category(c).startswith('C'))[:255]
        self._records[task_id] = dict(
            status=dict(task_id=task_id, state='queued', stage=None, result=None, error=None),
            metadata=dict(task_id=task_id, filename=label, clip_seconds=clip_seconds,
                          created_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')))
        return task_id

    def processing(self, task_id: str) -> None:
        self._records[task_id]['status'].update(state='processing', stage='extract')

    def finish(self, task_id: str, prediction: dict) -> None:
        record = self._records[task_id]
        record['status'].update(state='done', stage=None,
                                result=dict(**record['metadata'], prediction=deepcopy(prediction)), error=None)
        self._terminal[task_id] = self.clock()
        self._prune()

    def fail(self, task_id: str, code: str, message: str) -> None:
        self._records[task_id]['status'].update(state='error', stage=None, result=None,
                                              error=dict(code=code, message=message))
        self._terminal[task_id] = self.clock()
        self._prune()

    def get(self, task_id: str) -> dict | None:
        self._prune()
        record = self._records.get(task_id)
        return deepcopy(record['status']) if record else None


async def wait_owned(worker):
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Callers may leave; owned work must finish before its files or slot can go away.
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


async def run_thread(function, *args):
    worker = asyncio.create_task(anyio.to_thread.run_sync(partial(function, *args), abandon_on_cancel=False))
    return await wait_owned(worker)


async def execute_job(task_id, video, runtime, store, directory):
    try:
        store.processing(task_id)
        prediction = await run_thread(predict_video, video, runtime)
        store.finish(task_id, prediction)
    except PredictionError:
        store.fail(task_id, 'invalid_video', "We couldn't read that video. Try re-exporting it as MP4.")
    except Exception:
        logger.exception('Video job failed')
        store.fail(task_id, 'processing_failed', 'Video processing failed.')
    finally:
        directory.cleanup()
