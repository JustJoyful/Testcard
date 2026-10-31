"""Bounded metadata retention, lossless prediction transport and safe display labels."""
import importlib
from datetime import datetime
from uuid import UUID

import pytest


def test_job_store_terminal_bound_and_ttl():
    now = [0.]
    store = importlib.import_module('serving.jobs').JobStore(clock=lambda: now[0])
    active = store.create('active.mp4', 1.)
    store.processing(active)
    ids = []
    for index in range(33):
        task_id = store.create('clip.mp4', 1.)
        ids.append(task_id)
        store.finish(task_id, {'score': index})
    assert store.get(ids[0]) is None
    assert all(store.get(task_id)['state'] == 'done' for task_id in ids[1:])
    now[0] = 899.999
    assert store.get(ids[-1])['state'] == 'done'
    now[0] = 900
    assert all(store.get(task_id) is None for task_id in ids)
    assert store.get(active) == dict(task_id=active, state='processing', stage='extract', result=None, error=None)
    failed = store.create('bad.mp4', 2.)
    store.fail(failed, 'invalid_video', 'Cannot read video.')
    assert store.get(failed)['error'] == dict(code='invalid_video', message='Cannot read video.')
    now[0] = 1800
    assert store.get(failed) is None
    assert store.get(active)['state'] == 'processing'


@pytest.mark.parametrize('score', [-.25, 5.25])
@pytest.mark.parametrize('filename,label', [('../../clip.mp4', 'clip.mp4'),
    (r'C:\private\東京.mp4', '東京.mp4'), ('bad\x00\x1f\x7f\u0085.mp4', 'bad.mp4'),
    ('a' * 300 + '.mp4', 'a' * 255), ('<img src=x onerror=alert(1)>.mp4', '<img src=x onerror=alert(1)>.mp4')])
def test_status_preserves_prediction_and_sanitizes_filename(score, filename, label):
    store = importlib.import_module('serving.jobs').JobStore()
    task_id = store.create(filename, 19.42)
    assert UUID(task_id).version == 4 and len(task_id) == 32
    assert store.get(task_id) == dict(task_id=task_id, state='queued', stage=None, result=None, error=None)
    prediction = dict(score=score, timings_seconds={'head': .0000012345}, unknown={'full': [1, 2, 'text']})
    store.processing(task_id)
    store.finish(task_id, prediction)
    response = store.get(task_id)
    assert response['state'] == 'done' and response['stage'] is response['error'] is None
    result = response['result']
    assert set(result) == {'task_id', 'filename', 'clip_seconds', 'created_at', 'prediction'}
    assert result['prediction'] == prediction
    assert result['filename'] == label and result['clip_seconds'] == 19.42 and result['task_id'] == task_id
    assert result['created_at'].endswith('Z')
    assert datetime.fromisoformat(result['created_at'].replace('Z', '+00:00')).utcoffset().total_seconds() == 0
