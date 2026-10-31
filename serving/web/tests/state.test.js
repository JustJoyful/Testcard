import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { dispatch, getState, currentGeneration, subscribe, _resetForTests } from '../src/state.js';

const fileA = { name: 'a.mp4', size: 10 };
const fileB = { name: 'b.mp4', size: 10 };
const result = (id) => ({ task_id: id, prediction: { score: 3.6 } });

beforeEach(_resetForTests);

test('state has exactly the DESIGN.md fields', () => {
  assert.deepEqual(Object.keys(getState()).sort(), ['error', 'file', 'name', 'result', 'stage', 'taskId', 'uploadPct']);
});

test('stale_generation_cannot_replace_new_selection', () => {
  dispatch({ type: 'select', file: fileA });
  const genA = currentGeneration();
  dispatch({ type: 'upload-start', gen: genA });
  dispatch({ type: 'reset' });
  dispatch({ type: 'select', file: fileB });
  const genB = currentGeneration();
  dispatch({ type: 'upload-start', gen: genB });

  let selections = 0;
  subscribe((s) => { if (s.name === 'inserting') selections += 1; });
  dispatch({ type: 'progress', gen: genA, pct: 90 });
  dispatch({ type: 'status', gen: genA, status: { state: 'processing', stage: 'extract', task_id: 'A' } });
  dispatch({ type: 'done', gen: genA, result: result('A') });
  dispatch({ type: 'fail', gen: genA, error: { message: 'x' } });

  const s = getState();
  assert.equal(s.name, 'uploading');
  assert.equal(s.file, fileB);
  assert.equal(s.uploadPct, 0);
  assert.equal(s.result, null);
  assert.equal(selections, 0);

  dispatch({ type: 'done', gen: genB, result: result('B') });
  assert.equal(getState().result.task_id, 'B');
});

test('a second selection while busy is ignored, so no second submission starts', () => {
  dispatch({ type: 'select', file: fileA });
  const gen = currentGeneration();
  dispatch({ type: 'select', file: fileB });
  assert.equal(currentGeneration(), gen);
  assert.equal(getState().file, fileA);
});

test('status maps queued and processing with unknown stage kept as null', () => {
  dispatch({ type: 'select', file: fileA });
  const gen = currentGeneration();
  dispatch({ type: 'upload-start', gen });
  dispatch({ type: 'status', gen, status: { state: 'queued', stage: null, task_id: 't1' } });
  assert.equal(getState().name, 'queued');
  assert.equal(getState().taskId, 't1');
  dispatch({ type: 'status', gen, status: { state: 'processing' } });
  assert.equal(getState().name, 'processing');
  assert.equal(getState().stage, null);
});

test('invalid leaves on the next user action and drag does not interrupt a request', () => {
  dispatch({ type: 'invalid', error: { message: 'bad' } });
  assert.equal(getState().name, 'invalid');
  dispatch({ type: 'dragenter' });
  assert.equal(getState().name, 'dragover');
  assert.equal(getState().error, null);
  dispatch({ type: 'dragleave' });
  assert.equal(getState().name, 'idle');

  dispatch({ type: 'select', file: fileA });
  dispatch({ type: 'dragenter' });
  assert.equal(getState().name, 'inserting');
});

test('reset from done returns to a blank idle and invalidates the old generation', () => {
  dispatch({ type: 'select', file: fileA });
  const gen = currentGeneration();
  dispatch({ type: 'done', gen, result: result('A') });
  assert.equal(getState().name, 'done');
  dispatch({ type: 'reset' });
  dispatch({ type: 'done', gen, result: result('A') });
  assert.equal(getState().name, 'idle');
  assert.equal(getState().result, null);
});

test('standby: health failure parks only an idle screen; a tape offered offline never starts a request', () => {
  dispatch({ type: 'standby' });
  assert.equal(getState().name, 'standby');
  dispatch({ type: 'dragenter' });
  assert.equal(getState().name, 'standby');
  const gen = currentGeneration();
  dispatch({ type: 'offline' });
  assert.equal(getState().name, 'error');
  assert.equal(getState().error.code, 'offline');
  assert.equal(currentGeneration(), gen);
  dispatch({ type: 'online' });
  assert.equal(getState().name, 'idle');

  dispatch({ type: 'select', file: fileA });
  dispatch({ type: 'standby' });
  dispatch({ type: 'offline' });
  assert.equal(getState().name, 'inserting');
});
