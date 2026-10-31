import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyze, validateFile, COPY, MAX_BYTES, TIMEOUT_MS } from '../src/api.js';

const video = new File([new Uint8Array(4)], 'clip.mp4', { type: 'video/mp4' });
const prediction = { score: 3.6, extra: { nested: 1.0000000001 } };
const done = { task_id: 't', state: 'done', stage: null, result: { task_id: 't', prediction }, error: null };
const pending = (state = 'processing') => ({ task_id: 't', state, stage: state === 'processing' ? 'extract' : null, result: null, error: null });

function harness({ xhr = 'accept', statuses }) {
  let clock = 0;
  const posts = [];
  const gets = [];
  const sleeps = [];
  let inFlight = 0;
  let overlap = false;

  class FakeXHR {
    upload = {};
    open(method, url) { posts.push({ method, url }); }
    send() {
      queueMicrotask(() => {
        if (xhr === 'network') return this.onerror();
        this.upload.onprogress?.({ lengthComputable: true, loaded: 2, total: 4 });
        if (xhr === 'accept') { this.status = 202; this.responseText = '{"task_id":"t"}'; }
        else { this.status = xhr.status; this.responseText = JSON.stringify(xhr.body); }
        this.onload();
      });
    }
    abort() { this.onabort?.(); }
  }

  const fetch = async (url) => {
    if (inFlight) overlap = true;
    inFlight += 1;
    gets.push({ url, at: clock });
    await Promise.resolve();
    inFlight -= 1;
    const next = statuses(gets.length, clock);
    if (next instanceof Error) throw next;
    if (next.httpStatus) return { ok: false, status: next.httpStatus, json: async () => next.body };
    return { ok: true, status: 200, json: async () => next };
  };
  const sleep = async (ms) => { sleeps.push(ms); clock += ms; };
  return { deps: { XHR: FakeXHR, fetch, now: () => clock, sleep }, posts, gets, sleeps, wasOverlapping: () => overlap };
}

test('empty_mime_still_allows_supported_extension', () => {
  assert.equal(validateFile(new File([], 'HOLIDAY.MP4', { type: '' })), null);
  assert.equal(validateFile({ name: 'a.webm', size: MAX_BYTES }), null);
  assert.equal(validateFile({ name: 'a.mov', size: MAX_BYTES + 1 }).message, COPY.file_too_large);
  assert.equal(validateFile({ name: 'a.mkv', size: 1 }).message, COPY.unsupported_type);
  assert.equal(validateFile({ name: 'mp4', size: 1 }).code, 'unsupported_type');
});

test('happy path reports progress, status and returns the exact result', async () => {
  const h = harness({ statuses: (n) => (n < 3 ? pending(n === 1 ? 'queued' : 'processing') : done) });
  const progress = [];
  const seen = [];
  const out = await analyze(video, { onProgress: (p) => progress.push(p), onStatus: (s) => seen.push(s.state) }, h.deps);
  assert.deepEqual(out, done.result);
  assert.deepEqual(progress, [50, 100]);
  assert.deepEqual(seen, ['queued', 'processing']);
  assert.equal(h.posts.length, 1);
});

test('lost_202_never_reposts', async () => {
  const h = harness({ xhr: 'network', statuses: () => done });
  await assert.rejects(analyze(video, {}, h.deps), (e) => e.message === COPY.generic);
  assert.equal(h.posts.length, 1);
  assert.equal(h.gets.length, 0);
});

test('nested upload errors map to DESIGN.md copy or the server wording', async () => {
  for (const [status, code, message, expected] of [
    [413, 'file_too_large', 'x', COPY.file_too_large],
    [415, 'unsupported_type', 'x', COPY.unsupported_type],
    [422, 'invalid_video', 'x', COPY.invalid_video],
    [503, 'busy', 'Busy processing another video; retry later.', 'Busy processing another video; retry later.'],
    [500, 'processing_failed', '/tmp/secret traceback', COPY.generic],
  ]) {
    const h = harness({ xhr: { status, body: { error: { code, message } } }, statuses: () => done });
    await assert.rejects(analyze(video, {}, h.deps), (e) => e.message === expected && e.code === code);
  }
});

test('polling_backoff_and_deadline', async () => {
  // Cadence: 1000 ms through the first 30 s, then 3000 ms; never overlapping.
  const h = harness({ statuses: (n, clock) => (clock >= 45000 ? done : pending()) });
  await analyze(video, {}, h.deps);
  assert.equal(h.wasOverlapping(), false);
  const fast = h.sleeps.filter((ms) => ms === 1000).length;
  assert.equal(fast, 30);
  assert.ok(h.sleeps.slice(30).every((ms) => ms === 3000));

  // Transport failures: three retries at 1000/2000/4000 ms, then error.
  const fail = harness({ statuses: () => new TypeError('offline') });
  await assert.rejects(analyze(video, {}, fail.deps), (e) => e.message === COPY.generic);
  assert.deepEqual(fail.sleeps, [1000, 2000, 4000]);
  assert.equal(fail.gets.length, 4);

  // A success in between resets the retry budget.
  const flaky = harness({ statuses: (n) => (n === 6 ? done : n % 2 ? new TypeError('blip') : pending()) });
  await analyze(video, {}, flaky.deps);

  // 404 stops immediately with the server's retry message.
  const gone = harness({ statuses: () => ({ httpStatus: 404, body: { error: { code: 'task_not_found', message: 'This result is no longer available. Upload the clip again.' } } }) });
  await assert.rejects(analyze(video, {}, gone.deps), (e) => e.code === 'task_not_found' && /Upload the clip again/.test(e.message));
  assert.equal(gone.gets.length, 1);

  // Terminal error stops polling.
  const bad = harness({ statuses: () => ({ ...pending(), state: 'error', error: { code: 'invalid_video', message: 'x' } }) });
  await assert.rejects(analyze(video, {}, bad.deps), (e) => e.message === COPY.invalid_video);
  assert.equal(bad.gets.length, 1);

  // A stuck task errors at the 600 s client deadline.
  const stuck = harness({ statuses: () => pending() });
  await assert.rejects(analyze(video, {}, stuck.deps), (e) => e.message === COPY.timeout);
  assert.ok(stuck.sleeps.reduce((a, b) => a + b, 0) >= TIMEOUT_MS);
});

test('a non-finite score becomes a recoverable error, not NaN on screen', async () => {
  for (const score of [NaN, null, '3.6', undefined]) {
    const h = harness({ statuses: () => ({ ...done, result: { prediction: { score } } }) });
    await assert.rejects(analyze(video, {}, h.deps), (e) => e.message === COPY.generic);
  }
});

test('abort stops client activity without another request', async () => {
  const controller = new AbortController();
  const h = harness({ statuses: () => { controller.abort(); return pending(); } });
  h.deps.sleep = (ms, signal) => (signal.aborted ? Promise.reject(new DOMException('Aborted', 'AbortError')) : Promise.resolve());
  await assert.rejects(analyze(video, { signal: controller.signal }, h.deps), (e) => e.name === 'AbortError');
  assert.equal(h.gets.length, 1);
});
