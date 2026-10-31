// Transport only: one XHR upload, then status polling. Never touches the DOM.

export const MAX_BYTES = 104857600; // must equal the API's file limit (docs/phase4b-contract.md)
export const EXTENSIONS = ['.mp4', '.mov', '.webm'];
export const TIMEOUT_MS = 10 * 60 * 1000;
export const FAST_POLL_MS = 1000;
export const SLOW_POLL_MS = 3000;
export const FAST_POLL_WINDOW_MS = 30 * 1000;
export const RETRY_DELAYS_MS = [1000, 2000, 4000];
export const STATUS_REQUEST_TIMEOUT_MS = 15000;

export const COPY = {
  unsupported_type: "That file type isn't supported. Use MP4, MOV, or WEBM.",
  file_too_large: 'That file is over 100 MiB. Trim it or choose a shorter clip.',
  invalid_video: "We couldn't read that video. Try re-exporting it as MP4.",
  generic: 'Something went wrong while scoring. Try again.',
  timeout: 'This is taking longer than expected. Try again later.',
};

// Codes whose server message is already user-facing and has no DESIGN.md copy of its own.
const SERVER_WORDED = new Set(['busy', 'upload_timeout', 'invalid_upload', 'task_not_found']);

export class ApiError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function errorFor(code, serverMessage) {
  if (COPY[code]) return new ApiError(code, COPY[code]);
  if (SERVER_WORDED.has(code) && typeof serverMessage === 'string' && serverMessage) {
    return new ApiError(code, serverMessage);
  }
  return new ApiError(code || 'processing_failed', COPY.generic);
}

export function validateFile(file) {
  const name = String(file?.name ?? '').toLowerCase();
  // MIME is advisory (often empty); the server checks the actual container bytes.
  if (!EXTENSIONS.some((ext) => name.endsWith(ext))) return { code: 'unsupported_type', message: COPY.unsupported_type };
  if (file.size > MAX_BYTES) return { code: 'file_too_large', message: COPY.file_too_large };
  return null;
}

// Monotonic start of the current request; the VCR counter reads elapsed time from it.
let startedAt = null;
export const requestStartedAt = () => startedAt;
export const markRequestStart = (now = performance.now()) => { startedAt = now; };

const abortError = () => new DOMException('Aborted', 'AbortError');

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(abortError());
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

const defaults = {
  base: '',
  XHR: globalThis.XMLHttpRequest,
  fetch: (...args) => globalThis.fetch(...args),
  now: () => performance.now(),
  sleep,
};

function upload(file, { signal, onProgress }, deps) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(abortError());
    const xhr = new deps.XHR();
    xhr.open('POST', `${deps.base}/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) onProgress?.((e.loaded / e.total) * 100);
    };
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // non-JSON body: reported as a generic failure below
      }
      if (xhr.status === 202 && typeof body?.task_id === 'string') return resolve(body.task_id);
      reject(errorFor(body?.error?.code, body?.error?.message));
    };
    // The POST is never retried: a lost 202 may already be an admitted task.
    xhr.onerror = () => reject(new ApiError('network', COPY.generic));
    xhr.onabort = () => reject(abortError());
    signal?.addEventListener('abort', () => xhr.abort(), { once: true });
    const form = new FormData();
    form.append('file', file, file.name);
    xhr.send(form);
  });
}

async function getStatus(taskId, signal, deps) {
  // A hung GET times out and counts as a transport failure, so the page can never spin forever.
  const timeout = AbortSignal.timeout(STATUS_REQUEST_TIMEOUT_MS);
  const response = await deps.fetch(`${deps.base}/status/${encodeURIComponent(taskId)}`, {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  });
  if (response.status === 404) {
    const body = await response.json().catch(() => null);
    throw errorFor('task_not_found', body?.error?.message);
  }
  if (!response.ok) throw new TypeError(`status ${response.status}`); // transient: retried
  return response.json();
}

function checkedResult(result) {
  if (typeof result?.prediction?.score !== 'number' || !Number.isFinite(result.prediction.score)) {
    throw new ApiError('processing_failed', COPY.generic);
  }
  return result;
}

export async function analyze(file, { signal, onProgress, onStatus } = {}, overrides = {}) {
  const deps = { ...defaults, ...overrides };
  const start = deps.now();
  markRequestStart(start);
  const taskId = await upload(file, { signal, onProgress }, deps);
  onProgress?.(100);

  const pollStart = deps.now();
  let failures = 0;
  for (;;) {
    if (deps.now() - start >= TIMEOUT_MS) throw new ApiError('timeout', COPY.timeout);
    let status;
    try {
      status = await getStatus(taskId, signal, deps);
      failures = 0;
    } catch (error) {
      if (error?.name === 'AbortError' || error instanceof ApiError) throw error;
      if (failures >= RETRY_DELAYS_MS.length) throw new ApiError('network', COPY.generic);
      await deps.sleep(RETRY_DELAYS_MS[failures++], signal);
      continue;
    }
    if (status.state === 'done') return checkedResult(status.result);
    if (status.state === 'error') throw errorFor(status.error?.code, status.error?.message);
    onStatus?.({ ...status, task_id: status.task_id ?? taskId });
    const elapsed = deps.now() - pollStart;
    await deps.sleep(elapsed < FAST_POLL_WINDOW_MS ? FAST_POLL_MS : SLOW_POLL_MS, signal);
  }
}
