// Development-only fake backend (DESIGN.md section 11). Imported dynamically under import.meta.env.DEV && ?mock.
// Fixture timings below are pretend; they are never performance evidence.
import { ApiError, COPY, markRequestStart } from './api.js';

const PREDICTION = {
  score: 3.596115361254694,
  timings_seconds: { extraction: 23.884803736000322, head: 0.00724809399980586, total: 23.892053884992492 },
  verified: {
    model_sha256: '4c9782aefca4b81da4d15da31681f11ca8ded4e890d36d67fb478663583327a3',
    weights_sha256: '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca',
    weights_enum: 'ResNet50_Weights.IMAGENET1K_V2',
    preprocessing_sha256: 'acc2e85da8cce49f23d21d990af382111cd1a9352d2fdeeb4731d06e3fe1678b',
    cpu_threads: 4,
  },
  startup_seconds: 1.2390047389926622,
};

function wait(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('Aborted', 'AbortError'));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    }, { once: true });
  });
}

export async function analyzeMock(file, { signal, onProgress, onStatus } = {}) {
  markRequestStart();
  const params = new URLSearchParams(location.search);
  const scale = params.has('slow') ? 10 : 1;
  const taskId = crypto.randomUUID().replaceAll('-', '');
  for (let pct = 0; pct <= 100; pct += 20) {
    onProgress?.(pct);
    await wait(150 * scale, signal);
  }
  onStatus?.({ task_id: taskId, state: 'queued', stage: null });
  await wait(2000 * scale, signal);
  onStatus?.({ task_id: taskId, state: 'processing', stage: 'extract' });
  await wait(6000 * scale, signal);
  if (params.get('fail') === 'decode') throw new ApiError('invalid_video', COPY.invalid_video);
  onStatus?.({ task_id: taskId, state: 'processing', stage: 'infer' });
  await wait(500 * scale, signal);
  const out = params.get('out');
  return {
    task_id: taskId,
    filename: file.name,
    clip_seconds: 19.42,
    created_at: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
    prediction: out === null ? PREDICTION : { ...PREDICTION, score: Number(out) },
  };
}
