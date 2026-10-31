// Single source of truth. Every module renders from this object; it changes only through dispatch().

const BUSY = new Set(['inserting', 'uploading', 'queued', 'processing']);
const blank = { name: 'idle', file: null, taskId: null, stage: null, uploadPct: 0, result: null, error: null };

let state = Object.freeze({ ...blank });
// Request generation: events from a superseded request carry an old number and are dropped.
let generation = 0;
const listeners = new Set();

export const getState = () => state;
export const currentGeneration = () => generation;
export const isBusy = (s = state) => BUSY.has(s.name);

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function set(next) {
  state = Object.freeze({ ...state, ...next });
  for (const listener of listeners) listener(state);
}

export function dispatch(event) {
  const { type } = event;
  if ('gen' in event && event.gen !== generation) return;

  switch (type) {
    case 'reset':
      generation += 1;
      return set(blank);
    case 'standby': // /health failed; only an idle screen changes, work in progress is left alone
      if (state.name === 'idle') set({ ...blank, name: 'standby' });
      return;
    case 'online':
      if (state.name === 'standby' || (state.name === 'error' && state.error?.code === 'offline')) set(blank);
      return;
    case 'offline': // a tape was offered while the model is offline: nothing is uploaded
      if (!isBusy()) set({ ...blank, name: 'error', error: { code: 'offline', message: 'The model is offline. Try again later.' } });
      return;
    case 'dragenter':
      if (!isBusy() && state.name !== 'dragover' && state.name !== 'standby') set({ ...blank, name: 'dragover' });
      return;
    case 'dragleave':
      if (state.name === 'dragover') set(blank);
      return;
    case 'invalid':
      if (!isBusy()) set({ ...blank, name: 'invalid', file: event.file ?? null, error: event.error });
      return;
    case 'select':
      if (isBusy()) return;
      generation += 1;
      return set({ ...blank, name: 'inserting', file: event.file });
    case 'upload-start':
      if (state.name === 'inserting') set({ name: 'uploading', uploadPct: 0 });
      return;
    case 'progress':
      if (state.name === 'uploading') set({ uploadPct: Math.max(state.uploadPct, Math.min(100, Math.round(event.pct))) });
      return;
    case 'status': {
      const { status } = event;
      if (isBusy() && (status.state === 'queued' || status.state === 'processing')) {
        set({ name: status.state, taskId: status.task_id ?? state.taskId, stage: status.stage ?? null, uploadPct: 100 });
      }
      return;
    }
    case 'done':
      if (isBusy()) set({ name: 'done', taskId: event.result.task_id ?? state.taskId, stage: null, result: event.result });
      return;
    case 'fail':
      if (isBusy()) set({ name: 'error', stage: null, error: event.error });
      return;
    default:
      throw new Error(`Unknown event ${type}`);
  }
}

// Test-only: restore a clean module between cases.
export function _resetForTests() {
  generation = 0;
  state = Object.freeze({ ...blank });
  listeners.clear();
}
