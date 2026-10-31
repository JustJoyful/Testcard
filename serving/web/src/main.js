import './styles.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/vt323/400.css';
import '@fontsource/permanent-marker/400.css';
import { dispatch, getState, subscribe, currentGeneration, isBusy } from './state.js';
import { analyze, validateFile, requestStartedAt, ApiError, COPY } from './api.js';
import { renderDetails } from './result.js';
import { mountTv } from './tv.js';
import { scoreText } from './format.js';

const $ = (id) => document.getElementById(id);
const page = document.body;
const input = $('file');
const live = $('live');
const helper = $('helper');
const errorText = $('error-text');
const details = $('details');

const mock = import.meta.env.DEV && new URLSearchParams(location.search).has('mock');
let transport = analyze;
if (mock) {
  ({ analyzeMock: transport } = await import('./mock.js'));
  $('mock-badge').hidden = false;
}

let controller = null;

// Model availability from /health; while it is down nothing is ever uploaded.
let online = null;
let healthCheck = null;
async function checkHealth() {
  let ok = mock;
  if (!mock) {
    try {
      const response = await fetch('/health', { cache: 'no-store', signal: AbortSignal.timeout(5000) });
      const body = await response.json().catch(() => null);
      ok = response.ok && (body?.ok === true || body?.status === 'ready');
    } catch {
      ok = false;
    }
  }
  online = ok;
  dispatch({ type: ok ? 'online' : 'standby' });
  return ok;
}
healthCheck = checkHealth();
setInterval(() => { if (!document.hidden) healthCheck = checkHealth(); }, 15000);

function chooseFile() {
  if (!isBusy()) input.click();
}

async function handleFile(file) {
  if (!file || isBusy()) return;
  if (online === null) await healthCheck;
  if (!online) return dispatch({ type: 'offline' });
  const problem = validateFile(file);
  if (problem) return dispatch({ type: 'invalid', file, error: problem });
  start(file);
}

function start(file) {
  controller?.abort();
  controller = new AbortController();
  dispatch({ type: 'select', file });
  const gen = currentGeneration();
  dispatch({ type: 'upload-start', gen });
  transport(file, {
    signal: controller.signal,
    onProgress: (pct) => dispatch({ type: 'progress', gen, pct }),
    onStatus: (status) => dispatch({ type: 'status', gen, status }),
  }).then(
    (result) => {
      const score = result?.prediction?.score;
      if (typeof score !== 'number' || !Number.isFinite(score)) {
        return dispatch({ type: 'fail', gen, error: { code: 'processing_failed', message: COPY.generic } });
      }
      dispatch({ type: 'done', gen, result });
    },
    (error) => {
      if (error?.name === 'AbortError') return;
      const message = error instanceof ApiError ? error.message : COPY.generic;
      dispatch({ type: 'fail', gen, error: { code: error?.code ?? 'processing_failed', message } });
    },
  );
}

// Eject stops client activity only; an accepted server job still runs to completion.
function eject() {
  controller?.abort();
  controller = null;
  dispatch({ type: 'reset' });
  if (online === false) dispatch({ type: 'standby' });
  tv.focusTape();
}

let detailsView = null;
function openDetails(tab) {
  const s = getState();
  if (s.name !== 'done') return;
  if (!detailsView) detailsView = renderDetails(details, s.result, { tab });
  else detailsView.select(tab);
  details.hidden = false;
  requestAnimationFrame(() => {
    details.classList.add('is-open');
    // The panel sits below the TV, usually under the fold; bring it into view without moving focus.
    const smooth = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    details.scrollIntoView({ block: 'nearest', behavior: smooth ? 'smooth' : 'auto' });
  });
  tv.setExpanded(tab);
}
function closeDetails() {
  details.hidden = true;
  details.classList.remove('is-open');
  details.replaceChildren();
  detailsView = null;
  tv.setExpanded(null);
}

input.addEventListener('change', () => {
  const [file] = input.files;
  input.value = '';
  handleFile(file);
});

// The whole page accepts a dropped file; dropping never navigates away.
let depth = 0;
const hasFiles = (e) => Array.from(e.dataTransfer?.types ?? []).includes('Files');
window.addEventListener('dragenter', (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  depth += 1;
  dispatch({ type: 'dragenter' });
});
window.addEventListener('dragover', (e) => {
  e.preventDefault();
  if (hasFiles(e)) e.dataTransfer.dropEffect = isBusy() ? 'none' : 'copy';
});
window.addEventListener('dragleave', () => {
  depth = Math.max(0, depth - 1);
  if (depth === 0) dispatch({ type: 'dragleave' });
});
window.addEventListener('drop', (e) => {
  e.preventDefault();
  depth = 0;
  dispatch({ type: 'dragleave' });
  handleFile(e.dataTransfer?.files?.[0]);
});

// Cmd/Ctrl+O opens the picker instead of the browser's own open-file dialog.
const mac = /Mac|iPhone|iPad/.test(navigator.platform);
window.addEventListener('keydown', (e) => {
  if ((mac ? e.metaKey : e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'o') {
    e.preventDefault();
    chooseFile();
  }
});

const tv = mountTv($('set'), { chooseFile, onMenu: (act) => openDetails(act === 'json' ? 'json' : 'summary'), onEject: eject, startedAt: requestStartedAt });

// The screen's light follows it wherever the layout puts it.
const stage = $('main');
const screen = document.querySelector('.screen');
new ResizeObserver(() => {
  const a = stage.getBoundingClientRect();
  const b = screen.getBoundingClientRect();
  stage.style.setProperty('--sx', `${Math.round(b.left + b.width / 2 - a.left)}px`);
  stage.style.setProperty('--sy', `${Math.round(b.top + b.height / 2 - a.top)}px`);
}).observe(stage);

// The live region says what the screen shows, in plain words; upload progress only at 25 % steps.
const SAY = {
  idle: 'Insert a tape', standby: 'Standby. The model is offline.', dragover: 'Release to load',
  queued: 'Waiting', inserting: '', uploading: '',
};
const STAGE = { load: 'Warming up', extract: 'Reading frames', infer: 'Scoring' };
let announced = '';
let bucket = -1;
function announce(text) {
  if (text && text !== announced) {
    announced = text;
    live.textContent = text;
  }
}
function speak(s, first) {
  if (s.name === 'uploading') {
    const next = Math.floor(s.uploadPct / 25) * 25;
    if (next !== bucket) {
      bucket = next;
      announce(`Uploading, ${next} percent`);
    }
    return;
  }
  bucket = -1;
  if (first && s.name === 'idle') return; // the page already says this on load
  if (s.name === 'processing') announce(STAGE[s.stage] ?? 'Processing');
  else if (s.name === 'done') announce(`Result: ${scoreText(s.result.prediction.score)} out of 5`);
  else if (s.name === 'error' || s.name === 'invalid') announce(s.error.message);
  else announce(SAY[s.name]);
}

let previous = null;
function render(s) {
  page.dataset.state = s.name;
  const failed = s.name === 'error' || s.name === 'invalid';
  helper.hidden = failed;
  errorText.hidden = !failed;
  errorText.textContent = failed ? s.error.message : '';
  if (s.name === 'error' && previous?.name !== 'error') errorText.focus({ preventScroll: true });
  if (s.name !== 'done' && detailsView) closeDetails();

  tv.render(s);
  speak(s, previous === null);
  previous = s;
}

subscribe(render);
render(getState());
