// Result panel: renders read-only server data. All values are set as text, never parsed as HTML.
import { secondsText, shortHash, middleEllipsis } from './format.js';
import { renderJson, serialize, copyText, downloadJson } from './json-view.js';

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== false && c !== ''));
  return node;
}

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

function flash(button, text, ms = 1500) {
  const original = button.dataset.label ?? button.textContent;
  button.dataset.label = original;
  button.textContent = text;
  clearTimeout(button._flash);
  button._flash = setTimeout(() => { button.textContent = original; }, ms);
}

function row(label, value, { muted = false, title } = {}) {
  if (!value) return null;
  return el('div', { class: `row${muted ? ' row-muted' : ''}`, title },
    el('dt', { text: label }), el('dd', { text: value }));
}

function hashRow(label, full, note) {
  if (typeof full !== 'string' || !full) return null;
  const button = el('button', {
    type: 'button',
    class: 'btn-mini',
    'aria-label': `Copy full ${label.toLowerCase()} hash`,
    text: 'Copy',
    onclick: () => copyText(full).then(
      () => { flash(button, 'Copied'); note.textContent = `${label} hash copied.`; },
      () => { flash(button, 'Failed'); note.textContent = 'Copy failed. Select the hash in the JSON tab instead.'; },
    ),
  });
  return el('div', { class: 'row' },
    el('dt', { text: label }),
    el('dd', {}, el('span', { class: 'hash', title: full, text: shortHash(full) }), button));
}

function summary(result, note) {
  const p = result.prediction;
  const t = isObj(p.timings_seconds) ? p.timings_seconds : {};
  const v = isObj(p.verified) ? p.verified : null;

  let analysis = secondsText(t.total);
  if (analysis && isNum(result.clip_seconds)) analysis += ` for a ${result.clip_seconds.toFixed(2)} s clip`;

  const timing = el('dl', { class: 'facts' },
    row('Analysis time', analysis),
    row('Feature extraction', secondsText(t.extraction)),
    row('Head', secondsText(t.head, { head: true })),
    row('Model startup', secondsText(p.startup_seconds), {
      muted: true,
      title: 'Time the server took to load the model when it started, not part of this request.',
    }),
    v && isNum(v.cpu_threads) ? row('CPU threads', String(v.cpu_threads)) : null,
  );

  // The score itself lives on the TV; this panel carries how it was produced.
  const model = v && el('div', { class: 'model-block' },
    el('p', { class: 'verified' }, el('span', { class: 'verified-mark', 'aria-hidden': 'true', text: '✓' }), 'Verified'),
    el('dl', { class: 'facts facts-muted' },
      row('Backbone weights', typeof v.weights_enum === 'string' ? v.weights_enum : ''),
      hashRow('Model', v.model_sha256, note),
      hashRow('Weights', v.weights_sha256, note),
      hashRow('Preprocessing', v.preprocessing_sha256, note),
    ));

  return el('div', { class: 'summary' }, timing, model);
}

function jsonPanel(result, note) {
  const text = serialize(result);
  const copy = el('button', {
    type: 'button', class: 'btn-ghost', text: 'Copy',
    onclick: () => copyText(text).then(
      () => { flash(copy, 'Copied'); note.textContent = 'JSON copied.'; },
      () => { flash(copy, 'Copy failed'); note.textContent = 'Copy failed. Use Download instead.'; },
    ),
  });
  const name = `testcard-${String(result.task_id ?? 'result').replace(/[^\w-]/g, '')}.json`;
  return el('div', { class: 'json-panel' },
    el('div', { class: 'json-actions' }, copy,
      el('button', { type: 'button', class: 'btn-ghost', text: 'Download', onclick: () => downloadJson(name, text) })),
    renderJson(result));
}

const TABS = ['summary', 'json'];

export function renderDetails(container, result, { tab = 'summary' } = {}) {
  const note = el('p', { class: 'sr-only', 'aria-live': 'polite' });
  const bodies = { summary: summary(result, note), json: jsonPanel(result, note) };
  const buttons = TABS.map((id) => el('button', {
    type: 'button', role: 'tab', class: 'tab', id: `tab-${id}`, 'aria-controls': `panel-${id}`,
    text: id === 'json' ? 'JSON' : 'Summary',
  }));
  const panels = TABS.map((id) => el('div', {
    role: 'tabpanel', id: `panel-${id}`, 'aria-labelledby': `tab-${id}`, class: 'panel',
  }, bodies[id]));

  const select = (index, focus) => {
    buttons.forEach((b, i) => {
      b.setAttribute('aria-selected', String(i === index));
      b.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
    });
    if (focus) buttons[index].focus();
  };
  const tablist = el('div', { role: 'tablist', 'aria-label': 'Result details', class: 'tabs' }, ...buttons);
  tablist.addEventListener('click', (e) => {
    const i = buttons.indexOf(e.target.closest('[role="tab"]'));
    if (i >= 0) select(i, false);
  });
  tablist.addEventListener('keydown', (e) => {
    const current = buttons.indexOf(document.activeElement);
    const last = buttons.length - 1;
    const next = { ArrowRight: current + 1, ArrowLeft: current - 1, Home: 0, End: last }[e.key];
    if (next === undefined || current < 0) return;
    e.preventDefault();
    select((next + buttons.length) % buttons.length, true);
  });

  const filename = typeof result.filename === 'string' ? result.filename : '';
  container.replaceChildren(el('section', { class: 'details-card', 'aria-label': 'Result details' },
    filename && el('p', { class: 'result-file', title: filename, text: middleEllipsis(filename, 48) }),
    tablist,
    ...panels,
    note));
  select(TABS.indexOf(tab), false);
  return { select: (id) => select(TABS.indexOf(id), false) };
}
