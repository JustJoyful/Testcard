import { test, expect } from '@playwright/test';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

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
const RESULT = (over = {}) => ({
  task_id: 't1', filename: 'clip.mp4', clip_seconds: 19.42, created_at: '2026-10-31T00:17:00Z', prediction: PREDICTION, ...over,
});

// Fake backend behind the real api.js transport. steps are returned in order, then the terminal result.
async function fakeBackend(page, { result = RESULT(), steps = [{ state: 'processing', stage: 'extract' }], upload } = {}) {
  const calls = { upload: 0, status: 0 };
  await page.route('**/upload', async (route) => {
    calls.upload += 1;
    const { status = 202, body = { task_id: 't1' } } = upload ?? {};
    await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  });
  await page.route('**/status/*', async (route) => {
    const step = steps[calls.status++];
    const body = step
      ? { task_id: 't1', stage: null, result: null, error: null, ...step }
      : { task_id: 't1', state: 'done', stage: null, result, error: null };
    await route.fulfill({ json: body });
  });
  return calls;
}

// The model is up unless a test says otherwise; Playwright arms file-chooser interception lazily,
// so arm it before any instant key press.
test.beforeEach(async ({ page }) => {
  page.on('filechooser', () => {});
  await page.route('**/health', (route) => route.fulfill({ json: { status: 'ready', verified: {} } }));
});

async function pick(page, name = 'clip.mp4') {
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('.tape')]);
  await chooser.setFiles({ name, mimeType: '', buffer: Buffer.alloc(64) });
}
async function ready(page, url = '/') {
  await page.goto(url);
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'idle');
}

const view = (page) => page.locator('#set').getAttribute('data-view');
const frames = (page) => page.locator('canvas.noise').getAttribute('data-frames').then(Number);
const score = (page) => page.locator('.osd-score-value');
const menu = (page, act) => page.locator(`.osd-item[data-act="${act}"]`);

async function recordViews(page) {
  await page.evaluate(() => {
    const set = document.querySelector('#set');
    window.__views = [set.dataset.view];
    new MutationObserver(() => {
      if (window.__views.at(-1) !== set.dataset.view) window.__views.push(set.dataset.view);
    }).observe(set, { attributes: true, attributeFilter: ['data-view'] });
  });
}

// ---------- The flow ----------

test('keyboard only: insert, result, details, JSON, eject', async ({ page }) => {
  const calls = await fakeBackend(page, { steps: [{ state: 'queued' }, { state: 'processing', stage: 'extract' }] });
  await ready(page);
  await expect(page.locator('#mock-badge')).toBeHidden();

  for (let i = 0; i < 12 && !(await page.evaluate(() => document.activeElement?.classList.contains('tape'))); i += 1) {
    await page.keyboard.press('Tab');
  }
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.keyboard.press('Enter')]);
  await chooser.setFiles({ name: 'clip.mp4', mimeType: 'video/mp4', buffer: Buffer.alloc(64) });

  await expect(menu(page, 'details')).toBeFocused({ timeout: 10_000 });
  await expect(score(page)).toHaveText('3.60');
  expect(calls.upload).toBe(1);

  await page.keyboard.press('Enter');
  await expect(page.locator('#panel-summary')).toBeVisible();
  await expect(menu(page, 'details')).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.press('ArrowRight');
  await expect(menu(page, 'json')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('#panel-json')).toBeVisible();
  await page.keyboard.press('End');
  await expect(menu(page, 'eject')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('body')).toHaveAttribute('data-state', 'idle');
  await expect(page.locator('#details')).toBeHidden();
  await expect(page.locator('.tape')).toBeFocused();
});

test('dev mock reaches the score screen with a badge', async ({ page }) => {
  await page.goto('/?mock');
  await expect(page.locator('#mock-badge')).toBeVisible();
  await pick(page, 'mock.mp4');
  await expect(score(page)).toHaveText('3.60', { timeout: 15_000 });
});

test('production build: ?mock uses the real /upload and ships no fixture', async ({ page }) => {
  const calls = await fakeBackend(page, { steps: [] });
  await ready(page, 'http://127.0.0.1:4173/?mock');
  await expect(page.locator('#mock-badge')).toBeHidden();
  await pick(page);
  await expect(score(page)).toHaveText('3.60', { timeout: 10_000 });
  expect(calls.upload).toBe(1);

  const assets = fileURLToPath(new URL('../../dist/assets/', import.meta.url));
  for (const file of readdirSync(assets).filter((f) => f.endsWith('.js'))) {
    const text = readFileSync(assets + file, 'utf8');
    expect(text).not.toContain('analyzeMock');
    expect(text).not.toContain('4c9782aefca4');
    expect(text).not.toContain('Mock backend');
  }
});

test('errors show a short reason on screen, the full sentence below, and allow a retry', async ({ page }) => {
  await fakeBackend(page, { steps: [{ state: 'error', error: { code: 'invalid_video', message: 'raw' } }] });
  await ready(page);
  await pick(page);
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'error');
  await expect(page.locator('.osd-main-text')).toHaveText('NO SIGNAL');
  await expect(page.locator('.osd-sub')).toHaveText("CAN'T READ THIS VIDEO");
  await expect(page.locator('#error-text')).toHaveText("We couldn't read that video. Try re-exporting it as MP4.");
  await expect(page.locator('#error-text')).toBeFocused();
  await expect(page.locator('.vcr-eject')).toBeEnabled();
  await page.click('.vcr-eject');
  await expect(page.locator('body')).toHaveAttribute('data-state', 'idle');
  await expect(page.locator('.vcr-eject')).toBeDisabled();

  await pick(page, 'notes.mkv');
  await expect(page.locator('.osd-main-text')).toHaveText('TAPE REJECTED');
  await expect(page.locator('.osd-sub')).toHaveText('MP4 · MOV · WEBM ONLY');
  await expect(page.locator('#error-text')).toHaveText("That file type isn't supported. Use MP4, MOV, or WEBM.");
});

test('busy server shows NO SIGNAL with its reason', async ({ page }) => {
  await fakeBackend(page, { upload: { status: 503, body: { error: { code: 'busy', message: 'Busy processing another video; retry later.' } } } });
  await ready(page);
  await pick(page);
  await expect(page.locator('.osd-main-text')).toHaveText('NO SIGNAL');
  await expect(page.locator('.osd-sub')).toHaveText('BUSY · TRY AGAIN SOON');
  await expect(page.locator('#error-text')).toHaveText('Busy processing another video; retry later.');
});

test('real backend, when one is running', async ({ page }) => {
  const video = process.env.VQA_REAL_VIDEO;
  test.skip(!video, 'set VQA_REAL_VIDEO to a clip and run the API on 127.0.0.1:8001');
  test.setTimeout(240_000);
  await page.unroute('**/health');
  await ready(page);
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('.tape')]);
  await chooser.setFiles(video);
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'processing', { timeout: 30_000 });
  await expect(score(page)).toHaveText(/^-?\d+\.\d\d$/, { timeout: 200_000 });
  await menu(page, 'json').click();
  const json = JSON.parse(await page.locator('.json-view').textContent());
  expect(Number.isFinite(json.prediction.score)).toBe(true);
  expect(json.prediction.verified.model_sha256).toMatch(/^[0-9a-f]{64}$/);
});

// ---------- Standby ----------

test('standby: with the model offline the TV says so and nothing is uploaded', async ({ page }) => {
  const calls = await fakeBackend(page, { steps: [] });
  await page.route('**/health', (route) => route.fulfill({ status: 503, body: '' }));
  await page.goto('/');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'standby');
  await expect(page.locator('.osd-main-text')).toHaveText('STANDBY');
  await expect(page.locator('.osd-sub')).toHaveText('MODEL OFFLINE');
  expect(await page.locator('.vcr-led').evaluate((e) => getComputedStyle(e).boxShadow)).toContain('inset');

  await pick(page);
  await expect(page.locator('.osd-main-text')).toHaveText('NO SIGNAL');
  await expect(page.locator('.osd-sub')).toHaveText('TRY AGAIN LATER');
  const dt = await page.evaluateHandle(() => { const d = new DataTransfer(); d.items.add(new File(['x'], 'dropped.mp4')); return d; });
  await page.dispatchEvent('main', 'drop', { dataTransfer: dt });
  await page.waitForTimeout(300);
  expect(calls.upload).toBe(0);
});

// ---------- Score screen and details ----------

test('details keep unknown fields and small times; summary has no second score', async ({ page }) => {
  const result = RESULT({ extra: { nested: [1.0000000000001, null, true] }, prediction: { ...PREDICTION, future_field: 'kept' } });
  await fakeBackend(page, { steps: [], result });
  await ready(page);
  await pick(page);
  await expect(score(page)).toHaveText('3.60');
  await expect(page.locator('.osd-meta')).toHaveText(/ANALYSIS 23\.89 S\s+·\s+MODEL 4C9782AE\s+·\s+VERIFIED/);
  await expect(page.locator('.osd-range')).toBeHidden();
  await menu(page, 'details').click();
  const summary = page.locator('#panel-summary');
  await expect(summary).toContainText('23.89 s for a 19.42 s clip');
  await expect(summary).toContainText('23.88 s');
  await expect(summary).toContainText('< 0.01 s');
  await expect(summary).toContainText('1.24 s');
  await expect(summary).toContainText('CPU threads');
  await expect(summary).toContainText('Verified');
  await expect(summary).not.toContainText('3.60');
  for (const [label, hash] of [['Model', PREDICTION.verified.model_sha256], ['Weights', PREDICTION.verified.weights_sha256], ['Preprocessing', PREDICTION.verified.preprocessing_sha256]]) {
    await expect(summary.locator('.hash', { hasText: `${hash.slice(0, 8)}…` })).toHaveAttribute('title', hash);
    await expect(summary.getByRole('button', { name: `Copy full ${label.toLowerCase()} hash` })).toBeVisible();
  }
  await expect(summary).not.toContainText('future_field');

  await menu(page, 'json').click();
  expect(JSON.parse(await page.locator('.json-view').textContent())).toEqual(result);
  expect(await page.locator('.json-view').textContent()).toContain('1.0000000000001');
});

for (const [value, text, outside] of [[0.4, '0.40', true], [1, '1.00', false], [5, '5.00', false], [5.25, '5.25', true], [-0.25, '-0.25', true]]) {
  test(`score ${value} is unclipped, out-of-range line ${outside}`, async ({ page }) => {
    await fakeBackend(page, { steps: [], result: RESULT({ prediction: { ...PREDICTION, score: value } }) });
    await ready(page);
    await pick(page);
    await expect(score(page)).toHaveText(text);
    if (outside) await expect(page.locator('.osd-range')).toHaveText('OUTSIDE TRAINED RANGE 1–5');
    else await expect(page.locator('.osd-range')).toBeHidden();
  });
}

test('a result with only a score never claims VERIFIED', async ({ page }) => {
  await fakeBackend(page, { steps: [], result: { prediction: { score: 2 } } });
  await ready(page);
  await pick(page);
  await expect(score(page)).toHaveText('2.00');
  await expect(page.locator('.osd-meta')).toHaveText('');
  await menu(page, 'details').click();
  const summary = page.locator('#panel-summary');
  await expect(summary).not.toContainText('Verified');
  await expect(summary).not.toContainText('Analysis time');
});

test('result values are text, not HTML', async ({ page }) => {
  const evil = '<img src=x onerror="window.__pwned=1">';
  await fakeBackend(page, { steps: [], result: RESULT({ filename: evil, [evil]: evil, prediction: { ...PREDICTION, weights_enum: evil } }) });
  await ready(page);
  await pick(page);
  await expect(score(page)).toHaveText('3.60');
  await menu(page, 'json').click();
  await expect(page.locator('#details img, #details script, #set img')).toHaveCount(0);
  expect(await page.evaluate(() => window.__pwned)).toBeUndefined();
  await expect(page.locator('.result-file')).toHaveAttribute('title', evil);
});

test('an invalid score goes to the error flow', async ({ page }) => {
  await fakeBackend(page, { steps: [], result: RESULT({ prediction: { ...PREDICTION, score: 'NaN' } }) });
  await ready(page);
  await pick(page);
  await expect(page.locator('#error-text')).toHaveText('Something went wrong while scoring. Try again.');
  await expect(page.locator('#details')).toBeHidden();
});

test('details tabs, JSON copy and download preserve the exact object', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const result = RESULT({ unknown: { a: [1, 2] } });
  await fakeBackend(page, { steps: [], result });
  await ready(page);
  await pick(page);
  await menu(page, 'details').click();
  await page.focus('#tab-summary');
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('#tab-json')).toBeFocused();
  await expect(page.locator('#panel-json')).toBeVisible();
  await page.keyboard.press('Home');
  await expect(page.locator('#panel-summary')).toBeVisible();
  await page.keyboard.press('End');

  const copy = page.locator('#panel-json .json-actions button').first();
  await copy.click();
  await expect(copy).toHaveText('Copied');
  expect(JSON.parse(await page.evaluate(() => navigator.clipboard.readText()))).toEqual(result);
  await expect(copy).toHaveText('Copy', { timeout: 2500 });

  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download' }).click()]);
  expect(download.suggestedFilename()).toBe('vqa-t1.json');
  expect(JSON.parse(readFileSync(await download.path(), 'utf8'))).toEqual(result);
});

// ---------- The screen in every state ----------

test('screen states', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' }); // static rendering: no animation holds
  const shots = process.env.VQA_SHOTS;
  const snap = async (name) => shots && page.screenshot({ path: `${shots}/${name}.png` });
  let release;
  const gate = new Promise((r) => { release = r; });
  const steps = [{ state: 'queued' }, { state: 'processing', stage: 'extract' }];
  await page.route('**/upload', async (route) => { await gate; await route.fulfill({ status: 202, json: { task_id: 't1' } }); });
  let n = 0;
  await page.route('**/status/*', (route) => route.fulfill({
    json: n < 2 ? { task_id: 't1', result: null, error: null, ...steps[n++] } : { task_id: 't1', state: 'done', result: RESULT(), error: null },
  }));
  await ready(page);

  const box = await page.locator('.screen').boundingBox();
  expect(Math.abs(box.width / box.height - 4 / 3)).toBeLessThan(0.02);
  await expect(page.locator('.osd-main-text')).toHaveText('INSERT TAPE');
  await expect(page.locator('.osd-tl')).toHaveText('CH 03');
  await expect(page.locator('.osd-cursor')).toBeVisible();
  await expect(page.locator('.vfd-text')).toHaveText('12:00');
  await snap('idle');

  const dt = await page.evaluateHandle(() => { const d = new DataTransfer(); d.items.add(new File(['x'], 'a.mp4')); return d; });
  await page.dispatchEvent('main', 'dragenter', { dataTransfer: dt });
  expect(await view(page)).toBe('dragover');
  await expect(page.locator('.osd-main-text')).toHaveText('RELEASE TO LOAD');
  await snap('dragover');
  await page.dispatchEvent('main', 'dragleave', { dataTransfer: dt });

  await pick(page, 'bad.mkv');
  expect(await view(page)).toBe('invalid');
  await expect(page.locator('.vfd-text')).toHaveText('ERR');
  await snap('invalid');

  await pick(page, 'A very long holiday video filename take 3.mp4');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'uploading');
  await expect(page.locator('.osd-main-text')).toHaveText('LOADING');
  await expect(page.locator('.cells i')).toHaveCount(10);
  await expect(page.locator('.osd-pct')).toHaveText(/^\d+%$/);
  const label = await page.locator('.label-file').textContent();
  expect(Array.from(label)).toHaveLength(22);
  await snap('uploading');
  release();

  await expect(page.locator('#set')).toHaveAttribute('data-view', 'queued');
  await expect(page.locator('.osd-main-text')).toHaveText('WAITING');
  await expect(page.locator('.osd-tl')).toHaveText('▶ PLAY');
  await expect(page.locator('.osd-br')).toHaveText(/^\d\d:\d\d:\d\d$/);
  await snap('queued');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'processing');
  await expect(page.locator('.osd-main-text')).toHaveText('READING FRAMES');
  await snap('processing');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'done');
  await expect(page.locator('.osd-tl')).toHaveText('■ STOP');
  await expect(score(page)).toHaveText('3.60');
  await expect(page.locator('.vfd-text')).toHaveText('END');
  await snap('done');
});

for (const [width, height] of [[1920, 1080], [1440, 900]]) {
  test(`the object and helper line fit at ${width}x${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await ready(page);
    const box = await page.evaluate(() => {
      const r = (s) => document.querySelector(s).getBoundingClientRect();
      return { tape: r('.tape').bottom, tapeRight: r('.tape').right, helper: r('.helper-line').bottom, screen: r('.screen').width, vw: innerWidth, vh: innerHeight };
    });
    expect(box.tape).toBeLessThanOrEqual(box.vh);
    expect(box.helper).toBeLessThanOrEqual(box.vh);
    expect(box.tapeRight).toBeLessThanOrEqual(box.vw);
    expect(box.screen).toBeLessThanOrEqual(Math.min(720, 0.62 * width) + 1);
  });
}

for (const width of [360, 640, 1024, 1440]) {
  test(`no horizontal scroll at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const name = `${'Ünïcödé 名前 très long '.repeat(6)}.mp4`;
    await fakeBackend(page, { steps: [], result: RESULT({ filename: name, notes: 'x'.repeat(600) }) });
    await ready(page);
    const fits = () => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
    expect(await fits()).toBe(true);
    await pick(page, name);
    await expect(score(page)).toBeVisible();
    await menu(page, 'json').click();
    expect(await fits()).toBe(true);
  });
}

test('phone: tapping the tape opens the picker, no drag-only instruction', async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 360, height: 780 }, hasTouch: true, isMobile: true });
  const page = await context.newPage();
  page.on('filechooser', () => {});
  await page.route('**/health', (route) => route.fulfill({ json: { status: 'ready' } }));
  await page.goto('http://127.0.0.1:5173/');
  await expect(page.locator('.label-drag')).toBeHidden();
  await expect(page.locator('.label-tap')).toBeVisible();
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.tap('.tape')]);
  expect(chooser).toBeTruthy();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360);
  await context.close();
});

test('only the listed elements live outside the object', async ({ page }) => {
  await ready(page);
  const stray = await page.evaluate(() => [...document.querySelectorAll('body *')]
    .filter((e) => !e.closest('#set, .topbar, .helper-line, #details, .footer, .skip-link, .sr-only, script, .spill')
      && e.getClientRects().length && e.children.length === 0 && e.textContent.trim())
    .map((e) => e.outerHTML.slice(0, 80)));
  expect(stray).toEqual([]);
  await expect(page.locator('.nav a')).toHaveText(['Method', 'Results', 'GitHub']);
  for (const gone of ['.eyebrow', '.pill', '.key', 'kbd', '.amb-grid', '.amb-vignette', '.steps', '.facts-grid', '.vcr-model', '.btn-primary', '.desk']) {
    await expect(page.locator(gone)).toHaveCount(0);
  }
  const families = await page.evaluate(() => [...new Set([...document.querySelectorAll('*')].map((e) => getComputedStyle(e).fontFamily))].join('|'));
  expect(families).not.toContain('Instrument');
});

// ---------- Motion and lifecycle ----------

test('motion: done during insert skips stale states', async ({ page }) => {
  await fakeBackend(page, { steps: [] });
  await ready(page);
  await recordViews(page);
  const start = Date.now();
  await pick(page);
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'done', { timeout: 3000 });
  expect(Date.now() - start).toBeLessThan(2500); // one 600 ms insert hold plus load; the view sequence below is the real check
  expect(await page.evaluate(() => window.__views)).toEqual(['idle', 'inserting', 'done']);
});

test('motion: eject during the done burst ignores old callbacks', async ({ page }) => {
  await fakeBackend(page, { steps: [] });
  await ready(page);
  await pick(page);
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'done');
  await page.click('.vcr-eject');
  await page.waitForTimeout(1500);
  await expect(page.locator('body')).toHaveAttribute('data-state', 'idle');
  expect(await view(page)).toBe('idle');
  await expect(page.locator('#details')).toBeHidden();
  await expect(page.locator('.osd-main-text')).toHaveText('INSERT TAPE');
});

test('motion: noise pauses when hidden and after dispose', async ({ page }) => {
  await ready(page);
  await page.waitForTimeout(300);
  expect(await frames(page)).toBeGreaterThan(2);
  const setHidden = (hidden) => page.evaluate((h) => {
    Object.defineProperty(document, 'hidden', { configurable: true, get: () => h });
    document.dispatchEvent(new Event('visibilitychange'));
  }, hidden);
  await setHidden(true);
  const paused = await frames(page);
  await page.waitForTimeout(400);
  expect(await frames(page)).toBe(paused);
  await setHidden(false);
  await page.waitForTimeout(300);
  expect(await frames(page)).toBeGreaterThan(paused);

  const [before, after] = await page.evaluate(async () => {
    const { mountTv } = await import('/src/tv.js');
    const el = document.createElement('div');
    el.className = 'set';
    document.body.append(el);
    const tv = mountTv(el, { chooseFile() {}, onMenu() {}, onEject() {} });
    tv.render({ name: 'idle', file: null, uploadPct: 0, stage: null, result: null, error: null });
    await new Promise((r) => setTimeout(r, 300));
    tv.dispose();
    const n = Number(el.querySelector('canvas').dataset.frames);
    await new Promise((r) => setTimeout(r, 400));
    return [n, Number(el.querySelector('canvas').dataset.frames)];
  });
  expect(before).toBeGreaterThan(0);
  expect(after).toBe(before);
});

test('reduced motion: same screens and keyboard flow, nothing moves', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await fakeBackend(page, { steps: [{ state: 'processing', stage: 'extract' }, { state: 'processing', stage: 'extract' }] });
  await ready(page);
  await page.waitForTimeout(300);
  expect(await frames(page)).toBe(1);
  const anim = (sel) => page.locator(sel).first().evaluate((e) => getComputedStyle(e).animationName);
  expect(await anim('.vfd-text')).toBe('none');
  expect(await anim('.osd-cursor')).toBe('none');
  expect(await anim('.spill')).toBe('none');
  expect(await page.locator('.power-on').evaluate((e) => getComputedStyle(e).display)).toBe('none');

  await page.focus('.tape');
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.keyboard.press('Enter')]);
  await chooser.setFiles({ name: 'clip.mp4', mimeType: '', buffer: Buffer.alloc(64) });
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'processing');
  expect(await page.locator('.tracking').evaluate((e) => getComputedStyle(e).display)).toBe('none');
  await expect(page.locator('.osd-main-text')).toHaveText('READING FRAMES');
  await expect(menu(page, 'details')).toBeFocused({ timeout: 10_000 });
  await expect(score(page)).toHaveText('3.60');
});

test('announcements say what the screen shows, upload at 25 percent steps', async ({ page }) => {
  await page.goto('/?mock');
  await page.evaluate(() => {
    const live = document.querySelector('#live');
    window.__said = [];
    new MutationObserver(() => window.__said.push(live.textContent)).observe(live, { childList: true, characterData: true, subtree: true });
  });
  await pick(page, 'mock.mp4');
  await expect(menu(page, 'details')).toBeFocused({ timeout: 15_000 });
  await menu(page, 'eject').click();
  await expect.poll(() => page.evaluate(() => window.__said.at(-1))).toBe('Insert a tape');
  expect(await page.evaluate(() => window.__said)).toEqual([
    'Uploading, 0 percent', 'Uploading, 25 percent', 'Uploading, 50 percent', 'Uploading, 75 percent', 'Uploading, 100 percent',
    'Waiting', 'Reading frames', 'Scoring', 'Result: 3.60 out of 5', 'Insert a tape',
  ]);
});

test('Ctrl O opens the picker and a drop anywhere loads the tape', async ({ page }) => {
  const calls = await fakeBackend(page, { steps: [] });
  await ready(page);
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.keyboard.press('Control+o')]);
  expect(chooser).toBeTruthy();
  const dt = await page.evaluateHandle(() => { const d = new DataTransfer(); d.items.add(new File(['x'], 'dropped.mp4')); return d; });
  await page.dispatchEvent('.footer', 'drop', { dataTransfer: dt });
  await expect(score(page)).toHaveText('3.60', { timeout: 10_000 });
  expect(calls.upload).toBe(1);
});

test('the screen light follows state', async ({ page }) => {
  await fakeBackend(page, { steps: [{ state: 'processing', stage: 'extract' }, { state: 'processing', stage: 'extract' }] });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await ready(page);
  const spill = () => page.evaluate(() => getComputedStyle(document.body).getPropertyValue('--spill').trim());
  const idle = await spill();
  await pick(page);
  await expect(page.locator('body')).toHaveAttribute('data-state', 'processing');
  const blue = await spill();
  await expect(page.locator('body')).toHaveAttribute('data-state', 'done', { timeout: 10_000 });
  const warm = await spill();
  expect(new Set([idle, blue, warm]).size).toBe(3);
});

test('motion: the tape travels into the slot when a file is chosen', async ({ page }) => {
  await fakeBackend(page, { steps: [{ state: 'processing', stage: 'extract' }] });
  await ready(page);
  await pick(page);
  const motion = await page.evaluate(() => {
    const tape = document.querySelector('.tape');
    const a = tape.getAnimations().find((x) => !(x instanceof CSSTransition) && !(x instanceof CSSAnimation));
    if (!a) return null;
    a.pause();
    a.currentTime = 0.5 * a.effect.getTiming().duration; // squared up over the slot
    const tapeBox = tape.getBoundingClientRect();
    const slot = document.querySelector('.vcr-slot').getBoundingClientRect();
    const last = a.effect.getKeyframes().at(-1);
    return {
      duration: a.effect.getTiming().duration,
      overSlot: tapeBox.left >= slot.left - 2 && tapeBox.right <= slot.right + 2 && Math.abs(tapeBox.bottom - slot.top) < 6,
      endsClipped: /100%/.test(last.clipPath),
    };
  });
  expect(motion).toEqual({ duration: 600, overSlot: true, endsClipped: true });
});

test('opening details or JSON brings the panel on screen', async ({ page }) => {
  await fakeBackend(page);
  await ready(page);
  await pick(page);
  await expect(score(page)).toHaveText('3.60', { timeout: 10_000 });
  for (const act of ['details', 'json']) {
    await menu(page, act).click();
    await expect(page.locator('#details')).toBeInViewport({ ratio: 0.25 });
    await page.evaluate(() => window.scrollTo(0, 0));
  }
});
