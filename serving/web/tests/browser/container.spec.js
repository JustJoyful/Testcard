import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { writeFileSync } from 'node:fs';

// Runs against the built image only (see playwright.config.js). Every request leaving the page origin is aborted.
const video = process.env.VQA_REAL_VIDEO;
let blocked;

test.beforeEach(async ({ page, baseURL }) => {
  blocked = [];
  page.on('filechooser', () => {});
  await page.route('**/*', (route) => {
    const url = route.request().url();
    if (new URL(url).origin === new URL(baseURL).origin || url.startsWith('data:')) return route.continue();
    blocked.push(url);
    return route.abort();
  });
});

test('built page and every font load from the image alone', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'idle', { timeout: 10_000 });
  await expect(page.locator('#mock-badge')).toBeHidden();
  const faces = await page.evaluate(async () => {
    await Promise.all([...document.fonts].map((face) => face.load()));
    return [...document.fonts].map((face) => [face.family.replaceAll('"', ''), face.status]);
  });
  expect(faces.filter(([, status]) => status !== 'loaded')).toEqual([]);
  expect(new Set(faces.map(([family]) => family))).toEqual(new Set(['Inter', 'JetBrains Mono', 'VT323', 'Permanent Marker']));
  expect(blocked).toEqual([]);
});

test('production ?mock still uses the real backend', async ({ page }) => {
  await page.goto('/?mock');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'idle', { timeout: 10_000 });
  await expect(page.locator('#mock-badge')).toBeHidden();
  expect(blocked).toEqual([]);
});

test('a real clip is scored end to end offline', async ({ page }) => {
  test.skip(!video, 'set VQA_REAL_VIDEO to a supported clip');
  test.setTimeout(240_000);
  await page.goto('/');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'idle', { timeout: 10_000 });
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('.tape')]);
  await chooser.setFiles(video);
  await expect(page.locator('.osd-score-value')).toHaveText(/^-?\d+\.\d\d$/, { timeout: 200_000 });
  await page.locator('.osd-item[data-act="json"]').click();
  const json = JSON.parse(await page.locator('.json-view').textContent());
  expect(Number.isFinite(json.prediction.score)).toBe(true);
  expect(json.prediction.verified.model_sha256).toMatch(/^[0-9a-f]{64}$/);
  expect(blocked).toEqual([]);
  await page.locator('.osd-item[data-act="details"]').click();
  await page.screenshot({ path: fileURLToPath(new URL('../../../../results/phase4b/browser/container-real-result.png', import.meta.url)) });
});

// Acceptance on the real backend: keyboard-only flow, what the page shows, and the 360 px layout.
const out = process.env.VQA_ACCEPTANCE_OUT;

test('keyboard only on the real backend: insert, result, details, JSON, eject', async ({ page }) => {
  test.skip(!video || !out, 'set VQA_REAL_VIDEO and VQA_ACCEPTANCE_OUT');
  test.setTimeout(240_000);
  await page.goto('/');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'idle', { timeout: 10_000 });
  for (let i = 0; i < 12 && !(await page.evaluate(() => document.activeElement?.classList.contains('tape'))); i += 1) {
    await page.keyboard.press('Tab');
  }
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.keyboard.press('Enter')]);
  await chooser.setFiles(video);
  const details = page.locator('.osd-item[data-act="details"]');
  await expect(details).toBeFocused({ timeout: 200_000 });
  const scoreText = await page.locator('.osd-score-value').textContent();
  const osdMeta = await page.locator('.osd-meta').textContent();
  await page.keyboard.press('Enter');
  await expect(page.locator('#panel-summary')).toBeVisible();
  const hashRows = await page.locator('#panel-summary .hash').evaluateAll((nodes) => nodes.map((n) => ({ text: n.textContent, full: n.title })));
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Enter');
  await expect(page.locator('#panel-json')).toBeVisible();
  const json = JSON.parse(await page.locator('.json-view').textContent());
  await page.keyboard.press('End');
  await expect(page.locator('.osd-item[data-act="eject"]')).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('body')).toHaveAttribute('data-state', 'idle');
  await expect(page.locator('.tape')).toBeFocused();
  expect(blocked).toEqual([]);
  writeFileSync(`${out}/ui_result.json`, JSON.stringify({ scoreText, osdMeta, hashRows, json }, null, 2));
});

test('360 px: no horizontal scroll at idle or with a real result', async ({ browser, baseURL }) => {
  test.skip(!video, 'set VQA_REAL_VIDEO');
  test.setTimeout(240_000);
  const context = await browser.newContext({ viewport: { width: 360, height: 780 }, hasTouch: true, isMobile: true, baseURL });
  const page = await context.newPage();
  page.on('filechooser', () => {});
  await page.goto('/');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'idle', { timeout: 10_000 });
  const width = () => page.evaluate(() => document.documentElement.scrollWidth);
  expect(await width()).toBeLessThanOrEqual(360);
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.tap('.tape')]);
  await chooser.setFiles(video);
  await expect(page.locator('.osd-score-value')).toHaveText(/^-?\d+\.\d\d$/, { timeout: 200_000 });
  await page.locator('.osd-item[data-act="json"]').tap();
  expect(await width()).toBeLessThanOrEqual(360);
  await context.close();
});
