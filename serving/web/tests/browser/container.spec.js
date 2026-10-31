import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';

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
