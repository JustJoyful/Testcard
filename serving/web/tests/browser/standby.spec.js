import { test, expect } from '@playwright/test';

// The built page with its backend stopped (VQA_STANDBY_URL serves dist/ with a dead API proxy).
test('backend stopped: the TV shows standby and nothing is ever uploaded', async ({ page }) => {
  const uploads = [];
  page.on('request', (request) => { if (new URL(request.url()).pathname === '/upload') uploads.push(request.url()); });
  page.on('filechooser', () => {});
  await page.goto('/');
  await expect(page.locator('#set')).toHaveAttribute('data-view', 'standby', { timeout: 15_000 });
  await expect(page.locator('.osd-main-text')).toHaveText('STANDBY');
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('.tape')]);
  await chooser.setFiles({ name: 'clip.mp4', mimeType: 'video/mp4', buffer: Buffer.alloc(64) });
  await page.waitForTimeout(1000);
  expect(uploads).toEqual([]);
});
