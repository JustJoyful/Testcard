import { defineConfig, devices } from '@playwright/test';

// Dev server (mock + routed transport) and a production preview of the built assets.
// VQA_CONTAINER_URL instead runs only the offline checks against an already running built image.
const container = process.env.VQA_CONTAINER_URL;
export default defineConfig({
  testDir: 'tests/browser',
  testMatch: container ? 'container.spec.js' : 'demo.spec.js',
  timeout: 60_000,
  fullyParallel: true,
  reporter: [['list']],
  use: { ...devices['Desktop Chrome'], baseURL: container ?? 'http://127.0.0.1:5173', viewport: { width: 1440, height: 900 } },
  webServer: container ? [] : [
    { command: 'npx vite', url: 'http://127.0.0.1:5173', reuseExistingServer: true },
    { command: 'npx vite build && npx vite preview', url: 'http://127.0.0.1:4173', reuseExistingServer: true, timeout: 120_000 },
  ],
});
