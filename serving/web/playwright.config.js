import { defineConfig, devices } from '@playwright/test';

// Dev server (mock + routed transport) and a production preview of the built assets.
// VQA_CONTAINER_URL instead runs only the offline checks against an already running built image.
const container = process.env.VQA_CONTAINER_URL;
// VQA_STANDBY_URL runs only the stopped-backend check against an already running preview of dist/.
const standby = process.env.VQA_STANDBY_URL;
export default defineConfig({
  testDir: 'tests/browser',
  testMatch: container ? 'container.spec.js' : standby ? 'standby.spec.js' : 'demo.spec.js',
  timeout: 60_000,
  fullyParallel: true,
  // The real backend admits one upload at a time, so container checks run one by one.
  workers: container ? 1 : undefined,
  reporter: [['list']],
  use: { ...devices['Desktop Chrome'], baseURL: container ?? standby ?? 'http://127.0.0.1:5173', viewport: { width: 1440, height: 900 } },
  webServer: container || standby ? [] : [
    { command: 'npx vite', url: 'http://127.0.0.1:5173', reuseExistingServer: true },
    { command: 'npx vite build && npx vite preview', url: 'http://127.0.0.1:4173', reuseExistingServer: true, timeout: 120_000 },
  ],
});
