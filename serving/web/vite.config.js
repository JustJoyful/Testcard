import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';

// Development and preview only: production is same-origin behind FastAPI.
const api = process.env.VQA_API_URL ?? 'http://127.0.0.1:8001';
const proxy = Object.fromEntries(['/upload', '/status', '/health'].map((path) => [path, { target: api, changeOrigin: true }]));

export default defineConfig({
  plugins: [tailwindcss()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
  preview: { host: '127.0.0.1', port: 4173, strictPort: true, proxy },
  build: { target: 'es2022' },
});
