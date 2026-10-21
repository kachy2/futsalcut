import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  reporter: 'line',
  use: {
    channel: 'chrome', // Google Chrome (H.264/AAC); Chromium builds lack proprietary codecs
    headless: true,
    baseURL: 'http://localhost:4173',
    launchOptions: { args: ['--allow-file-access-from-files'] },
  },
  webServer: {
    command: 'npx vite preview --port 4173 --strictPort',
    port: 4173,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
