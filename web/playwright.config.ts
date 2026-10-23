import { defineConfig } from '@playwright/test';

// APP_URL=https://<user>.github.io/<repo>/ tests a deployed copy instead of the local preview server
const appUrl = process.env.APP_URL;

export default defineConfig({
  testDir: 'tests/e2e',
  reporter: 'line',
  use: {
    channel: 'chrome', // Google Chrome (H.264/AAC); Chromium builds lack proprietary codecs
    headless: true,
    baseURL: appUrl ?? 'http://localhost:4173',
    launchOptions: { args: ['--allow-file-access-from-files'] },
  },
  webServer: appUrl ? undefined : {
    command: 'npx vite preview --port 4173 --strictPort',
    port: 4173,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
