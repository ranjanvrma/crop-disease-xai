// Browser tests of the built app at phone, tablet and desktop sizes.
//   npm run build && npm run test:e2e
// Needs web/public/model/ (scripts/export_web.py --demo) and web/test/fixtures/photo.png.
import { defineConfig, devices } from '@playwright/test';

const chromium = process.env.CHROMIUM_PATH ? { launchOptions: { executablePath: process.env.CHROMIUM_PATH } } : {};
const phone = (name, d) => ({ name, use: { ...devices[d], ...chromium, defaultBrowserType: 'chromium' } });

export default defineConfig({
  testDir: 'e2e',
  timeout: 120_000,
  outputDir: 'test-results',
  use: { baseURL: 'http://localhost:4173/', trace: 'off' },
  webServer: { command: 'npx vite preview --port 4173 --strictPort', url: 'http://localhost:4173/', reuseExistingServer: true },
  projects: [
    phone('small-phone', 'iPhone SE'),
    phone('android-phone', 'Pixel 7'),
    phone('budget-android', 'Galaxy S8'),
    phone('tablet', 'iPad Mini'),
    { name: 'desktop', use: { ...devices['Desktop Chrome'], ...chromium, viewport: { width: 1366, height: 860 } } },
  ],
});
