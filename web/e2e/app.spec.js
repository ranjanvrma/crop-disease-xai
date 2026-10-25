import { expect, test } from '@playwright/test';

const PHOTO = new URL('../test/fixtures/photo.png', import.meta.url).pathname;

async function ready(page) {
  await page.goto('./');
  await expect(page.locator('body[data-ready="true"]')).toBeAttached({ timeout: 60_000 });
}

async function noSideScroll(page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
}

test('diagnoses a photo and explains it on the device', async ({ page }, info) => {
  await ready(page);
  await expect(page.locator('#banner')).toBeVisible();
  await noSideScroll(page);

  await page.setInputFiles('#gallery', PHOTO);
  await expect(page.locator('#verdict h2')).toBeVisible();
  await expect(page.locator('#classpick .pill')).toHaveCount(3);
  await expect(page.locator('#faithful')).toContainText(/Passes|Doubtful/, { timeout: 60_000 });

  // the heatmap was really drawn (not a blank canvas)
  const varied = await page.evaluate(() => {
    const c = document.getElementById('overlay');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    return new Set(Array.from({ length: 500 }, (_, i) => d[(i * 997 * 4) % d.length])).size;
  });
  expect(varied).toBeGreaterThan(20);

  await page.locator('#classpick .pill').nth(1).click();
  await expect(page.locator('#heatnote')).toContainText('instead of the top answer');

  await page.locator('#expert summary').click();
  await expect(page.locator('#hoodbody')).toContainText('this split is exact');
  await expect(page.locator('#hoodbody canvas[data-channel]')).toHaveCount(6);
  await noSideScroll(page);
  await page.screenshot({ path: `test-results/${info.project.name}.png`, fullPage: true });
});

test('keeps working offline after the first visit', async ({ page, context }) => {
  await ready(page);
  await page.evaluate(() => navigator.serviceWorker.ready);
  await expect.poll(() => page.evaluate(async () => (await caches.keys()).length)).toBeGreaterThan(0);
  await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);
  await context.setOffline(true);
  await ready(page);
  await expect(page.locator('#net')).toContainText('Offline');
  await page.setInputFiles('#gallery', PHOTO);
  await expect(page.locator('#verdict h2')).toBeVisible();
});

test('warns when a photo is not a supported crop, and lists the crops', async ({ page }) => {
  await ready(page);
  await expect(page.locator('#supported .cropchips li')).toHaveCount(14);
  await page.locator('[data-sample$="cat-not-a-crop.jpg"]').click();
  await expect(page.locator('#verdict')).toHaveClass(/unsupported/);
  await expect(page.locator('#remedy')).toContainText('No advice');
  await page.locator('#anyway').click();
  await expect(page.locator('#verdict')).not.toHaveClass(/unsupported/);
  await expect(page.locator('#verdict')).toContainText('Shown on request');

  await page.locator('[data-sample$="apple-scab-lab.jpg"]').click();
  await expect(page.locator('#verdict h2')).toHaveText('Apple scab');
});
