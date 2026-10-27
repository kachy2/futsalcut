// The English root and the static Korean page share one bundle but differ in language, title and canonical URL.
import { expect, test } from '@playwright/test';

test('English at the root, Korean at /ko/', async ({ page }) => {
  const app = process.env.APP_URL ? '' : '/'; // '' = the base URL itself (keeps a /<repo>/ path)
  const ko = process.env.APP_URL ? 'ko/' : '/ko/';
  const modelOk = (p: typeof page) => p.waitForResponse((r) => r.url().endsWith('/models/ensemble_v1.json'));
  let model = modelOk(page);
  await page.goto(app);
  expect((await model).status()).toBe(200);
  await expect(page.locator('h1')).toHaveText('FutsalCut');
  await expect(page).toHaveTitle(/FutsalCut — keep only the play/);
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  model = modelOk(page);
  await page.goto(ko);
  expect((await model).status()).toBe(200); // the shared bundle must find its model files from /ko/ as well
  await expect(page.locator('h1')).toHaveText('풋살컷');
  await expect(page).toHaveTitle(/풋살컷 FutsalCut/);
  await expect(page.locator('html')).toHaveAttribute('lang', 'ko');
  await expect(page.locator('link[rel=canonical]')).toHaveAttribute('href', /\/futsalcut\/ko\/$/);
  await expect(page.locator('.card.error')).toHaveCount(0);
});
