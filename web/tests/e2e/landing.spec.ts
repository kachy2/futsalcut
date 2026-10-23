// The English root and the static Korean page share one bundle but differ in language, title and canonical URL.
import { expect, test } from '@playwright/test';

test('English at the root, Korean at /ko/', async ({ page }) => {
  const app = process.env.APP_URL ? '' : '/'; // '' = the base URL itself (keeps a /<repo>/ path)
  const ko = process.env.APP_URL ? 'ko/' : '/ko/';
  await page.goto(app);
  await expect(page.locator('h1')).toHaveText('FutsalCut');
  await expect(page).toHaveTitle(/FutsalCut — keep only the play/);
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await page.goto(ko);
  await expect(page.locator('h1')).toHaveText('풋살컷');
  await expect(page).toHaveTitle(/풋살컷 FutsalCut/);
  await expect(page.locator('html')).toHaveAttribute('lang', 'ko');
  await expect(page.locator('link[rel=canonical]')).toHaveAttribute('href', /\/futsalcut\/ko\/$/);
});
