// The service worker makes the page cross-origin isolated (WASM threads). A forced refresh (Ctrl+F5) loads the page without the
// worker; the app has to put itself back under it, on every forced refresh and also when the worker is registered during one.
import { expect, test } from '@playwright/test';
import { launch } from './helpers';

test('isolation is restored after forced refreshes', async ({ baseURL }) => {
  test.setTimeout(3 * 60_000);
  const { ctx, page } = await launch(baseURL);
  const mark = () => page.evaluate(() => { (window as unknown as { __old?: boolean }).__old = true; });
  // a document that is new (the marker of the old one is gone) and isolated
  const freshAndIsolated = () => page.evaluate(() => !(window as unknown as { __old?: boolean }).__old && crossOriginIsolated).catch(() => false);

  await page.goto('/');
  await expect.poll(() => page.evaluate(() => crossOriginIsolated).catch(() => false), { timeout: 60_000 }).toBe(true);
  const cdp = await ctx.newCDPSession(page);

  for (let i = 0; i < 4; i++) {
    await mark();
    await cdp.send('Page.reload', { ignoreCache: true }); // what Ctrl+F5 does: the document is not served by the worker
    await expect.poll(freshAndIsolated, { timeout: 20_000, message: `forced refresh ${i + 1}` }).toBe(true);
  }

  // the worker is registered while the refresh is forced (a visitor's first load of this version)
  await page.evaluate(async () => { for (const r of await navigator.serviceWorker.getRegistrations()) await r.unregister(); });
  await mark();
  await cdp.send('Page.reload', { ignoreCache: true });
  await expect.poll(freshAndIsolated, { timeout: 20_000, message: 'forced refresh without a registered worker' }).toBe(true);
  await ctx.close();
});
