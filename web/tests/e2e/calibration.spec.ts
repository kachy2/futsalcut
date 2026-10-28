// Quick calibration: "Apply" takes effect at once (mode and slider change, a note says what was chosen), Undo returns to Auto,
// and a new analysis starts from Auto again because the calibration belonged to the old result.
import { expect, test, type Page } from '@playwright/test';
import { analyze, DRAW, launch, openApp } from './helpers';

interface Hook { t: number[]; prob: number[]; threshold: number | null }
const hook = (page: Page) => page.evaluate(() => (window as unknown as { __futsalcut: Hook }).__futsalcut);

/** Open the calibration, answer every scene the way the app's own probabilities see it, press Apply. */
async function calibrate(page: Page): Promise<void> {
  const h = await hook(page);
  const title = /빠른 보정|Quick calibration/;
  await page.getByRole('button', { name: title }).click();
  const card = page.locator('.card', { has: page.getByRole('heading', { name: title }) });
  const items = card.locator('div[style*="grid-template-columns"] > div');
  const n = await items.count();
  expect(n).toBeGreaterThan(2);
  await expect(card.getByRole('button', { name: /^(적용|Apply)$/ })).toBeDisabled(); // nothing answered yet
  for (let k = 0; k < n; k++) {
    const item = items.nth(k);
    const [m, s] = ((await item.locator('span').first().textContent()) ?? '0:00').split(':').map(Number);
    const i = h.t.findIndex((v) => Math.floor(v) === m * 60 + s);
    expect(i).toBeGreaterThanOrEqual(0);
    const keep = h.prob[i] >= (h.threshold ?? 0.5); // a threshold that separates these answers exists
    await item.getByRole('button', { name: keep ? /^(남김|Keep)$/ : /^(제거|Drop)$/ }).click();
  }
  await card.getByRole('button', { name: /^(적용|Apply)$/ }).click();
  await expect(card).toHaveCount(0); // the card closes
}

test('quick calibration is applied at once, can be undone and does not outlive its result', async ({ baseURL }) => {
  test.setTimeout(20 * 60_000);
  const { ctx, page } = await launch(baseURL);
  await openApp(page);
  await page.getByRole('button', { name: /default/ }).click();
  await analyze(page);
  const mode = page.locator('select').first();
  const note = page.locator('#cal-note');
  await expect(mode).toHaveValue('auto');
  await expect(note).toHaveCount(0);

  await calibrate(page);
  await expect(mode).toHaveValue('sens');
  await expect(note).toBeVisible();
  const text = (await note.textContent()) ?? '';
  const m = /agrees with (\d+) of your (\d+) answers/.exec(text) ?? /답 (\d+)개 중 (\d+)개/.exec(text);
  expect(m, text).not.toBeNull();
  expect(Number(m![1])).toBeGreaterThan(2);
  const [agree, total] = /agrees/.test(text) ? [Number(m![1]), Number(m![2])] : [Number(m![2]), Number(m![1])];
  expect(agree).toBe(total); // the answers came from the app's own probabilities: a threshold fits all of them

  await note.getByRole('button', { name: /Undo|되돌리기/ }).click();
  await expect(note).toHaveCount(0);
  await expect(mode).toHaveValue('auto');

  // a calibration that is still in force is dropped when the zone is redrawn and analyzed again
  await calibrate(page);
  await expect(mode).toHaveValue('sens');
  await page.locator('#redraw').click();
  await expect(page.getByRole('heading', { name: DRAW })).toBeVisible();
  await analyze(page);
  await expect(mode).toHaveValue('auto');
  await expect(note).toHaveCount(0);
  await ctx.close();
});
