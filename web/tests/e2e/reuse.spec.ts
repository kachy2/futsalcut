// Redrawing the zone: the decoded samples and detector boxes of the first run are reused, and the result equals a fresh analysis.
// Also: Cancel returns to the zone editor instead of leaving a dead end.
import { expect, test } from '@playwright/test';
import { analyze, DRAW, drawZone, launch, openApp, REVIEW, type Zone } from './helpers';

const ZONE_B: Zone = [[0.08, 0.3], [0.92, 0.3], [0.97, 0.97], [0.03, 0.97]];

test('a new zone reuses the stored analysis and matches a fresh run', async ({ baseURL }) => {
  test.setTimeout(40 * 60_000);
  const first = await launch(baseURL);
  const p = first.page;
  await openApp(p);
  await p.getByRole('button', { name: /default/ }).click();
  const a = await analyze(p);
  expect(a.diag.reused).toBe(false);

  // peek into the zone editor and come back without analyzing again
  await p.locator('#redraw').click();
  await expect(p.getByRole('heading', { name: DRAW })).toBeVisible();
  await expect(p.getByText(/stored frames are reused|저장해 둔 자료를 재사용/)).toBeVisible();
  await p.locator('#back-to-result').click();
  await expect(p.getByRole('heading', { name: REVIEW })).toBeVisible();
  expect(await p.evaluate(() => (window as unknown as { __futsalcut: { prob: number[] } }).__futsalcut.prob)).toEqual(a.hook.prob);

  // another zone, from the stored samples
  await p.locator('#redraw').click();
  await drawZone(p, ZONE_B);
  const b = await analyze(p);
  console.log('reuse run', JSON.stringify({ wallS: b.diag.wallS, decodeS: b.diag.decodeS, replayS: b.diag.replayS, detectorCalls: b.diag.detectorCalls,
    firstRunWallS: a.diag.wallS, firstRunDetectorCalls: a.diag.detectorCalls }));
  expect(b.diag.reused).toBe(true);
  expect(b.diag.decodeS).toBe(0);
  expect(b.diag.samples).toBe(a.diag.samples);
  expect(b.diag.wallS).toBeLessThan(a.diag.wallS);
  expect(b.hook.prob).not.toEqual(a.hook.prob); // the zone matters
  await first.ctx.close();

  // a fresh page with the same zone must give the same numbers
  const second = await launch(baseURL);
  await openApp(second.page);
  await drawZone(second.page, ZONE_B);
  const c = await analyze(second.page);
  expect(c.diag.reused).toBe(false);
  expect(c.hook.prob.length).toBe(b.hook.prob.length);
  expect(c.hook.prob).toEqual(b.hook.prob);
  expect(c.hook.intervals).toEqual(b.hook.intervals);
  expect(c.hook.detector).toEqual(b.hook.detector);
  await second.ctx.close();
});

test('cancel returns to the zone editor and the analysis can be started again', async ({ baseURL }) => {
  test.setTimeout(15 * 60_000);
  const { ctx, page } = await launch(baseURL);
  await openApp(page);
  await page.getByRole('button', { name: /default/ }).click();
  await page.getByRole('button', { name: /분석 시작|Analyze/ }).click();
  await page.getByRole('button', { name: /취소|Cancel/ }).click();
  await expect(page.getByRole('heading', { name: DRAW })).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('.card.error')).toHaveCount(0);
  await expect(page.locator('canvas')).toBeVisible();
  const { diag } = await analyze(page); // the zone is still there: analyze again from the editor
  expect(diag.samples).toBeGreaterThan(100);
  await ctx.close();
});
