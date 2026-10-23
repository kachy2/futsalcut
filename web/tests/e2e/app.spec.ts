// End-to-end: select the synthetic recording -> default zone -> analyze -> review -> save (OPFS stand-in for
// the save dialog) -> the written MP4's duration equals the kept duration shown in the UI.
import { chromium, expect, test } from '@playwright/test';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const VIDEO = process.env.APP_VIDEO ?? '/data/synthetic/syn_h264_gop1.mp4';

test('analyze and export the synthetic recording', async ({ baseURL }) => {
  test.setTimeout(30 * 60_000);
  // persistent profile: the default (incognito-like) context has a tiny storage quota for the OPFS stand-in
  const ctx = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), 'futsalcut-e2e-')), { channel: 'chrome', headless: true, baseURL });
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await page.addInitScript(() => {
    // headless runs cannot click through the native save dialog: write into the origin private file system
    (window as unknown as { showSaveFilePicker: unknown }).showSaveFilePicker = async () => {
      const root = await navigator.storage.getDirectory();
      return root.getFileHandle('e2e_output.mp4', { create: true });
    };
  });
  await page.goto(process.env.APP_URL ? '' : '/'); // '' = the base URL itself (keeps a /<repo>/ path)
  await page.setInputFiles('input[type=file]', VIDEO);
  await page.getByRole('button', { name: /다음|Next/ }).click();
  await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 });
  if (process.env.APP_ZONE) {
    // draw the given polygon like a user: click each corner on the preview canvas
    const zonePts: [number, number][] = JSON.parse(process.env.APP_ZONE);
    const box = (await page.locator('canvas').boundingBox())!;
    for (const [x, y] of zonePts) {
      await page.mouse.click(box.x + Math.min(0.999, Math.max(0.001, x)) * box.width, box.y + Math.min(0.999, Math.max(0.001, y)) * box.height);
    }
  } else {
    await page.getByRole('button', { name: /default/ }).click();
  }
  await page.getByRole('button', { name: /분석 시작|Analyze/ }).click();
  await expect(page.getByRole('heading', { name: /결과 확인|Review/ })).toBeVisible({ timeout: 15 * 60_000 });
  const diag = JSON.parse((await page.locator('details pre').textContent()) ?? '{}');
  console.log('diagnostics', JSON.stringify({ samples: diag.samples, xRealtime: diag.xRealtime, wallS: diag.wallS,
    formats: diag.frameFormats, exact: diag.exactPath, fallback: diag.fallbackPath, detector: diag.detectorBackend,
    detectorCalls: diag.detectorCalls, detectorMs: diag.detectorMsPerCall, perspective: diag.perspective }));
  expect(diag.samples).toBeGreaterThan(100);
  expect(diag.exactPath).toBe(diag.samples);
  const stat = (await page.locator('.stat').textContent())!.trim(); // m:ss of kept duration
  const [mm, ss] = stat.split(':').map(Number);
  const keptS = mm * 60 + ss;
  console.log('kept', stat, await page.locator('.stat + span').textContent());
  expect(keptS).toBeGreaterThan(0);
  const sc = await page.evaluate(() => (window as unknown as { __futsalcut: { intervals: [number, number][]; detector: [number, number][] } }).__futsalcut);
  console.log('intervals', JSON.stringify(sc.intervals.map(([a, b]) => [Math.round(a), Math.round(b)])), 'detector samples', sc.detector.length);
  if (process.env.TRUTH) {
    // synthetic ground truth: per-second keep flags; report recall/precision of kept seconds
    const truth: { t: number; keep: number }[] = JSON.parse(readFileSync(process.env.TRUTH, 'utf8')).per_second;
    const kept = (s: number) => sc.intervals.some(([a, b]) => s + 0.5 >= a && s + 0.5 < b);
    const tp = truth.filter((r) => r.keep && kept(r.t)).length;
    const pos = truth.filter((r) => r.keep).length;
    const pred = truth.filter((r) => kept(r.t)).length;
    console.log(`synthetic truth: recall ${(tp / pos).toFixed(2)} precision(raw, padding counts as FP) ${(tp / Math.max(1, pred)).toFixed(2)}`);
    expect(tp / pos).toBeGreaterThan(0.8);
  }
  await page.getByRole('button', { name: /MP4로 저장|Save as MP4/ }).click();
  await expect(page.getByText(/저장 완료|Saved/)).toBeVisible({ timeout: 5 * 60_000 });
  const saved = await page.locator('text=/저장 완료|Saved/').first().textContent();
  console.log('saved:', saved);
  const m = saved!.match(/(\d+):(\d\d)/);
  expect(m).not.toBeNull();
  expect(Math.abs(Number(m![1]) * 60 + Number(m![2]) - keptS)).toBeLessThanOrEqual(1);
  // detector self-test from the diagnostics panel
  await page.locator('details summary').click();
  await page.locator('#probe').click();
  await expect(page.locator('#probe-result')).not.toHaveText('…', { timeout: 5 * 60_000 });
  console.log('detector probe:', await page.locator('#probe-result').textContent());
  if (logs.some((l) => /^\[(pageerror|warning|error)\]/.test(l))) console.log(logs.join('\n'));
  await ctx.close();
});
