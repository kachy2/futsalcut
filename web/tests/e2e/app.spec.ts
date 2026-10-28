// End-to-end: select the synthetic recording -> default zone -> analyze -> review -> save (OPFS stand-in for
// the save dialog) -> the written MP4's duration equals the kept duration shown in the UI.
import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { analyze, launch, openApp, drawZone, type Zone } from './helpers';

// PROBE_WAIT_MIN: a software WebGPU adapter needs about ten minutes for the four inferences of the self-test
const PROBE_WAIT_MS = Number(process.env.PROBE_WAIT_MIN ?? 5) * 60_000;

test('analyze and export the synthetic recording', async ({ baseURL }) => {
  test.setTimeout(30 * 60_000);
  const { ctx, page, logs } = await launch(baseURL);
  await openApp(page);
  if (process.env.APP_ZONE) {
    await drawZone(page, JSON.parse(process.env.APP_ZONE) as Zone); // the given polygon, corner by corner
  } else {
    await page.getByRole('button', { name: /default/ }).click();
  }
  if (process.env.DETECTOR) await page.selectOption('#detector-backend', process.env.DETECTOR); // 'webgpu' exercises the fallback
  const { diag, hook: sc } = await analyze(page);
  console.log('diagnostics', JSON.stringify({ samples: diag.samples, xRealtime: diag.xRealtime, wallS: diag.wallS,
    formats: diag.frameFormats, exact: diag.exactPath, fallback: diag.fallbackPath, detector: diag.detectorBackend,
    asked: diag.detectorAsked, detectorCalls: diag.detectorCalls, detectorMs: diag.detectorMsPerCall, threads: diag.detectorThreads,
    isolated: diag.crossOriginIsolated, store: diag.sampleStore, gpuCheck: diag.detectorGpuCheck, webgpuFallback: diag.detectorFallbackReason,
    perspective: diag.perspective }));
  console.log('timing', JSON.stringify({ decodeS: diag.decodeS, lumaS: diag.lumaS, pass1S: diag.pass1S, thumbS: diag.thumbS,
    strideDetectorS: diag.strideDetectorS, featuresS: diag.featuresS, detectorS: diag.detectorS, redecodeS: diag.redecodeS,
    detectorSetupS: diag.detectorSetupS, lumaCopyS: diag.lumaCopyS, lumaShrinkS: diag.lumaShrinkS, frameToBoxesS: diag.frameToBoxesS }));
  expect(diag.detectorBackend).toMatch(/wasm|webgpu/); // the detector must always end up running
  if (process.env.EXPECT_BACKEND) expect(diag.detectorBackend).toBe(process.env.EXPECT_BACKEND);
  if (process.env.EXPECT_CHECK) expect(String(diag.detectorGpuCheck)).toMatch(new RegExp(process.env.EXPECT_CHECK));
  expect(diag.crossOriginIsolated).toBe(true); // the service worker made the page isolated (WASM threads)
  if (process.env.ORT_THREADS) expect(diag.detectorThreads).toBe(Number(process.env.ORT_THREADS));
  expect(diag.sampleStore).toMatch(/file|memory/); // the decoded samples are kept for a new zone
  expect(diag.samples).toBeGreaterThan(Number(process.env.MIN_SAMPLES ?? 100)); // MIN_SAMPLES for short excerpts
  expect(diag.env.cores).toBeGreaterThan(0); // environment block for reading speed numbers
  expect(diag.env.userAgent).toContain('Chrome');
  expect(diag.exactPath).toBe(diag.samples);
  const stat = (await page.locator('.stat').textContent())!.trim(); // m:ss of kept duration
  const [mm, ss] = stat.split(':').map(Number);
  const keptS = mm * 60 + ss;
  console.log('kept', stat, await page.locator('.stat + span').textContent());
  expect(keptS).toBeGreaterThan(0);
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
  await expect(page.locator('#probe-result')).not.toHaveText('…', { timeout: PROBE_WAIT_MS });
  console.log('detector probe:', await page.locator('#probe-result').textContent());
  if (process.env.DETECTOR === 'webgpu') {
    await page.locator('#probe-webgpu').click();
    await expect(page.locator('#probe-result')).not.toHaveText('…', { timeout: PROBE_WAIT_MS });
    console.log('webgpu probe:', await page.locator('#probe-result').textContent());
  }
  if (logs.some((l) => /^\[(pageerror|warning|error)\]/.test(l))) console.log(logs.join('\n'));
  await ctx.close();
});
