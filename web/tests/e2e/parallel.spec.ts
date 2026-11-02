// The helper workers (detector, luma shrinking and thumbnails, pass 1) must not change a single number: the same recording analyzed
// with them and with everything in the analysis worker (?parallel=0) gives identical probabilities, detector scores and scenes.
// A third run gives the detector two WASM threads, which are started from inside the detector's own worker. A second test sends
// every sample to the detector (?band=0,1) once with one detector and once with a helping CPU detector (?dualdet=1): the cascade
// is shared between them and the result stays the same.
import { expect, test } from '@playwright/test';
import { analyze, launch, openApp } from './helpers';

interface Run {
  diag: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  hook: { intervals: [number, number][]; prob: number[]; light: number[]; detector: [number, number][]; thumbs: number };
}

async function run(baseURL: string | undefined, query: string): Promise<Run> {
  const { ctx, page } = await launch(baseURL);
  await openApp(page, query);
  await page.getByRole('button', { name: /default/ }).click();
  const r = await analyze(page);
  const d = r.diag;
  console.log(query || '(default)', JSON.stringify({ wallS: d.wallS, decodeS: d.decodeS, lumaS: d.lumaS, lumaCopyS: d.lumaCopyS,
    lumaShrinkS: d.lumaShrinkS, lumaWaitS: d.lumaWaitS, pass1S: d.pass1S, thumbS: d.thumbS, sideWaitS: d.sideWaitS, tailWaitS: d.tailWaitS,
    strideDetectorS: d.strideDetectorS, detectorS: d.detectorS, detectorMs: d.detectorMsPerCall, calls: d.detectorCalls,
    split: d.detectorSplit, second: d.secondDetector, secondMs: d.secondDetectorMsPerCall, threads: d.detectorThreads,
    workers: [d.detectorWorker, d.sideWorker, d.lumaWorker], detectorIsolated: d.detectorIsolated, unsureShare: d.unsureShare,
    lightHistogram: d.lightHistogram, thumbs: r.hook.thumbs }));
  expect(r.hook.thumbs).toBe(d.samples); // every sample has its preview, from whichever worker made it
  expect(d.zone.length).toBeGreaterThanOrEqual(3);
  expect(d.lightHistogram.reduce((a: number, b: number) => a + b, 0)).toBe(d.samples);
  await ctx.close();
  return r;
}

const same = (a: Run, b: Run) => {
  expect(b.diag.detectorCalls).toBe(a.diag.detectorCalls);
  expect(b.hook.prob).toEqual(a.hook.prob);
  expect(b.hook.light).toEqual(a.hook.light);
  expect(b.hook.detector).toEqual(a.hook.detector);
  expect(b.hook.intervals).toEqual(a.hook.intervals);
  expect(b.diag.perspective).toEqual(a.diag.perspective);
  expect(b.diag.samples).toBe(a.diag.samples);
};

test('helper workers give the same result as the analysis worker alone, also with detector threads', async ({ baseURL }) => {
  test.setTimeout(30 * 60_000);
  const alone = await run(baseURL, '?parallel=0');
  expect([alone.diag.detectorWorker, alone.diag.sideWorker, alone.diag.lumaWorker]).toEqual([false, false, false]);

  const helped = await run(baseURL, '');
  expect([helped.diag.detectorWorker, helped.diag.sideWorker, helped.diag.lumaWorker]).toEqual([true, true, true]);
  same(alone, helped);

  const threaded = await run(baseURL, '?threads=2');
  expect(threaded.diag.detectorWorker).toBe(true);
  expect(threaded.diag.detectorThreads).toBe(2);
  expect(threaded.diag.detectorIsolated).toBe(true);
  expect(threaded.hook.intervals).toEqual(alone.hook.intervals);
  expect(threaded.hook.detector.length).toBe(alone.hook.detector.length);
});

test('a helping CPU detector shares the cascade and gives the same result', async ({ baseURL }) => {
  test.setTimeout(40 * 60_000);
  const one = await run(baseURL, '?band=0,1');
  expect(one.diag.unsureShare).toBeGreaterThan(0.9); // all but samples whose probability rounds to exactly 0 or 1
  expect(one.diag.detectorSplit.secondary).toBe(0);
  const two = await run(baseURL, '?band=0,1&dualdet=1');
  expect(two.diag.secondDetector).toMatch(/wasm/);
  expect(two.diag.detectorSplit.secondary).toBeGreaterThan(0);
  expect(two.diag.detectorSplit.primary + two.diag.detectorSplit.secondary).toBe(one.diag.detectorSplit.primary);
  same(one, two);
});
