// The helper workers (detector, pass 1 and thumbnails) must not change a single number: the same recording analyzed with them and
// with everything in the analysis worker (?parallel=0) gives identical probabilities, detector scores and scenes. A third run
// gives the detector two WASM threads, which are started from inside the detector's own worker.
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
  console.log(query || '(default)', JSON.stringify({ wallS: r.diag.wallS, decodeS: r.diag.decodeS, lumaS: r.diag.lumaS, pass1S: r.diag.pass1S,
    thumbS: r.diag.thumbS, sideWaitS: r.diag.sideWaitS, tailWaitS: r.diag.tailWaitS, strideDetectorS: r.diag.strideDetectorS,
    detectorS: r.diag.detectorS, detectorMs: r.diag.detectorMsPerCall, calls: r.diag.detectorCalls, threads: r.diag.detectorThreads,
    detectorWorker: r.diag.detectorWorker, sideWorker: r.diag.sideWorker, detectorIsolated: r.diag.detectorIsolated, thumbs: r.hook.thumbs }));
  expect(r.hook.thumbs).toBe(r.diag.samples); // every sample has its preview, from whichever worker made it
  await ctx.close();
  return r;
}

test('helper workers give the same result as the analysis worker alone, also with detector threads', async ({ baseURL }) => {
  test.setTimeout(30 * 60_000);
  const alone = await run(baseURL, '?parallel=0');
  expect([alone.diag.detectorWorker, alone.diag.sideWorker]).toEqual([false, false]);

  const helped = await run(baseURL, '');
  expect([helped.diag.detectorWorker, helped.diag.sideWorker]).toEqual([true, true]);
  expect(helped.diag.detectorCalls).toBe(alone.diag.detectorCalls);
  expect(helped.hook.prob).toEqual(alone.hook.prob);
  expect(helped.hook.light).toEqual(alone.hook.light);
  expect(helped.hook.detector).toEqual(alone.hook.detector);
  expect(helped.hook.intervals).toEqual(alone.hook.intervals);
  expect(helped.diag.perspective).toEqual(alone.diag.perspective);
  expect(helped.diag.samples).toBe(alone.diag.samples);

  const threaded = await run(baseURL, '?threads=2');
  expect(threaded.diag.detectorWorker).toBe(true);
  expect(threaded.diag.detectorThreads).toBe(2);
  expect(threaded.diag.detectorIsolated).toBe(true);
  expect(threaded.hook.intervals).toEqual(alone.hook.intervals);
  expect(threaded.hook.detector.length).toBe(alone.hook.detector.length);
});
