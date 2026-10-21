// Functional check of the single-file WebCodecs benchmark in headless Google Chrome (software decode).
import { expect, test } from '@playwright/test';
import path from 'node:path';

const BENCH = path.resolve('dist-bench/index.html');
const VIDEO = process.env.BENCH_VIDEO ?? '/data/synthetic/syn_h264_gop1.mp4';

test('bench runs end to end and reports keyframe throughput', async ({ page }) => {
  test.setTimeout(10 * 60_000);
  page.on('console', (m) => console.log('[page]', m.text()));
  await page.goto('file://' + BENCH);
  await page.setInputFiles('#file', VIDEO);
  await page.fill('#seconds', process.env.BENCH_SECONDS ?? '60');
  await page.click('#run');
  await expect(page.locator('#copy')).toBeEnabled({ timeout: 9 * 60_000 });
  const log = await page.locator('#log').textContent();
  console.log(log);
  expect(log).not.toContain('오류');
  expect(log).toContain('"keyframes":');
});
