// Micro-benchmark of pass-1 parts on the synthetic fixture (prints timings; always passes).
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { it } from 'vitest';
import { blobsOf, blockWindow, lowerMedian, majority3, pass1Sample } from '../../src/core/features';
import { polygonMask, type Point } from '../../src/core/roi';

const DIR = resolve(__dirname, '../fixtures/synthetic');
it.skipIf(!existsSync(resolve(DIR, 'luma.u8')))('pass-1 timing breakdown', () => {
  const fx = JSON.parse(readFileSync(resolve(DIR, 'fixture.json'), 'utf8'));
  const { n, W, H } = fx.meta;
  const raw = readFileSync(resolve(DIR, 'luma.u8'));
  const frame = (i: number) => new Uint8Array(raw.buffer, raw.byteOffset + i * W * H, W * H);
  const zone = polygonMask(fx.zone as Point[], W, H);
  let t = performance.now();
  const bgs: Uint8Array[] = [];
  for (let b = 0; b < Math.ceil(n / 10); b++) bgs.push(lowerMedian(blockWindow(n, b).map(frame), W * H));
  const tMed = performance.now() - t;
  t = performance.now();
  for (let i = 0; i < n; i++) pass1Sample(frame(i), bgs[Math.floor(i / 10)], i ? frame(i - 1) : null, zone, W, H);
  const tP1 = performance.now() - t;
  t = performance.now();
  const m = new Uint8Array(W * H).map((_, i) => (i % 7 === 0 ? 1 : 0));
  for (let i = 0; i < n; i++) majority3(m, W, H);
  const tMaj = performance.now() - t;
  t = performance.now();
  for (let i = 0; i < n; i++) blobsOf(majority3(m, W, H), zone, W, H);
  const tBlob = performance.now() - t;
  console.log(`per sample: lowerMedian ${(tMed / n).toFixed(1)} ms (amortized), pass1Sample ${(tP1 / n).toFixed(1)} ms, ` +
    `majority3 ${(tMaj / n).toFixed(1)} ms, blobs(dense) ${(tBlob / n).toFixed(1)} ms`);
}, 300_000);
