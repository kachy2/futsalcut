// Exporter on the synthetic set (Node: Blob sources + a file-backed WritableStream). The outputs are then
// checked by the Python verifier in the futsalcut-py container (tools: eval/runners/p05_remux_test.py --verify-prefix).
import { openAsBlob } from 'node:fs';
import { open } from 'node:fs/promises';
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { exportLossless, type ExportSegment } from '../../src/core/exporter';

const SYN = process.env.SYN_DIR ?? '/data/synthetic';
const OUT = process.env.EXPORT_OUT ?? '/data/outputs/p05';
const ok = existsSync(resolve(SYN, 'syn_h264_gop1.mp4'));

async function fileWritable(path: string): Promise<WritableStream<{ type: 'write'; data: Uint8Array; position: number }>> {
  const fh = await open(path, 'w');
  return new WritableStream({
    async write(c) { await fh.write(c.data, 0, c.data.byteLength, c.position); },
    async close() { await fh.close(); },
  });
}

const CASES: Record<string, [string[], [number, number, number][]]> = {
  h264_gop1: [['syn_h264_gop1.mp4'], [[0, 3, 9], [0, 20, 27], [0, 40, 52], [0, 70, 75], [0, 100, 112]]],
  hevc_opengop: [['syn_hevc_opengop.mp4'], [[0, 2, 6], [0, 9, 13], [0, 18, 22], [0, 27, 31], [0, 35, 40]]],
  rot90: [['syn_rot90.mp4'], [[0, 2, 6], [0, 9, 13], [0, 18, 22], [0, 27, 31], [0, 35, 40]]],
  split_pair: [['syn_split_a.mp4', 'syn_split_b.mp4'], [[0, 10, 20], [0, 50, 60], [1, 0, 8], [1, 30, 36], [1, 50, 60]]],
};

describe.skipIf(!ok)('exportLossless (synthetic set)', () => {
  mkdirSync(OUT, { recursive: true });
  for (const [name, [srcs, segs]] of Object.entries(CASES)) {
    it(name, async () => {
      const blobs = await Promise.all(srcs.map((s) => openAsBlob(resolve(SYN, s))));
      const segments: ExportSegment[] = segs.map(([file, start, end]) => ({ file, start, end }));
      const rep = await exportLossless(blobs, segments, await fileWritable(resolve(OUT, `web_export_${name}.mp4`)));
      const expected = segs.reduce((s, [, a, b]) => s + (b - a), 0);
      expect(Math.abs(rep.videoDurationS - expected)).toBeLessThan(0.05);
      expect(Math.abs(rep.audioDurationS - rep.videoDurationS)).toBeLessThan(0.025);
      expect(rep.maxCutAvErrorS).toBeLessThan(0.0215);
      if (name === 'hevc_opengop') expect(rep.eosInserted).toBe(4);
    }, 120_000);
  }
});
