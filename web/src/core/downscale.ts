/**
 * Integer box downscale of an 8-bit plane — bit-exact twin of `box_downscale` in
 * python/src/futsalcut/sampler.py. Output (r, c) averages rows [r*H/h, (r+1)*H/h) and
 * cols [c*W/w, (c+1)*W/w) (floored), rounding half up.
 */
export function boxDownscale(
  src: Uint8Array | Uint8ClampedArray,
  srcW: number,
  srcH: number,
  srcStride: number,
  outW: number,
  outH: number,
  out: Uint8Array = new Uint8Array(outW * outH),
): Uint8Array {
  if (outW > srcW || outH > srcH) throw new Error('boxDownscale only shrinks');
  const cb = new Int32Array(outW + 1);
  for (let c = 0; c <= outW; c++) cb[c] = Math.floor((c * srcW) / outW);
  const acc = new Uint32Array(outW);
  for (let r = 0; r < outH; r++) {
    const r0 = Math.floor((r * srcH) / outH);
    const r1 = Math.floor(((r + 1) * srcH) / outH);
    acc.fill(0);
    for (let y = r0; y < r1; y++) {
      const base = y * srcStride;
      for (let c = 0; c < outW; c++) {
        let s = 0;
        const x1 = base + cb[c + 1];
        for (let x = base + cb[c]; x < x1; x++) s += src[x];
        acc[c] += s;
      }
    }
    const rh = r1 - r0;
    const row = r * outW;
    for (let c = 0; c < outW; c++) {
      const n = rh * (cb[c + 1] - cb[c]);
      out[row + c] = Math.floor((acc[c] + (n >> 1)) / n);
    }
  }
  return out;
}

/** Output size keeping aspect ratio: long side = `longSide`, both sides even (twin of `target_size`). */
export function targetSize(width: number, height: number, longSide = 640): [number, number] {
  if (width >= height) return [longSide, Math.round((height * longSide) / width / 2) * 2];
  return [Math.round((width * longSide) / height / 2) * 2, longSide];
}
