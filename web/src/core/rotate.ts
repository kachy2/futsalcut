/** Clockwise rotation of an 8-bit image by 0/90/180/270 degrees — twin of python sampler.rotate_cw. */
export function rotateCw(src: Uint8Array, w: number, h: number, degrees: number): { data: Uint8Array; w: number; h: number } {
  const k = (((degrees / 90) % 4) + 4) % 4;
  if (k === 0) return { data: src, w, h };
  if (k === 2) {
    const out = new Uint8Array(w * h);
    for (let i = 0, n = w * h; i < n; i++) out[i] = src[n - 1 - i];
    return { data: out, w, h };
  }
  const out = new Uint8Array(w * h); // output is h x w (width h, height w)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = src[y * w + x];
      if (k === 1) out[x * h + (h - 1 - y)] = v;      // 90 cw: (x, y) -> (h-1-y, x)
      else out[(w - 1 - x) * h + y] = v;              // 270 cw: (x, y) -> (y, w-1-x)
    }
  }
  return { data: out, w: h, h: w };
}
