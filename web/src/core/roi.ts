/**
 * Regions of interest — twin of python/src/futsalcut/roi.py. Polygons are normalized [0,1]
 * coordinates in display orientation; a pixel is inside when its centre passes the even-odd test.
 */
export type Point = [number, number];

export function pointInPolygon(poly: Point[], x: number, y: number): boolean {
  let inside = false;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % n];
    if ((y1 > y) !== (y2 > y)) {
      const xc = ((x2 - x1) * (y - y1)) / (y2 - y1) + x1;
      if (x < xc) inside = !inside;
    }
  }
  return inside;
}

export function polygonMask(poly: Point[], width: number, height: number): Uint8Array {
  const m = new Uint8Array(width * height);
  for (let r = 0; r < height; r++) {
    const y = (r + 0.5) / height;
    for (let c = 0; c < width; c++) {
      if (pointInPolygon(poly, (c + 0.5) / width, y)) m[r * width + c] = 1;
    }
  }
  return m;
}

/** Sutherland–Hodgman clip keeping the part with y >= yMin (image y grows downwards). */
export function clipBelow(poly: Point[], yMin: number): Point[] {
  const out: Point[] = [];
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const cur = poly[i];
    const nxt = poly[(i + 1) % n];
    const cin = cur[1] >= yMin;
    const nin = nxt[1] >= yMin;
    if (cin) out.push(cur);
    if (cin !== nin) {
      const t = (yMin - cur[1]) / (nxt[1] - cur[1]);
      out.push([cur[0] + t * (nxt[0] - cur[0]), yMin]);
    }
  }
  return out;
}
