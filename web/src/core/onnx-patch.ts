/**
 * Byte-level edit of an ONNX file (a protobuf): give named initializers new dims and copy everything else unchanged.
 * ModelProto.graph (field 7) -> GraphProto.initializer (field 5) -> TensorProto.name (8) / dims (1).
 */
interface Field { num: number; wire: number; from: number; to: number; bodyFrom: number; bodyTo: number }

function readVarint(buf: Uint8Array, pos: number): [number, number] {
  let value = 0;
  let scale = 1;
  for (;;) {
    const b = buf[pos++];
    value += (b & 0x7f) * scale;
    if (b < 0x80) return [value, pos];
    scale *= 128;
  }
}

function* fieldsOf(buf: Uint8Array, from: number, to: number): Generator<Field> {
  let pos = from;
  while (pos < to) {
    const start = pos;
    const [tag, afterTag] = readVarint(buf, pos);
    const wire = tag & 7;
    let bodyFrom = afterTag;
    let bodyTo: number;
    if (wire === 0) bodyTo = readVarint(buf, afterTag)[1];
    else if (wire === 1) bodyTo = afterTag + 8;
    else if (wire === 5) bodyTo = afterTag + 4;
    else if (wire === 2) {
      const [len, afterLen] = readVarint(buf, afterTag);
      bodyFrom = afterLen;
      bodyTo = afterLen + len;
    } else throw new Error(`unsupported protobuf wire type ${wire}`);
    yield { num: Math.floor(tag / 8), wire, from: start, to: bodyTo, bodyFrom, bodyTo };
    pos = bodyTo;
  }
}

function varint(n: number): number[] {
  const out: number[] = [];
  while (n >= 128) {
    out.push((n % 128) | 0x80);
    n = Math.floor(n / 128);
  }
  out.push(n);
  return out;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

const lenField = (num: number, payload: Uint8Array) => concat([Uint8Array.from([...varint(num * 8 + 2), ...varint(payload.length)]), payload]);

function tensorName(buf: Uint8Array, from: number, to: number): string | null {
  for (const f of fieldsOf(buf, from, to)) if (f.num === 8 && f.wire === 2) return new TextDecoder().decode(buf.subarray(f.bodyFrom, f.bodyTo));
  return null;
}

function withDims(buf: Uint8Array, from: number, to: number, dims: number[]): Uint8Array {
  const parts: Uint8Array[] = [];
  let placed = false;
  for (const f of fieldsOf(buf, from, to)) {
    if (f.num !== 1) { parts.push(buf.subarray(f.from, f.to)); continue; } // not a dim: unchanged
    if (!placed) { // the old dims (packed or one entry each) are replaced where the first one stood
      for (const d of dims) parts.push(Uint8Array.from([...varint(8), ...varint(d)]));
      placed = true;
    }
  }
  return concat(placed ? parts : [...dims.map((d) => Uint8Array.from([...varint(8), ...varint(d)])), ...parts]);
}

/** The initializers of the model with their dims, for checks. */
export function initializerDims(model: Uint8Array): Map<string, number[]> {
  const out = new Map<string, number[]>();
  for (const f of fieldsOf(model, 0, model.length)) {
    if (f.num !== 7 || f.wire !== 2) continue;
    for (const g of fieldsOf(model, f.bodyFrom, f.bodyTo)) {
      if (g.num !== 5 || g.wire !== 2) continue;
      const dims: number[] = [];
      for (const t of fieldsOf(model, g.bodyFrom, g.bodyTo)) {
        if (t.num !== 1) continue;
        if (t.wire === 0) dims.push(readVarint(model, t.bodyFrom)[0]);
        else for (let p = t.bodyFrom; p < t.bodyTo;) { const [d, n] = readVarint(model, p); dims.push(d); p = n; }
      }
      const name = tensorName(model, g.bodyFrom, g.bodyTo);
      if (name !== null) out.set(name, dims);
    }
  }
  return out;
}

/** A copy of the model in which the named initializers have the given dims (data untouched), or null if none of them exists. */
export function reshapeInitializers(model: Uint8Array, newDims: Record<string, number[]>): Uint8Array | null {
  let changed = 0;
  const top: Uint8Array[] = [];
  for (const f of fieldsOf(model, 0, model.length)) {
    if (f.num !== 7 || f.wire !== 2) { top.push(model.subarray(f.from, f.to)); continue; }
    const graph: Uint8Array[] = [];
    for (const g of fieldsOf(model, f.bodyFrom, f.bodyTo)) {
      const name = g.num === 5 && g.wire === 2 ? tensorName(model, g.bodyFrom, g.bodyTo) : null;
      if (name !== null && name in newDims) {
        graph.push(lenField(5, withDims(model, g.bodyFrom, g.bodyTo, newDims[name])));
        changed++;
      } else graph.push(model.subarray(g.from, g.to));
    }
    top.push(lenField(7, concat(graph)));
  }
  return changed ? concat(top) : null;
}
