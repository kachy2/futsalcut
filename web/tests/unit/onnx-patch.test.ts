import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { initializerDims, reshapeInitializers } from '../../src/core/onnx-patch';

const model = new Uint8Array(readFileSync(new URL('../../public/models/dfine_n_coco.onnx', import.meta.url)));
const PROJECT = '/m/model/decoder/Concat_output_0'; // the 33 values the three `integral` MatMuls project the distribution bins with

describe('onnx patch (WebGPU needs this weight to be 2-D)', () => {
  it('adds a trailing axis to the weight and leaves every other initializer alone', () => {
    const before = initializerDims(model);
    expect(before.get(PROJECT)).toEqual([33]);
    const patched = reshapeInitializers(model, { [PROJECT]: [33, 1] })!;
    expect(patched).not.toBeNull();
    const after = initializerDims(patched);
    expect(after.get(PROJECT)).toEqual([33, 1]);
    expect(after.size).toBe(before.size);
    for (const [name, dims] of before) if (name !== PROJECT) expect(after.get(name)).toEqual(dims);
    expect(patched.length).toBe(model.length + 2); // one more dims entry; the enclosing lengths keep their size
  });
  it('keeps the start and the end of the file byte for byte', () => {
    const patched = reshapeInitializers(model, { [PROJECT]: [33, 1] })!;
    expect(Buffer.from(patched.subarray(0, 16))).toEqual(Buffer.from(model.subarray(0, 16))); // ir_version, producer
    expect(Buffer.from(patched.subarray(32, 4096))).toEqual(Buffer.from(model.subarray(32, 4096))); // after the graph's length prefix
    expect(Buffer.from(patched.subarray(patched.length - 4096))).toEqual(Buffer.from(model.subarray(model.length - 4096)));
  });
  it('does nothing for names the model does not have', () => {
    expect(reshapeInitializers(model, { nope: [1] })).toBeNull();
  });
});
