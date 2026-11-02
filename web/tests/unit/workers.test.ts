// The client side of the two helper workers, with a fake Worker: what happens when a worker cannot start, stays silent, fails, or is
// slower than the caller (the real workers are covered by tests/e2e/parallel.spec.ts).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPipelinedDetector } from '../../src/worker/detector-client';
import { openSide } from '../../src/worker/side-client';

vi.mock('../../src/worker/detector', () => ({
  createDetector: vi.fn(async () => ({
    backend: 'in-thread', fallbackReason: null, gpuCheck: null, threads: 1,
    detect: async () => [[0.1, 0.1, 0.2, 0.2, 0.9]], release: async () => undefined,
  })),
}));

type Msg = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

class FakeWorker {
  static all: FakeWorker[] = [];
  static onCreate: (w: FakeWorker) => void = () => undefined;
  static onMessage: (w: FakeWorker, m: Msg) => void = () => undefined;
  onmessage: ((ev: { data: Msg }) => void) | null = null;
  onerror: ((ev: { message?: string }) => void) | null = null;
  terminated = false;
  sent: { msg: Msg; transfer?: unknown[] }[] = [];

  constructor(public url: unknown, public options: unknown) {
    FakeWorker.all.push(this);
    FakeWorker.onCreate(this);
  }
  postMessage(msg: Msg, transfer?: unknown[]): void {
    this.sent.push({ msg, transfer });
    FakeWorker.onMessage(this, msg);
  }
  terminate(): void { this.terminated = true; }
  /** a message from the worker, delivered asynchronously like a real one */
  say(data: Msg): void { setTimeout(() => this.onmessage?.({ data }), 0); }
  fail(message: string): void { setTimeout(() => this.onerror?.({ message }), 0); }
}

const frame = () => ({ close: () => undefined }) as unknown as VideoFrame;
const luma = () => new Uint8Array(4);

beforeEach(() => {
  FakeWorker.all = [];
  FakeWorker.onCreate = () => undefined;
  FakeWorker.onMessage = () => undefined;
  vi.stubGlobal('Worker', FakeWorker);
  vi.stubGlobal('self', { crossOriginIsolated: false });
  vi.stubGlobal('OffscreenCanvas', class { getContext() { return {}; } });
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('openSide (pass 1 and thumbnails)', () => {
  it('works in the calling thread when asked not to use a helper, without starting a worker', async () => {
    const side = await openSide(false, () => undefined);
    expect(side.remote).toBe(false);
    expect(FakeWorker.all).toHaveLength(0);
  });

  it('falls back to the calling thread when the worker cannot be created, fails to load, or stays silent', async () => {
    FakeWorker.onCreate = () => { throw new Error('no workers here'); };
    expect((await openSide(true, () => undefined)).remote).toBe(false);

    FakeWorker.all = [];
    FakeWorker.onCreate = (w) => w.fail('script failed');
    expect((await openSide(true, () => undefined)).remote).toBe(false);
    expect(FakeWorker.all[0].terminated).toBe(true);

    FakeWorker.all = [];
    FakeWorker.onCreate = () => undefined; // never answers the ping
    vi.useFakeTimers();
    const pending = openSide(true, () => undefined);
    await vi.advanceTimersByTimeAsync(10_000);
    expect((await pending).remote).toBe(false);
    expect(FakeWorker.all[0].terminated).toBe(true);
  });

  it('uses the worker once it answers the ping, and holds the caller back when the worker is 3 samples behind', async () => {
    FakeWorker.onMessage = (w, m) => { if (m.type === 'ping') w.say({ type: 'pong' }); };
    const side = await openSide(true, () => undefined);
    expect(side.remote).toBe(true);
    const w = FakeWorker.all[0];
    side.start(8, 4, [[0, 0], [1, 0], [1, 1]]);
    expect(w.sent.map((s) => s.msg.type)).toEqual(['ping', 'start']);

    for (let i = 0; i < 3; i++) await side.push(luma(), frame(), 0); // three samples are handed over at once
    let fourth = false;
    const waiting = side.push(luma(), frame(), 0).then(() => { fourth = true; });
    await new Promise((r) => setTimeout(r, 5));
    expect(fourth).toBe(false); // the worker has not taken care of any yet
    expect(w.sent.filter((s) => s.msg.type === 'sample')).toHaveLength(3);
    w.say({ type: 'ack' });
    await waiting;
    expect(w.sent.filter((s) => s.msg.type === 'sample')).toHaveLength(4);

    const end = side.end();
    w.say({ type: 'thumbs', start: 0, blobs: [] });
    w.say({ type: 'result', pass1: [], pass1S: 1.5, thumbS: 0.5 });
    expect(await end).toMatchObject({ pass1: [], pass1S: 1.5, thumbS: 0.5 });
    side.close();
    expect(w.terminated).toBe(true);
  });

  it('delivers the thumbnails the worker sends, in the order they come', async () => {
    FakeWorker.onMessage = (w, m) => { if (m.type === 'ping') w.say({ type: 'pong' }); };
    const got: [number, number][] = [];
    const side = await openSide(true, (start, blobs) => got.push([start, blobs.length]));
    const w = FakeWorker.all[0];
    w.say({ type: 'thumbs', start: 0, blobs: new Array(40).fill(null) });
    w.say({ type: 'thumbs', start: 40, blobs: new Array(7).fill(null) });
    await new Promise((r) => setTimeout(r, 5));
    expect(got).toEqual([[0, 40], [40, 7]]);
    side.close();
  });

  it('turns an error of the worker into a failure of the caller, also while it waits and when it asks for the result', async () => {
    FakeWorker.onMessage = (w, m) => { if (m.type === 'ping') w.say({ type: 'pong' }); };
    const side = await openSide(true, () => undefined);
    const w = FakeWorker.all[0];
    for (let i = 0; i < 3; i++) await side.push(luma(), frame(), 0);
    const waiting = side.push(luma(), frame(), 0); // held back
    w.say({ type: 'error', message: 'pass 1 broke' });
    await expect(waiting).rejects.toThrow('pass 1 broke');
    await expect(side.push(luma(), frame(), 0)).rejects.toThrow('pass 1 broke');
    await expect(side.end()).rejects.toThrow('pass 1 broke');
    side.close();
  });

  it('works in the calling thread: pass 1 results come out at the end without any frame work', async () => {
    const side = await openSide(false, () => undefined);
    side.start(8, 4, [[0, 0], [1, 0], [1, 1], [0, 1]]);
    const r = await side.end();
    expect(r.pass1).toEqual([]);
    expect(r.pass1S).toBeGreaterThanOrEqual(0);
  });
});

describe('createPipelinedDetector', () => {
  const reply = (w: FakeWorker, m: Msg, extra: Msg = {}) => w.say({ id: m.id, ...extra });
  const state = { backend: 'wasm', fallbackReason: null, gpuCheck: null, threads: 2, isolated: true };

  it('runs the detector in its own worker, hands the pixels over without copying, and reports what the worker reports', async () => {
    FakeWorker.onMessage = (w, m) => {
      if (m.type === 'init') reply(w, m, { state });
      else if (m.type === 'detect') reply(w, m, { boxes: [[0, 0, 1, 1, 0.5]], ms: 12, state: { ...state, backend: 'wasm (webgpu failed)' } });
      else reply(w, m);
    };
    const d = await createPipelinedDetector('model.onnx', 'auto', 2, {}, true);
    expect(d.remote).toBe(true);
    expect([d.backend, d.threads, d.isolated]).toEqual(['wasm', 2, true]);
    const px = new Uint8ClampedArray(16);
    const r = await d.submit(px);
    expect(r).toEqual({ boxes: [[0, 0, 1, 1, 0.5]], ms: 12 });
    const detect = FakeWorker.all[0].sent.find((s) => s.msg.type === 'detect')!;
    expect(detect.transfer).toHaveLength(1);
    expect(detect.transfer![0]).toBe(px.buffer); // handed over, not copied
    expect(d.backend).toBe('wasm (webgpu failed)'); // the backend can change while running
    await d.release();
    expect(FakeWorker.all[0].terminated).toBe(true);
  });

  it('falls back to the calling thread when the worker script does not load, and tells apart a failing detector', async () => {
    FakeWorker.onCreate = (w) => w.fail('script failed');
    const d = await createPipelinedDetector('model.onnx', 'wasm', undefined, {}, true);
    expect([d.remote, d.backend]).toEqual([false, 'in-thread']);
    expect(await d.submit(new Uint8ClampedArray(4))).toMatchObject({ boxes: [[0.1, 0.1, 0.2, 0.2, 0.9]] });

    // the worker runs but the detector cannot be set up there: the calling thread would fail the same way, so no fallback
    const { createDetector } = await import('../../src/worker/detector');
    vi.mocked(createDetector).mockClear();
    FakeWorker.all = [];
    FakeWorker.onCreate = () => undefined;
    FakeWorker.onMessage = (w, m) => reply(w, m, { error: 'model missing' });
    await expect(createPipelinedDetector('model.onnx', 'wasm', undefined, {}, true)).rejects.toThrow('model missing');
    expect(FakeWorker.all[0].terminated).toBe(true);
    expect(createDetector).not.toHaveBeenCalled();
  });

  it('does not start a worker when asked to stay in the calling thread', async () => {
    const d = await createPipelinedDetector('model.onnx', 'wasm', undefined, {}, false);
    expect(d.remote).toBe(false);
    expect(FakeWorker.all).toHaveLength(0);
  });

  it('fails the waiting calls and every later one when the worker dies, and does not wait forever to release it', async () => {
    FakeWorker.onMessage = (w, m) => { if (m.type === 'init') reply(w, m, { state }); };
    const d = await createPipelinedDetector('model.onnx', 'wasm', undefined, {}, true);
    const w = FakeWorker.all[0];
    const waiting = d.submit(new Uint8ClampedArray(4));
    w.fail('out of memory');
    await expect(waiting).rejects.toThrow('out of memory');
    await expect(d.submit(new Uint8ClampedArray(4))).rejects.toThrow('out of memory');
    await d.release();
    expect(w.terminated).toBe(true);
  });

  it('gives up waiting for a worker that does not answer the release', async () => {
    FakeWorker.onMessage = (w, m) => { if (m.type === 'init') reply(w, m, { state }); };
    const d = await createPipelinedDetector('model.onnx', 'wasm', undefined, {}, true);
    vi.useFakeTimers();
    const released = d.release();
    await vi.advanceTimersByTimeAsync(5000);
    await released;
    expect(FakeWorker.all[0].terminated).toBe(true);
  });
});
