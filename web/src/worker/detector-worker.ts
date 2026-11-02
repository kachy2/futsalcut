/// <reference lib="webworker" />
/**
 * The person detector in a worker of its own: the analysis worker hands frames over and goes on decoding while this one infers
 * (see detector-client.ts). Requests are answered one at a time, in the order they arrive.
 */
import { createDetector, type Detector } from './detector';
import type { DetectorReply, DetectorRequest, DetectorState } from './detector-client';

let detector: Detector | null = null;
let queue: Promise<void> = Promise.resolve();

const state = (): DetectorState | undefined => detector
  ? { backend: detector.backend, fallbackReason: detector.fallbackReason, gpuCheck: detector.gpuCheck, threads: detector.threads, isolated: self.crossOriginIsolated }
  : undefined;
const reply = (m: DetectorReply) => (self as unknown as Worker).postMessage(m);

async function handle(m: DetectorRequest): Promise<void> {
  try {
    if (m.type === 'init') {
      detector = await createDetector(m.modelUrl, m.prefer, m.threads, m.opts);
      reply({ id: m.id, state: state() });
    } else if (m.type === 'detect') {
      if (!detector) throw new Error('the detector is not running');
      const t0 = performance.now();
      const boxes = await detector.detect(m.rgba);
      reply({ id: m.id, boxes, ms: performance.now() - t0, state: state() }); // the backend may have changed while running
    } else {
      await detector?.release();
      detector = null;
      reply({ id: m.id });
    }
  } catch (e) {
    reply({ id: m.id, error: String((e as Error)?.message ?? e) });
  }
}

self.onmessage = (ev: MessageEvent<DetectorRequest>) => {
  queue = queue.then(() => handle(ev.data)); // handle never rejects
};
