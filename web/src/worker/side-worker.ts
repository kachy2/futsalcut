/// <reference lib="webworker" />
/**
 * Helper of the analysis worker (see side-client.ts): streaming pass 1 over the luma samples and the thumbnails of the frames, so that
 * the analysis worker can go on decoding. Messages are handled one at a time, in the order they were sent.
 */
import { polygonMask } from '../core/roi';
import { StreamingPass1 } from '../core/streaming';
import type { SideReply, SideRequest } from './side-client';
import { Thumbnailer } from './thumbs';

const send = (m: SideReply) => (self as unknown as Worker).postMessage(m);
const lap = (since: number) => (performance.now() - since) / 1000;

let sp: StreamingPass1 | null = null;
const thumbs = new Thumbnailer((start, blobs) => send({ type: 'thumbs', start, blobs }));
let pass1S = 0;
let thumbS = 0;
let queue: Promise<void> = Promise.resolve();

async function handle(m: SideRequest): Promise<void> {
  try {
    if (m.type === 'ping') {
      send({ type: 'pong' });
    } else if (m.type === 'start') {
      sp = new StreamingPass1(m.W, m.H, polygonMask(m.zone, m.W, m.H));
    } else if (m.type === 'sample') {
      try {
        let s = performance.now();
        sp!.push(m.luma);
        pass1S += lap(s);
        if (m.frame) { // without a frame the thumbnail is made by the luma helper
          s = performance.now();
          await thumbs.add(m.frame, m.rotation);
          thumbS += lap(s);
        }
      } finally {
        m.frame?.close();
      }
      send({ type: 'ack' });
    } else {
      const s = performance.now();
      const pass1 = sp!.end();
      pass1S += lap(s);
      thumbs.flush();
      send({ type: 'result', pass1, pass1S, thumbS });
    }
  } catch (e) {
    send({ type: 'error', message: String((e as Error)?.message ?? e) });
  }
}

self.onmessage = (ev: MessageEvent<SideRequest>) => {
  queue = queue.then(() => handle(ev.data)); // handle never rejects
};
