/// <reference lib="webworker" />
/** Export worker: lossless cut-and-join straight to the file the user picked (File System Access API). */
import { exportLossless, type ExportSegment } from '../core/exporter';

const signal = { cancelled: false };

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  if (msg?.type === 'cancel') signal.cancelled = true;
  if (msg?.type !== 'export') return;
  signal.cancelled = false;
  try {
    const handle = msg.handle as FileSystemFileHandle;
    const writable = await handle.createWritable();
    const report = await exportLossless(msg.files as File[], msg.segments as ExportSegment[], writable,
      (doneS, totalS) => postMessage({ type: 'progress', doneS, totalS }), signal);
    // self-check: reopen the written file and read its duration from the index
    const { ALL_FORMATS, BlobSource, Input } = await import('mediabunny');
    const written = await handle.getFile();
    const check = new Input({ source: new BlobSource(written), formats: ALL_FORMATS });
    const durationS = await check.computeDuration();
    postMessage({ type: 'done', report, check: { bytes: written.size, durationS } });
  } catch (e) {
    postMessage({ type: 'error', message: String((e as Error)?.message ?? e) });
  }
};
