/**
 * Lossless cut-and-join with Mediabunny — twin of python/src/futsalcut/remux.py (plan §3.5).
 * DOM-free: sources are Blobs/Files, the output goes to a WritableStream of Mediabunny StreamTarget chunks
 * (a FileSystemWritableFileStream in the browser, a file in Node tests).
 */
import {
  ALL_FORMATS, BlobSource, EncodedAudioPacketSource, EncodedPacket, EncodedPacketSink, EncodedVideoPacketSource,
  Input, Mp4OutputFormat, Output, StreamTarget, type StreamTargetChunk, type EncodedPacket as Pkt,
} from 'mediabunny';

export const SNAP_TOL = 0.05; // a boundary within 50 ms of a keyframe means that keyframe

export interface ExportSegment { file: number; start: number; end: number } // local seconds (from first video timestamp)

export interface ExportReport {
  segments: number;
  videoPackets: number;
  audioPackets: number;
  droppedLeading: number;
  eosInserted: number;
  videoDurationS: number;
  audioDurationS: number;
  maxCutAvErrorS: number;
  bytes: number;
  wallS: number;
}

function hevcEos(description: AllowSharedBufferSource | undefined): Uint8Array {
  const d = !description ? null
    : ArrayBuffer.isView(description)
      ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
      : new Uint8Array(description as ArrayBuffer);
  const lenSize = d && d.length > 21 ? (d[21] & 3) + 1 : 4;
  const eos = new Uint8Array(lenSize + 2);
  eos[lenSize - 1] = 2;
  eos[lenSize] = 0x48;
  eos[lenSize + 1] = 0x01;
  return eos;
}

export async function exportLossless(sources: Blob[], segments: ExportSegment[],
  writable: WritableStream<StreamTargetChunk>, onProgress?: (doneS: number, totalS: number) => void,
  signal?: { cancelled: boolean }): Promise<ExportReport> {
  const t0 = performance.now();
  const inputs = sources.map((b) => new Input({ source: new BlobSource(b), formats: ALL_FORMATS }));
  const vt = await Promise.all(inputs.map((i) => i.getPrimaryVideoTrack()));
  const at = await Promise.all(inputs.map((i) => i.getPrimaryAudioTrack()));
  if (vt.some((t) => !t)) throw new Error('a source has no video track');
  const vcfg = (await vt[0]!.getDecoderConfig())!;
  const acfg = at[0] && at[0].codec === 'aac' ? await at[0].getDecoderConfig() : null;
  const eos = vt[0]!.codec === 'hevc' ? hevcEos(vcfg.description) : null;
  const origins = await Promise.all(vt.map((t) => t!.getFirstTimestamp()));

  let bytes = 0;
  const counting = new WritableStream<StreamTargetChunk>({
    async write(chunk) {
      bytes = Math.max(bytes, chunk.position + chunk.data.byteLength);
      const w = writable.getWriter();
      try { await w.write(chunk); } finally { w.releaseLock(); }
    },
    async close() { await writable.close(); },
    async abort(r) { await writable.abort(r); },
  });
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: false }), target: new StreamTarget(counting, { chunked: true }) });
  const vsrc = new EncodedVideoPacketSource(vt[0]!.codec!);
  output.addVideoTrack(vsrc, { rotation: vt[0]!.rotation });
  const asrc = acfg ? new EncodedAudioPacketSource('aac') : null;
  if (asrc) output.addAudioTrack(asrc);
  await output.start();

  const VT = vt[0]!.timeResolution;
  const sr = acfg?.sampleRate ?? 0;
  const flen = 1024;
  const ticks = (s: number) => Math.round(s * VT);
  const totalS = segments.reduce((s, g) => s + (g.end - g.start), 0);
  let cumV = 0;
  let framesA = 0;
  let firstV = true;
  let firstA = true;
  const rep: ExportReport = { segments: 0, videoPackets: 0, audioPackets: 0, droppedLeading: 0, eosInserted: 0,
    videoDurationS: 0, audioDurationS: 0, maxCutAvErrorS: 0, bytes: 0, wallS: 0 };

  for (let si = 0; si < segments.length; si++) {
    if (signal?.cancelled) throw new Error('cancelled');
    const { file: fi } = segments[si];
    const start = origins[fi] + segments[si].start;
    const end = origins[fi] + segments[si].end;
    const vsink = new EncodedPacketSink(vt[fi]!);
    const key = await vsink.getKeyPacket(start + SNAP_TOL);
    if (!key) continue;
    let stopKey = await vsink.getKeyPacket(end + SNAP_TOL);
    if (stopKey && stopKey.timestamp < end - SNAP_TOL) stopKey = await vsink.getNextKeyPacket(stopKey);
    if (stopKey && stopKey.timestamp <= key.timestamp + SNAP_TOL) stopKey = await vsink.getNextKeyPacket(key);
    const keyT = ticks(key.timestamp);
    let segEndT = stopKey ? ticks(stopKey.timestamp) : null;
    const vIter = vsink.packets(key, stopKey ?? undefined);
    let lastEndT = keyT;
    const pull = async (): Promise<Pkt | null> => {
      for (;;) {
        const { value: p, done } = await vIter.next();
        if (done) return null;
        const pt = ticks(p.timestamp);
        if (pt < keyT) { rep.droppedLeading++; continue; }
        lastEndT = Math.max(lastEndT, pt + ticks(p.duration));
        return p;
      }
    };
    let ahead = await pull();
    let buffered: Pkt[] | null = null;
    if (segEndT === null) { // runs to the end of the stream: drain first to know the true end
      buffered = [];
      for (let p = ahead; p; p = await pull()) buffered.push(p);
      segEndT = lastEndT;
      ahead = buffered.shift() ?? null;
    }
    const nextRaw = async () => (buffered ? buffered.shift() ?? null : pull());

    let aIter: AsyncGenerator<Pkt> | null = null;
    let aNeed = 0;
    let aFrom = 0;
    let aTaken = 0;
    if (asrc) {
      const asink = new EncodedPacketSink(at[fi]!);
      const lead = (framesA * flen) / sr - cumV / VT;
      aFrom = key.timestamp + lead - (0.5 * flen) / sr;
      aNeed = Math.round((((cumV + segEndT - keyT) / VT) * sr) / flen) - framesA;
      const first = (await asink.getPacket(Math.max(aFrom, 0))) ?? (await asink.getFirstPacket());
      aIter = first ? asink.packets(first) : null;
    }
    const nextVideo = async (): Promise<Pkt | null> => {
      if (!ahead) return null;
      const cur = ahead;
      ahead = await nextRaw();
      const ts = (cumV + ticks(cur.timestamp) - keyT) / VT;
      if (!ahead && eos && si < segments.length - 1) {
        const data = new Uint8Array(cur.data.length + eos.length);
        data.set(cur.data, 0);
        data.set(eos, cur.data.length);
        rep.eosInserted++;
        return new EncodedPacket(data, cur.type, ts, cur.duration, cur.sequenceNumber);
      }
      return cur.clone({ timestamp: ts });
    };
    const nextAudio = async (): Promise<Pkt | null> => {
      if (!aIter || aTaken >= aNeed) return null;
      for (;;) {
        const { value: p, done } = await aIter.next();
        if (done) return null;
        if (p.timestamp < aFrom) continue;
        const ts = ((framesA + aTaken) * flen) / sr;
        aTaken++;
        if (aTaken >= aNeed) await aIter.return(undefined);
        return p.clone({ timestamp: ts });
      }
    };
    let v = await nextVideo();
    let a = await nextAudio();
    while (v || a) {
      if (v && (!a || v.timestamp <= a.timestamp)) {
        await vsrc.add(v, firstV ? { decoderConfig: vcfg } : undefined);
        firstV = false;
        rep.videoPackets++;
        v = await nextVideo();
      } else {
        await asrc!.add(a!, firstA ? { decoderConfig: acfg! } : undefined);
        firstA = false;
        rep.audioPackets++;
        a = await nextAudio();
      }
    }
    cumV += segEndT - keyT;
    framesA += aTaken;
    rep.segments++;
    if (sr) rep.maxCutAvErrorS = Math.max(rep.maxCutAvErrorS, Math.abs((framesA * flen) / sr - cumV / VT));
    onProgress?.(cumV / VT, totalS);
  }
  await output.finalize();
  rep.videoDurationS = cumV / VT;
  rep.audioDurationS = sr ? (framesA * flen) / sr : 0;
  rep.bytes = bytes;
  rep.wallS = (performance.now() - t0) / 1000;
  return rep;
}
