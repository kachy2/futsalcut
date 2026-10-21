// P0.5(c) web side: the same lossless cut-and-join rules as python/src/futsalcut/remux.py,
// implemented with Mediabunny in Node (packet copy needs no WebCodecs).
//   node spikes/remux.mjs [/data/synthetic] [/data/outputs/p05]
import {
  ALL_FORMATS, EncodedAudioPacketSource, EncodedPacket, EncodedPacketSink, EncodedVideoPacketSource,
  FilePathSource, FilePathTarget, Input, Mp4OutputFormat, Output,
} from 'mediabunny';
import fs from 'node:fs';
import path from 'node:path';

const SYN = process.argv[2] ?? '/data/synthetic';
const OUT = process.argv[3] ?? '/data/outputs/p05';

const CASES = {
  h264_gop1: [['syn_h264_gop1.mp4'], [[0, 3, 9], [0, 20, 27], [0, 40, 52], [0, 70, 75], [0, 100, 112]]],
  h264_bframes: [['syn_h264_bframes.mp4'], [[0, 2, 6], [0, 9, 13], [0, 18, 22], [0, 27, 31], [0, 35, 40]]],
  hevc_opengop: [['syn_hevc_opengop.mp4'], [[0, 2, 6], [0, 9, 13], [0, 18, 22], [0, 27, 31], [0, 35, 40]]],
  rot90: [['syn_rot90.mp4'], [[0, 2, 6], [0, 9, 13], [0, 18, 22], [0, 27, 31], [0, 35, 40]]],
  vfr: [['syn_vfr.mp4'], [[0, 2, 6], [0, 9, 13], [0, 18, 22], [0, 27, 31], [0, 35, 40]]],
  split_pair: [['syn_split_a.mp4', 'syn_split_b.mp4'],
    [[0, 10, 20], [0, 50, 60], [1, 0, 8], [1, 30, 36], [1, 50, 60]]],
};

const EPS = 0.05; // seconds: a boundary within 50 ms of a keyframe means that keyframe

/** End-of-sequence NAL (type 36) with the stream's NAL length size, read from hvcC. */
function hevcEos(description) {
  const d = !description ? null
    : ArrayBuffer.isView(description)
      ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
      : new Uint8Array(description);
  const lenSize = d && d.length > 21 ? (d[21] & 3) + 1 : 4;
  const eos = new Uint8Array(lenSize + 2);
  eos[lenSize - 1] = 2;
  eos[lenSize] = 0x48;
  eos[lenSize + 1] = 0x01;
  return eos;
}

export async function remux(sources, segments, outPath) {
  const t0 = performance.now();
  const inputs = sources.map((p) => new Input({ source: new FilePathSource(p), formats: ALL_FORMATS }));
  const vt = await Promise.all(inputs.map((i) => i.getPrimaryVideoTrack()));
  const at = await Promise.all(inputs.map((i) => i.getPrimaryAudioTrack()));
  const vcfg = await vt[0].getDecoderConfig();
  const acfg = at[0] && at[0].codec === 'aac' ? await at[0].getDecoderConfig() : null;
  const isHevc = vt[0].codec === 'hevc';
  const eos = isHevc ? hevcEos(vcfg.description) : null;

  const output = new Output({ format: new Mp4OutputFormat({ fastStart: false }), target: new FilePathTarget(outPath) });
  const vsrc = new EncodedVideoPacketSource(vt[0].codec);
  output.addVideoTrack(vsrc, { rotation: vt[0].rotation });
  const asrc = acfg ? new EncodedAudioPacketSource('aac') : null;
  if (asrc) output.addAudioTrack(asrc);
  await output.start();

  const VT = vt[0].timeResolution; // video ticks per second
  const sr = acfg?.sampleRate ?? 0;
  const flen = 1024;
  const ticks = (s) => Math.round(s * VT);
  let cumV = 0; // integer video ticks written
  let framesA = 0;
  let firstV = true;
  let firstA = true;
  const rep = { segments: [], videoPackets: 0, audioPackets: 0, droppedLeading: 0, eosInserted: 0, rotation: vt[0].rotation };

  // segment times are relative to each file's first video timestamp (same origin as the analysis)
  const origins = await Promise.all(vt.map((t) => t.getFirstTimestamp()));
  for (let si = 0; si < segments.length; si++) {
    const [fi, startRel, endRel] = segments[si];
    const start = origins[fi] + startRel;
    const end = origins[fi] + endRel;
    const vsink = new EncodedPacketSink(vt[fi]);
    const key = await vsink.getKeyPacket(start + EPS);
    if (!key) { rep.segments.push({ fi, start: startRel, end: endRel, skipped: true }); continue; }
    // actual end: first keyframe at/after the requested end, else the stream end
    let stopKey = await vsink.getKeyPacket(end + EPS);
    if (stopKey && stopKey.timestamp < end - EPS) stopKey = await vsink.getNextKeyPacket(stopKey);
    // (a key packet within EPS of the end counts as the end itself)
    if (stopKey && stopKey.timestamp <= key.timestamp + EPS) stopKey = await vsink.getNextKeyPacket(key);
    const keyT = ticks(key.timestamp);
    let segEndT = stopKey ? ticks(stopKey.timestamp) : null;

    let aIter = null;
    let aNeed = 0;
    let aFrom = 0;
    let aTaken = 0;
    const vIter = vsink.packets(key, stopKey ?? undefined);
    let lastEndT = keyT;

    // video packets are produced one ahead so the segment's last packet can carry an EOS NAL
    let ahead = null;
    const pullVideo = async () => {
      for (;;) {
        const { value: p, done } = await vIter.next();
        if (done) return null;
        const pt = ticks(p.timestamp);
        if (pt < keyT) { rep.droppedLeading++; continue; }
        lastEndT = Math.max(lastEndT, pt + ticks(p.duration));
        return p;
      }
    };
    ahead = await pullVideo();
    let buffered = null; // only used when the segment runs to the end of the stream
    if (segEndT === null) {
      // stream end: drain to learn the true end before sizing the audio
      buffered = [];
      for (let p = ahead; p; p = await pullVideo()) buffered.push(p);
      segEndT = lastEndT;
      ahead = buffered.shift() ?? null;
    }
    const nextRaw = async () => (buffered ? buffered.shift() ?? null : pullVideo());

    if (asrc) {
      const asink = new EncodedPacketSink(at[fi]);
      const lead = (framesA * flen) / sr - cumV / VT;
      aFrom = key.timestamp + lead - (0.5 * flen) / sr;
      aNeed = Math.round((((cumV + segEndT - keyT) / VT) * sr) / flen) - framesA;
      const first = (await asink.getPacket(Math.max(aFrom, 0))) ?? (await asink.getFirstPacket());
      aIter = first ? asink.packets(first) : null;
    }

    const nextVideo = async () => {
      if (!ahead) return null;
      const cur = ahead;
      ahead = await nextRaw();
      let data = null;
      if (!ahead && eos && si < segments.length - 1) {
        data = new Uint8Array(cur.data.length + eos.length);
        data.set(cur.data, 0);
        data.set(eos, cur.data.length);
        rep.eosInserted++;
      }
      const ts = (cumV + ticks(cur.timestamp) - keyT) / VT;
      return data
        ? new EncodedPacket(data, cur.type, ts, cur.duration, cur.sequenceNumber)
        : cur.clone({ timestamp: ts });
    };
    const nextAudio = async () => {
      if (!aIter || aTaken >= aNeed) return null;
      for (;;) {
        const { value: p, done } = await aIter.next();
        if (done) return null;
        if (p.timestamp < aFrom) continue;
        const ts = ((framesA + aTaken) * flen) / sr;
        aTaken++;
        if (aTaken >= aNeed) await aIter.return();
        return p.clone({ timestamp: ts });
      }
    };

    // k-way merge by output timestamp so neither track has to be buffered by the muxer
    let v = await nextVideo();
    let a = await nextAudio();
    while (v || a) {
      if (v && (!a || v.timestamp <= a.timestamp)) {
        await vsrc.add(v, firstV ? { decoderConfig: vcfg } : undefined);
        firstV = false;
        rep.videoPackets++;
        v = await nextVideo();
      } else {
        await asrc.add(a, firstA ? { decoderConfig: acfg } : undefined);
        firstA = false;
        rep.audioPackets++;
        a = await nextAudio();
      }
    }
    const segT = segEndT - keyT;
    cumV += segT;
    framesA += aTaken;
    rep.segments.push({ fi, start: startRel, end: endRel, keyStart: key.timestamp - origins[fi],
      keyEnd: segEndT / VT - origins[fi], dur: segT / VT, audioFrames: aTaken });
  }
  await output.finalize();
  rep.videoDurationS = cumV / VT;
  rep.audioDurationS = sr ? (framesA * flen) / sr : 0;
  rep.maxAvErrorS = Math.abs(rep.audioDurationS - rep.videoDurationS);
  rep.wallS = (performance.now() - t0) / 1000;
  return rep;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  fs.mkdirSync(OUT, { recursive: true });
  const results = {};
  for (const [name, [srcs, segs]] of Object.entries(CASES)) {
    const out = path.join(OUT, `mb_remux_${name}.mp4`);
    try {
      results[name] = await remux(srcs.map((s) => path.join(SYN, s)), segs, out);
      const r = results[name];
      console.log(`== ${name}: video ${r.videoDurationS.toFixed(3)}s audio ${r.audioDurationS.toFixed(3)}s`
        + ` dropped ${r.droppedLeading} eos ${r.eosInserted} rot ${r.rotation} wall ${r.wallS.toFixed(2)}s`);
    } catch (e) {
      results[name] = { error: String(e?.stack ?? e) };
      console.log(`== ${name}: ERROR ${e?.message ?? e}`);
    }
  }
  fs.writeFileSync(path.join(OUT, 'mb_remux_results.json'), JSON.stringify(results, null, 1));
}
