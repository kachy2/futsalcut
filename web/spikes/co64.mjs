// P0.5(d) web side: does Mediabunny write a valid MP4 index (co64 + 64-bit mdat size) past 4 GiB?
// Streams ~5.3 GiB of fake packets into a discarding target, keeps only the file head and tail,
// then inspects the boxes.   node spikes/co64.mjs [/data/synthetic/syn_h264_gop1.mp4]
import {
  ALL_FORMATS, EncodedPacket, EncodedVideoPacketSource, FilePathSource, Input, Mp4OutputFormat, Output, StreamTarget,
} from 'mediabunny';

const REF = process.argv[2] ?? '/data/synthetic/syn_h264_gop1.mp4';
const PACKET = 2 * 1024 * 1024;
const COUNT = 2700; // ~5.27 GiB

const input = new Input({ source: new FilePathSource(REF), formats: ALL_FORMATS });
const track = await input.getPrimaryVideoTrack();
const decoderConfig = await track.getDecoderConfig();

let size = 0;
const head = new Uint8Array(64);
let tail = [];
const writable = new WritableStream({
  write(chunk) {
    const end = chunk.position + chunk.data.length;
    size = Math.max(size, end);
    if (chunk.position < 64) head.set(chunk.data.subarray(0, 64 - chunk.position), chunk.position);
    tail.push({ position: chunk.position, data: chunk.data.slice(0, Math.min(chunk.data.length, 64 * 1024 * 1024)) });
    if (tail.length > 3) tail.shift();
  },
});
const output = new Output({ format: new Mp4OutputFormat({ fastStart: false }), target: new StreamTarget(writable, { chunked: true }) });
const src = new EncodedVideoPacketSource(track.codec);
output.addVideoTrack(src);
await output.start();
const payload = new Uint8Array(PACKET).fill(0x55);
const t0 = performance.now();
for (let i = 0; i < COUNT; i++) {
  const p = new EncodedPacket(payload, i % 30 === 0 ? 'key' : 'delta', i / 30, 1 / 30);
  await src.add(p, i === 0 ? { decoderConfig } : undefined);
}
await output.finalize();
const wall = (performance.now() - t0) / 1000;

// locate moov in the retained tail chunks
const last = tail[tail.length - 1];
const buf = Buffer.concat(tail.map((t) => Buffer.from(t.data)));
const base = tail[0].position;
const moovAt = buf.indexOf('moov');
const res = {
  bytesWritten: size,
  GiB: +(size / 2 ** 30).toFixed(3),
  wallS: +wall.toFixed(1),
  headBoxes: Buffer.from(head).toString('latin1').replace(/[^\x20-\x7e]/g, '.'),
  moovFound: moovAt >= 0,
  moovOffset: moovAt >= 0 ? base + moovAt - 4 : null,
  hasCo64: buf.indexOf('co64', moovAt) >= 0,
  hasStco: buf.indexOf('stco', moovAt) >= 0,
  lastChunkEnd: last.position + last.data.length,
};
// mdat header: size field 1 + 64-bit largesize means a large mdat
const hb = Buffer.from(head);
const mdatAt = hb.indexOf('mdat');
if (mdatAt >= 4) {
  const small = hb.readUInt32BE(mdatAt - 4);
  res.mdatSizeField = small;
  if (small === 1) res.mdatLargeSize = Number(hb.readBigUInt64BE(mdatAt + 4));
}
console.log(JSON.stringify(res, null, 1));
