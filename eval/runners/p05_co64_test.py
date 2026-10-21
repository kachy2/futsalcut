"""P0.5(d) Python side: does the FFmpeg MP4 muxer (via PyAV) write co64 / a 64-bit mdat past 4 GiB?

Streams ~5.3 GiB of fake H.264 packets into a file-like object that discards the payload and keeps
only the head and the tail, then inspects the boxes. Uses no disk space.
  python eval/runners/p05_co64_test.py /data/synthetic/syn_h264_gop1.mp4
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from fractions import Fraction

import av

PACKET = 2 * 1024 * 1024
COUNT = 2700


class SparseSink:
    """Seekable write-only sink that keeps the first 64 bytes and the last ~64 MiB of writes."""

    def __init__(self) -> None:
        self.pos = 0
        self.size = 0
        self.head = bytearray(64)
        self.tail: deque[tuple[int, bytes]] = deque()
        self.tail_bytes = 0

    def write(self, b) -> int:
        b = bytes(b)
        if self.pos < 64:
            n = min(64 - self.pos, len(b))
            self.head[self.pos:self.pos + n] = b[:n]
        self.tail.append((self.pos, b))
        self.tail_bytes += len(b)
        while self.tail_bytes > 64 * 1024 * 1024:
            _, old = self.tail.popleft()
            self.tail_bytes -= len(old)
        self.pos += len(b)
        self.size = max(self.size, self.pos)
        return len(b)

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def flush(self) -> None:
        pass


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "/data/synthetic/syn_h264_gop1.mp4"
    sink = SparseSink()
    t0 = time.perf_counter()
    with av.open(ref) as src:
        vs = src.streams.video[0]
        out = av.open(sink, "w", format="mp4")
        ov = out.add_stream_from_template(vs)
        payload = b"\x55" * PACKET
        tb = Fraction(1, 30)
        for i in range(COUNT):
            pkt = av.Packet(payload)
            pkt.pts = pkt.dts = i
            pkt.duration = 1
            pkt.time_base = tb
            pkt.is_keyframe = i % 30 == 0
            pkt.stream = ov
            out.mux(pkt)
        out.close()
    wall = time.perf_counter() - t0
    tail = b"".join(b for _, b in sink.tail)
    moov = tail.find(b"moov")
    head = bytes(sink.head)
    mdat = head.find(b"mdat")
    res = {
        "bytes_written": sink.size,
        "GiB": round(sink.size / 2**30, 3),
        "wall_s": round(wall, 1),
        "moov_found": moov >= 0,
        "has_co64": tail.find(b"co64", max(moov, 0)) >= 0,
        "has_stco": tail.find(b"stco", max(moov, 0)) >= 0,
        "head": head.decode("latin1").encode("ascii", "replace").decode(),
    }
    if mdat >= 4:
        small = int.from_bytes(head[mdat - 4:mdat], "big")
        res["mdat_size_field"] = small
        if small == 1:
            res["mdat_largesize"] = int.from_bytes(head[mdat + 4:mdat + 12], "big")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
