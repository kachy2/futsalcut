"""Minimal MP4/MOV index reader.

Reads only the `moov` sample tables (no decoding, no full-file read), which is enough to get
codec/size/rotation, per-sample timing and the keyframe (sync sample) table of every track.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


def _boxes(f: BinaryIO, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield (type, payload_start, box_end) for the boxes in [start, end)."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, typ = struct.unpack(">I4s", f.read(8))
        hlen = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen or pos + size > end:
            return
        yield typ, pos + hlen, pos + size
        pos += size


@dataclass
class Track:
    track_id: int = 0
    handler: str = ""
    codec: str = ""
    timescale: int = 1
    duration: int = 0
    width: int = 0
    height: int = 0
    rotation: int = 0  # degrees, from the tkhd display matrix
    sample_rate: int = 0
    channels: int = 0
    sample_entry: bytes = b""  # raw stsd entry (includes avcC/hvcC) for compatibility checks
    stts: list[tuple[int, int]] = field(default_factory=list)  # (count, delta)
    ctts: list[tuple[int, int]] = field(default_factory=list)  # (count, offset)
    stss: list[int] | None = None  # 1-based sync sample numbers; None means every sample is sync
    sizes: list[int] = field(default_factory=list)
    elst: list[tuple[int, int, float]] = field(default_factory=list)  # (duration, media_time, rate)

    @property
    def sample_count(self) -> int:
        return len(self.sizes)

    @property
    def duration_s(self) -> float:
        return self.duration / self.timescale if self.timescale else 0.0

    def decode_times(self) -> list[int]:
        out, t = [], 0
        for count, delta in self.stts:
            for _ in range(count):
                out.append(t)
                t += delta
        return out

    def presentation_times(self) -> list[int]:
        dts = self.decode_times()
        if not self.ctts:
            return dts
        offsets = [off for count, off in self.ctts for _ in range(count)]
        return [d + o for d, o in zip(dts, offsets)]

    def keyframe_indices(self) -> list[int]:
        """0-based indices of sync samples."""
        if self.stss is None:
            return list(range(self.sample_count))
        return [n - 1 for n in self.stss]


@dataclass
class MediaIndex:
    path: str
    file_size: int
    creation_time: int = 0  # seconds since 1904-01-01 (mvhd)
    tracks: list[Track] = field(default_factory=list)

    @property
    def video(self) -> Track | None:
        return next((t for t in self.tracks if t.handler == "vide"), None)

    @property
    def audio(self) -> Track | None:
        return next((t for t in self.tracks if t.handler == "soun"), None)


def _full_box_version(f: BinaryIO, start: int) -> int:
    f.seek(start)
    return f.read(1)[0]


def _parse_trak(f: BinaryIO, start: int, end: int) -> Track:
    t = Track()

    def walk(s: int, e: int) -> None:
        for typ, bs, be in _boxes(f, s, e):
            if typ in _CONTAINERS:
                walk(bs, be)
            elif typ == b"tkhd":
                f.seek(bs)
                d = f.read(be - bs)
                v = d[0]
                if v == 1:
                    t.track_id = struct.unpack(">I", d[20:24])[0]
                    m = struct.unpack(">9i", d[52:88])
                else:
                    t.track_id = struct.unpack(">I", d[12:16])[0]
                    m = struct.unpack(">9i", d[40:76])
                a, b = m[0] / 65536.0, m[1] / 65536.0
                t.rotation = int(round(math.degrees(math.atan2(b, a)))) % 360
            elif typ == b"mdhd":
                f.seek(bs)
                d = f.read(be - bs)
                if d[0] == 1:
                    t.timescale, t.duration = struct.unpack(">IQ", d[20:32])
                else:
                    t.timescale, t.duration = struct.unpack(">II", d[12:20])
            elif typ == b"hdlr":
                f.seek(bs + 8)
                t.handler = f.read(4).decode("latin1")
            elif typ == b"stsd":
                f.seek(bs)
                d = f.read(be - bs)
                entry_size = struct.unpack(">I", d[8:12])[0]
                t.sample_entry = d[8:8 + entry_size]
                t.codec = d[12:16].decode("latin1")
                body = d[16:]  # sample entry after size+type
                if t.handler == "vide" and len(body) >= 28:
                    t.width, t.height = struct.unpack(">HH", body[24:28])
                elif t.handler == "soun" and len(body) >= 28:
                    t.channels = struct.unpack(">H", body[16:18])[0]
                    t.sample_rate = struct.unpack(">I", body[24:28])[0] >> 16
            elif typ == b"stts":
                f.seek(bs)
                d = f.read(be - bs)
                n = struct.unpack(">I", d[4:8])[0]
                t.stts = [struct.unpack(">II", d[8 + 8 * i:16 + 8 * i]) for i in range(n)]
            elif typ == b"ctts":
                f.seek(bs)
                d = f.read(be - bs)
                v = d[0]
                n = struct.unpack(">I", d[4:8])[0]
                fmt = ">Ii" if v == 1 else ">II"
                t.ctts = [struct.unpack(fmt, d[8 + 8 * i:16 + 8 * i]) for i in range(n)]
            elif typ == b"stss":
                f.seek(bs)
                d = f.read(be - bs)
                n = struct.unpack(">I", d[4:8])[0]
                t.stss = list(struct.unpack(">%dI" % n, d[8:8 + 4 * n]))
            elif typ == b"stsz":
                f.seek(bs)
                d = f.read(be - bs)
                ssz, n = struct.unpack(">II", d[4:12])
                t.sizes = [ssz] * n if ssz else list(struct.unpack(">%dI" % n, d[12:12 + 4 * n]))
            elif typ == b"elst":
                f.seek(bs)
                d = f.read(be - bs)
                v = d[0]
                n = struct.unpack(">I", d[4:8])[0]
                rec = 20 if v == 1 else 12
                for i in range(n):
                    r = d[8 + rec * i:8 + rec * (i + 1)]
                    if v == 1:
                        dur, mt = struct.unpack(">Qq", r[:16])
                        rate = struct.unpack(">i", r[16:20])[0] / 65536.0
                    else:
                        dur, mt = struct.unpack(">Ii", r[:8])
                        rate = struct.unpack(">i", r[8:12])[0] / 65536.0
                    t.elst.append((dur, mt, rate))

    walk(start, end)
    return t


def read_index(path: str) -> MediaIndex:
    size = os.path.getsize(path)
    idx = MediaIndex(path=path, file_size=size)
    with open(path, "rb") as f:
        for typ, bs, be in _boxes(f, 0, size):
            if typ != b"moov":
                continue
            for t2, s2, e2 in _boxes(f, bs, be):
                if t2 == b"mvhd":
                    v = _full_box_version(f, s2)
                    f.seek(s2 + 4)
                    idx.creation_time = struct.unpack(">Q" if v == 1 else ">I", f.read(8 if v == 1 else 4))[0]
                elif t2 == b"trak":
                    idx.tracks.append(_parse_trak(f, s2, e2))
    if not idx.tracks:
        raise ValueError(f"no moov/trak found in {path} (not an MP4/MOV file?)")
    return idx
