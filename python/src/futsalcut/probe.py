"""Describe a recording: codec, size, rotation, timing and GOP structure (index only, no decoding)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import statistics
from dataclasses import asdict, dataclass

from .mp4index import MediaIndex, read_index

_MP4_EPOCH = _dt.datetime(1904, 1, 1, tzinfo=_dt.timezone.utc)


@dataclass
class ProbeResult:
    path: str
    file_size: int
    creation_time_utc: str
    video_codec: str
    width: int
    height: int
    rotation: int
    fps: float
    duration_s: float
    frames: int
    has_bframes: bool
    keyframes: int
    gop_median_s: float
    gop_min_s: float
    gop_max_s: float
    keyframe_bytes_ratio: float
    video_bitrate_mbps: float
    video_entry_sha1: str  # sha1 of avcC/hvcC; identical across split files => lossless concat is safe
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int

    def to_dict(self) -> dict:
        return asdict(self)


def decoder_config_box(sample_entry: bytes) -> bytes:
    """The codec configuration child box (avcC/hvcC/av1C/vpcC) of a visual sample entry.

    Split recordings must have identical configuration to be joined losslessly; other child boxes
    (e.g. btrt bitrate stats) legitimately differ per file and are ignored.
    """
    pos = 86  # 8-byte box header + 78-byte VisualSampleEntry fields
    while pos + 8 <= len(sample_entry):
        size = int.from_bytes(sample_entry[pos:pos + 4], "big")
        typ = sample_entry[pos + 4:pos + 8]
        if size < 8:
            break
        if typ in (b"avcC", b"hvcC", b"av1C", b"vpcC"):
            return sample_entry[pos:pos + size]
        pos += size
    return sample_entry


def probe(path: str) -> ProbeResult:
    idx: MediaIndex = read_index(path)
    v = idx.video
    if v is None:
        raise ValueError(f"no video track in {path}")
    a = idx.audio
    n = v.sample_count
    dur = v.duration_s
    fps = n / dur if dur else 0.0
    keys = v.keyframe_indices()
    dts = v.decode_times()
    gaps = [(dts[b] - dts[a_]) / v.timescale for a_, b in zip(keys, keys[1:])] or [dur]
    total = sum(v.sizes)
    kbytes = sum(v.sizes[k] for k in keys)
    created = _MP4_EPOCH + _dt.timedelta(seconds=idx.creation_time) if idx.creation_time else None
    return ProbeResult(
        path=path,
        file_size=idx.file_size,
        creation_time_utc=created.isoformat() if created else "",
        video_codec=v.codec,
        width=v.width,
        height=v.height,
        rotation=v.rotation,
        fps=round(fps, 3),
        duration_s=round(dur, 3),
        frames=n,
        has_bframes=bool(v.ctts) and any(off != v.ctts[0][1] for _, off in v.ctts),
        keyframes=len(keys),
        gop_median_s=round(statistics.median(gaps), 3),
        gop_min_s=round(min(gaps), 3),
        gop_max_s=round(max(gaps), 3),
        keyframe_bytes_ratio=round(kbytes / total, 4) if total else 0.0,
        video_bitrate_mbps=round(total * 8 / dur / 1e6, 2) if dur else 0.0,
        video_entry_sha1=hashlib.sha1(decoder_config_box(v.sample_entry)).hexdigest(),
        audio_codec=a.codec if a else "",
        audio_sample_rate=a.sample_rate if a else 0,
        audio_channels=a.channels if a else 0,
    )
