"""Output checks for remuxed files: stream structure, decodability and A/V sync (flash/beep)."""
from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field

import av
import numpy as np


@dataclass
class StreamCheck:
    kind: str
    codec: str
    packets: int = 0
    keyframes: int = 0
    first_pts_s: float | None = None
    last_end_s: float | None = None
    dts_monotonic: bool = True
    rotation: float | None = None


@dataclass
class FileCheck:
    path: str
    streams: list[StreamCheck] = field(default_factory=list)
    decode_errors: list[str] = field(default_factory=list)
    # "non monotonically increasing dts ... N >= N" from ffmpeg's null muxer: two frames rounding to the
    # same output tick because the *source* timestamps jitter (also seen on untouched phone recordings)
    timestamp_jitter_warnings: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _rotation(path: str) -> float | None:
    """Clockwise display rotation from the MP4/MOV tkhd matrix."""
    try:
        from .mp4index import read_index
        v = read_index(path).video
        return float(v.rotation) if v else None
    except (ValueError, OSError):
        return None


def check_file(path: str, decode: bool = True) -> FileCheck:
    fc = FileCheck(path=path)
    rot = _rotation(path)
    with av.open(path) as c:
        checks = {}
        for s in c.streams:
            if s.type in ("video", "audio"):
                checks[s.index] = StreamCheck(kind=s.type, codec=s.codec_context.name,
                                              rotation=rot if s.type == "video" else None)
        last_dts: dict[int, int] = {}
        for pkt in c.demux():
            if pkt.dts is None or pkt.stream.index not in checks:
                continue
            sc = checks[pkt.stream.index]
            tb = pkt.time_base
            sc.packets += 1
            sc.keyframes += int(pkt.is_keyframe)
            if pkt.pts is not None:
                t0 = float(pkt.pts * tb)
                sc.first_pts_s = t0 if sc.first_pts_s is None else min(sc.first_pts_s, t0)
                t1 = float((pkt.pts + (pkt.duration or 0)) * tb)
                sc.last_end_s = t1 if sc.last_end_s is None else max(sc.last_end_s, t1)
            prev = last_dts.get(pkt.stream.index)
            if prev is not None and pkt.dts <= prev:
                sc.dts_monotonic = False
            last_dts[pkt.stream.index] = pkt.dts
        fc.streams = list(checks.values())
    if decode:
        r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-f", "null", "-"],
                           capture_output=True, text=True)
        errs = []
        for ln in r.stderr.splitlines():
            if not ln.strip():
                continue
            m = re.search(r"non monotonically increasing dts to muxer in stream \d+: (\d+) >= (\d+)", ln)
            if m and m.group(1) == m.group(2):
                fc.timestamp_jitter_warnings += 1
                continue
            errs.append(ln)
        fc.decode_errors = errs[:50]
    return fc


def flash_times(path: str, threshold: float = 200.0) -> list[float]:
    """Presentation times (s) of frames whose mean luma exceeds `threshold` (rising edges only)."""
    times, prev = [], False
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.codec_context.thread_type = "AUTO"
        for fr in c.decode(vs):
            y = fr.to_ndarray(format="gray")[::8, ::8]
            on = float(y.mean()) > threshold
            if on and not prev:
                times.append(float(fr.pts * fr.time_base))
            prev = on
    return times


def beep_times(path: str, min_gap_s: float = 0.5) -> list[float]:
    """Onset times (s) of 1 kHz beeps in the first audio track (energy rising edges)."""
    with av.open(path) as c:
        if not c.streams.audio:
            return []
        a = c.streams.audio[0]
        chunks, t0 = [], None
        for fr in c.decode(a):
            if t0 is None:
                t0 = float(fr.pts * fr.time_base) if fr.pts is not None else 0.0
            x = fr.to_ndarray()
            chunks.append(x.mean(axis=0) if x.ndim == 2 and x.shape[0] < x.shape[1] else x.reshape(-1))
            sr = fr.sample_rate
    sig = np.concatenate(chunks).astype(np.float64)
    win = max(1, sr // 1000)  # 1 ms
    n = len(sig) // win
    energy = (sig[:n * win].reshape(n, win) ** 2).mean(axis=1)
    thr = energy.max() * 0.05
    onsets, last = [], -1e9
    above = energy > thr
    for i in range(1, n):
        if above[i] and not above[i - 1]:
            t = (t0 or 0.0) + i * win / sr
            if t - last >= min_gap_s:
                onsets.append(t)
                last = t
    return onsets


def av_offsets(path: str) -> list[float]:
    """Audio-minus-video offsets (s) for each flash paired with its nearest beep."""
    fl, bp = flash_times(path), beep_times(path)
    if not fl or not bp:
        return []
    bp_arr = np.array(bp)
    return [float(bp_arr[np.argmin(np.abs(bp_arr - f))] - f) for f in fl]
