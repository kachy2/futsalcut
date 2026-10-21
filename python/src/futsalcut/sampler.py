"""Keyframe sampler: one downscaled luma image per GOP (1-second grid).

Parity contract with the web port (see plan §3.1): take the decoder's raw Y plane (no colour
conversion, no range expansion) and shrink it with `box_downscale`, which uses integer box
boundaries and round-half-up division. Both are bit-exact across conformant decoders.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import av
import numpy as np

SAMPLER_VERSION = 2
MIN_SPACING = 0.9  # seconds; take a keyframe when it is at least this far after the last taken one
GRID_EPS = 1e-3  # seconds; float tolerance for the spacing test


def target_size(width: int, height: int, long_side: int = 640) -> tuple[int, int]:
    """Output size keeping aspect ratio, long side = `long_side`, both sides even."""
    # floor(x + 0.5) instead of round(): matches JS Math.round (Python rounds halves to even)
    if width >= height:
        w = long_side
        h = math.floor(height * long_side / width / 2 + 0.5) * 2
    else:
        h = long_side
        w = math.floor(width * long_side / height / 2 + 0.5) * 2
    return w, h


def box_downscale(y: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Average over integer boxes: output (r, c) covers rows [r*H//h, (r+1)*H//h) and
    cols [c*W//w, (c+1)*W//w); result = (sum + n//2) // n."""
    H, W = y.shape
    if out_w > W or out_h > H:
        raise ValueError("box_downscale only shrinks")
    # exact box sums from an integral image (2x faster than strided numpy sums on 4K)
    integ = _integral(np.ascontiguousarray(y))
    rb = (np.arange(out_h + 1) * H) // out_h
    cb = (np.arange(out_w + 1) * W) // out_w
    r0, r1 = rb[:-1, None], rb[1:, None]
    c0, c1 = cb[None, :-1], cb[None, 1:]
    sums = integ[r1, c1] - integ[r0, c1] - integ[r1, c0] + integ[r0, c0]
    n = np.outer(np.diff(rb), np.diff(cb))
    return ((sums + n // 2) // n).astype(np.uint8)


def _integral(y: np.ndarray) -> np.ndarray:
    """(H+1, W+1) integral image with exact integer values."""
    try:
        import cv2
        if y.size * 255 < 2**31:
            return cv2.integral(y)  # int32
        return cv2.integral(y, sdepth=cv2.CV_64F).astype(np.int64)  # exact below 2**53
    except ImportError:
        out = np.zeros((y.shape[0] + 1, y.shape[1] + 1), np.int64)
        np.cumsum(np.cumsum(y, axis=0, dtype=np.int64), axis=1, out=out[1:, 1:])
        return out


def rotate_cw(img: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by 0/90/180/270 degrees (display rotation from the container)."""
    k = (degrees // 90) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=-k))


def display_rotation(path: str) -> int:
    """Clockwise display rotation in degrees (MP4/MOV tkhd matrix; 0 if unknown)."""
    try:
        from .mp4index import read_index
        v = read_index(path).video
        return int(v.rotation) if v else 0
    except (ValueError, OSError):
        return 0


_PLANAR_8BIT = {"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12", "nv21", "gray"}
_PLANAR_10BIT_LOW = {"yuv420p10le", "yuv422p10le", "yuv444p10le", "gray10le"}  # value in low 10 bits
_PLANAR_10BIT_HIGH = {"p010le"}  # value in high 10 bits


def luma_plane(frame: av.VideoFrame) -> np.ndarray:
    """Raw 8-bit Y plane (10-bit sources are truncated with >> 2)."""
    p = frame.planes[0]
    fmt = frame.format.name
    h, w = frame.height, frame.width
    if fmt in _PLANAR_8BIT:
        return np.frombuffer(p, np.uint8, count=p.line_size * h).reshape(h, p.line_size)[:, :w]
    if fmt in _PLANAR_10BIT_LOW or fmt in _PLANAR_10BIT_HIGH:
        a = np.frombuffer(p, np.uint16, count=(p.line_size // 2) * h).reshape(h, p.line_size // 2)[:, :w]
        if fmt in _PLANAR_10BIT_HIGH:
            a = a >> 6
        return (a >> 2).astype(np.uint8)
    raise ValueError(f"unsupported pixel format: {fmt}")


@dataclass
class SampleTiming:
    decode_s: float = 0.0
    luma_s: float = 0.0
    downscale_s: float = 0.0
    write_s: float = 0.0
    wall_s: float = 0.0
    packets_total: int = 0
    packets_decoded: int = 0


@dataclass
class SampleSet:
    """Metadata of a luma cache file (`<stem>.lumaWxH.u8`, frames stored back to back)."""
    source: str
    width: int
    height: int
    src_width: int
    src_height: int
    time_base: str
    strategy: str
    pts: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.pts)


def iter_grid_frames(path: str, strategy: str = "packets", threads: int = 2,
                     limit_s: float | None = None, timing: SampleTiming | None = None
                     ) -> Iterator[tuple[av.VideoFrame, float]]:
    """Yield (frame, seconds) at ~1 Hz: a frame is taken when it is >= MIN_SPACING after the last
    taken one (GOPs of ~1 s give every keyframe even if they drift slightly below 1.000 s).

    strategy:
      packets   - feed only keyframe packets to the decoder (fast path for GOP <= 4 s)
      skipframe - feed all packets, decoder skips non-keyframes (comparison/fallback)
      all       - decode every frame, pick one per second (long-GOP fallback)
    """
    timing = timing or SampleTiming()
    with av.open(path) as container:
        vs = container.streams.video[0]
        cc = vs.codec_context
        cc.thread_type = "FRAME" if strategy != "all" else "AUTO"
        cc.thread_count = threads
        if strategy == "skipframe":
            cc.skip_frame = "NONKEY"
        tb = vs.time_base
        start = vs.start_time or 0
        last_t = -math.inf

        def take(frames: list[av.VideoFrame]) -> Iterator[tuple[av.VideoFrame, float]]:
            nonlocal last_t
            for fr in frames:
                ts = float((fr.pts - start) * tb) if fr.pts is not None else last_t + 1.0
                if ts - last_t + GRID_EPS < MIN_SPACING:
                    continue
                last_t = ts
                yield fr, ts

        for pkt in container.demux(vs):
            if pkt.dts is None:
                continue
            timing.packets_total += 1
            if limit_s is not None and pkt.pts is not None and float((pkt.pts - start) * tb) > limit_s:
                break
            if strategy == "packets" and not pkt.is_keyframe:
                continue
            timing.packets_decoded += 1
            t0 = time.perf_counter()
            frames = cc.decode(pkt)
            timing.decode_s += time.perf_counter() - t0
            yield from take(frames)
        t0 = time.perf_counter()
        frames = cc.decode(None)
        timing.decode_s += time.perf_counter() - t0
        yield from take(frames)


def sample_file(path: str, out_dir: str, long_side: int = 640, strategy: str = "packets",
                threads: int = 2, limit_s: float | None = None,
                thumbs: bool = False) -> tuple[SampleSet, SampleTiming, str]:
    """Decode the 1-second grid of `path` into a luma cache under `out_dir`.

    With `thumbs`, also writes a colour JPEG per sample to `<stem>.thumbs/NNNNNN.jpg` (for labeling
    sheets and previews; not part of the parity contract).
    """
    timing = SampleTiming()
    wall0 = time.perf_counter()
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    ss: SampleSet | None = None
    data_path = ""
    with av.open(path) as c:
        vs = c.streams.video[0]
        src_w, src_h = vs.codec_context.width, vs.codec_context.height
        tb = str(vs.time_base)
    rot = display_rotation(path)
    # contract: shrink in coded orientation, then rotate the small image to display orientation
    rw, rh = target_size(src_w, src_h, long_side)
    w, h = (rh, rw) if rot in (90, 270) else (rw, rh)
    data_path = os.path.join(out_dir, f"{stem}.luma{w}x{h}.u8")
    ss = SampleSet(source=os.path.abspath(path), width=w, height=h, src_width=src_w, src_height=src_h,
                   time_base=tb, strategy=strategy)
    thumb_dir = os.path.join(out_dir, f"{stem}.thumbs")
    if thumbs:
        import cv2  # only needed for JPEG encoding
        os.makedirs(thumb_dir, exist_ok=True)
    with open(data_path, "wb") as out:
        for frame, ts in iter_grid_frames(path, strategy, threads, limit_s, timing):
            t0 = time.perf_counter()
            y = luma_plane(frame)
            t1 = time.perf_counter()
            small = rotate_cw(box_downscale(y, rw, rh), rot)
            t2 = time.perf_counter()
            out.write(small.tobytes())
            if thumbs:
                bgr = rotate_cw(frame.to_ndarray(format="bgr24", width=rw, height=rh), rot)
                ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with open(os.path.join(thumb_dir, f"{ss.count:06d}.jpg"), "wb") as tf:
                    tf.write(jpg.tobytes())
            t3 = time.perf_counter()
            timing.luma_s += t1 - t0
            timing.downscale_s += t2 - t1
            timing.write_s += t3 - t2
            ss.pts.append(int(frame.pts) if frame.pts is not None else -1)
            ss.t.append(round(ts, 6))
    timing.wall_s = time.perf_counter() - wall0
    meta = {
        "version": SAMPLER_VERSION,
        "source": ss.source,
        "source_size": os.path.getsize(path),
        "source_mtime": int(os.path.getmtime(path)),
        "src_width": src_w,
        "src_height": src_h,
        "rotation": rot,
        "width": w,
        "height": h,
        "time_base": tb,
        "strategy": strategy,
        "downscale": "box-int-round-half-up",
        "pts": ss.pts,
        "t": ss.t,
        "timing": timing.__dict__,
    }
    with open(data_path[:-3] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return ss, timing, data_path


def load_cache(data_path: str) -> tuple[np.ndarray, dict]:
    """Memory-map a luma cache written by `sample_file`."""
    with open(data_path[:-3] + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    n = len(meta["pts"])
    arr = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(n, meta["height"], meta["width"]))
    return arr, meta
