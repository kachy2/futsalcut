"""Generate synthetic test recordings with exactly known ground truth (no real people).

Scene: green pitch with white lines seen from a static camera, dark "players" (tall rectangles)
walking in and out of a near-half polygon on a scripted schedule. Every 2 seconds a one-frame white
flash coincides with a 20 ms 1 kHz beep, so A/V offsets can be measured after cutting/remuxing.

Outputs (in --out):
  syn_h264_gop1.mp4        1280x720, 120 s, H.264 GOP 1 s, no B-frames (Samsung-like) - scenario + A/V flashes
  syn_scene.mp4            the same scenario without flashes (for analysis tests)
  syn_split_a/b.mp4        the same recording split at 60 s on a keyframe (identical codec config)
  syn_h264_bframes.mp4     640x360, 40 s, H.264 with B-frames (ctts + video edit list)
  syn_hevc_opengop.mp4     640x360, 40 s, HEVC open GOP + B-frames (RASL leading pictures)
  syn_rot90.mp4            syn_h264_bframes with a 90-degree display matrix (stream copy)
  syn_vfr.mp4              640x360, 40 s, H.264 with dropped frames and jittered timestamps
  syn_truth.json           per-second ground truth for the scenario (players inside the near polygon)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import wave

import numpy as np

FPS = 30
BEEP_EVERY = 2.0
NEAR_POLY = [(0.05, 0.30), (0.95, 0.30), (1.0, 1.0), (0.0, 1.0)]  # normalized; lower 70% = near half

# (start_s, end_s, players); a player is a straight path of its feet point: ((x0, y0), (x1, y1))
SCHEDULE = [
    (0, 12, []),                                                             # empty pitch
    (12, 24, [((0.20, 0.15), (0.80, 0.18))]),                                # one far player
    (24, 36, [((0.30, 0.55), (0.60, 0.70))]),                                # one near player
    (36, 54, [((0.10, 0.50), (0.70, 0.85)), ((0.85, 0.60), (0.40, 0.75)),
              ((0.50, 0.40), (0.55, 0.90))]),                                # three near -> keep
    (54, 66, [((0.20, 0.12), (0.50, 0.15)), ((0.60, 0.14), (0.90, 0.10))]),  # two far only
    (66, 84, [((0.15, 0.80), (0.75, 0.55)), ((0.80, 0.85), (0.30, 0.60))]),  # two near -> keep
    (84, 96, []),
    (96, 110, [((0.40, 0.60), (0.45, 0.62)), ((0.50, 0.65), (0.52, 0.66))]),  # two near, almost still
    (110, 120, []),
]
SCENARIO_SECONDS = 120


def _inside(poly, x, y) -> bool:
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _players_at(t: float):
    for s, e, players in SCHEDULE:
        if s <= t < e:
            u = (t - s) / (e - s)
            return [(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u) for a, b in players]
    return []


def _background(w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    img = np.empty((h, w, 3), np.int16)
    img[:] = (60, 140, 60)
    img += rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    lw = max(2, w // 400)
    img[int(0.30 * h):int(0.30 * h) + lw, :] = 235                 # halfway line
    for k in range(h):                                              # diagonal touchline
        x = int(0.15 * w + k * 0.6 * w / 1280)
        if 0 <= x < w - lw:
            img[k, x:x + lw] = 235
    return img


def _frame(bg: np.ndarray, t: float, flash: bool) -> np.ndarray:
    h, w = bg.shape[:2]
    img = bg.copy()
    for fx, fy in _players_at(t):
        scale = (0.25 + 0.75 * fy) * h / 720                       # perspective: farther = smaller
        ph, pw = max(4, int(110 * scale)), max(2, int(36 * scale))
        x0, y1 = int(fx * w - pw / 2), int(fy * h)
        y0 = max(0, y1 - ph)
        xa, xb = max(0, x0), max(0, x0 + pw)
        img[y0:y1, xa:xb] = (30, 30, 160)                          # shirt
        img[max(0, y1 - ph // 3):y1, xa:xb] = (20, 20, 20)         # shorts/legs
    if flash:
        img[:] = 255
    return img


def _write_beeps(path: str, seconds: float, sr: int = 48000) -> None:
    tt = np.arange(int(seconds * sr)) / sr
    gate = (np.mod(tt, BEEP_EVERY) < 0.020).astype(np.float64)
    pcm = (np.sin(2 * np.pi * 1000 * tt) * gate * 0.5 * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _encode(path: str, size: tuple[int, int], seconds: int, vcodec_args: list[str],
            vf: str | None = None, extra_out: list[str] | None = None, flash: bool = True) -> None:
    w, h = size
    bg = _background(w, h)
    wav = path + ".tmp.wav"
    _write_beeps(wav, seconds)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-map", "0:v", "-map", "1:a",
           *(["-vf", vf] if vf else []),
           *vcodec_args, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest",
           *(extra_out or []), path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(seconds * FPS):
            ft = i / FPS
            proc.stdin.write(_frame(bg, ft, flash and (ft % BEEP_EVERY) < 1e-9 + 0.5 / FPS).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    os.remove(wav)


def _truth() -> list[dict]:
    rows = []
    for s in range(SCENARIO_SECONDS):
        ps = _players_at(s + 0.5)
        near = sum(1 for x, y in ps if _inside(NEAR_POLY, x, y))
        rows.append({"t": s, "players": len(ps), "near": near, "keep": int(near >= 2)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/synthetic")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    p = lambda name: os.path.join(out, name)  # noqa: E731

    x264_gop1 = ["-c:v", "libx264", "-preset", "veryfast", "-g", "30", "-keyint_min", "30",
                 "-sc_threshold", "0", "-bf", "0", "-b:v", "4M"]
    _encode(p("syn_h264_gop1.mp4"), (1280, 720), SCENARIO_SECONDS, x264_gop1)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p("syn_h264_gop1.mp4"), "-map", "0",
                    "-c", "copy", "-f", "segment", "-segment_time", "60", "-reset_timestamps", "1",
                    p("syn_split_%d.tmp.mp4")], check=True)
    os.replace(p("syn_split_0.tmp.mp4"), p("syn_split_a.mp4"))
    os.replace(p("syn_split_1.tmp.mp4"), p("syn_split_b.mp4"))

    # analysis scenario without the A/V-sync flashes (a white frame on every other keyframe would look like action)
    _encode(p("syn_scene.mp4"), (1280, 720), SCENARIO_SECONDS, x264_gop1, flash=False)

    small, secs = (640, 360), 40
    _encode(p("syn_h264_bframes.mp4"), small, secs,
            ["-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-keyint_min", "30",
             "-sc_threshold", "0", "-bf", "2", "-b:v", "1500k"])
    _encode(p("syn_hevc_opengop.mp4"), small, secs,
            ["-c:v", "libx265", "-preset", "ultrafast", "-tag:v", "hvc1", "-b:v", "1M",
             "-x265-params", "keyint=30:min-keyint=30:open-gop=1:bframes=3:scenecut=0:log-level=error"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation:v:0", "90",
                    "-i", p("syn_h264_bframes.mp4"), "-map", "0", "-c", "copy", p("syn_rot90.mp4")],
                   check=True)
    _encode(p("syn_vfr.mp4"), small, secs,
            ["-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-keyint_min", "30",
             "-sc_threshold", "0", "-bf", "0", "-b:v", "1500k", "-fps_mode", "vfr"],
            vf="select='not(eq(mod(n\\,7)\\,3))',setpts='PTS+0.008*sin(N*1.3)/TB'")

    with open(p("syn_truth.json"), "w") as f:
        json.dump({"near_polygon": NEAR_POLY, "schedule": SCHEDULE, "per_second": _truth()}, f, indent=1)
    print("synthetic set written to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
