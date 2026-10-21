"""Lossless cut-and-join: copy compressed packets of selected segments into one MP4.

Rules (shared with the web exporter, plan §3.5):
- A segment starts at the last keyframe at/before its requested start and ends at the first
  keyframe at/after its requested end (or the end of the stream). The selection step normally
  passes keyframe times already, so both are exact.
- Video packets are copied in decode order from the start keyframe; packets whose PTS precedes it
  (open-GOP leading pictures) are dropped; copying stops before the end keyframe.
- HEVC: an end-of-sequence NAL is appended to each segment's last packet, so the next segment's
  CRA starts a new coded video sequence (otherwise POC continuity breaks and frames reorder).
- Timestamps are rebased in integer ticks so the output video timeline is continuous.
- Audio uses the same time origin as video (the video stream start). It is laid out gaplessly:
  per segment the source frame nearest to the ideal start (video start + current audio lead) is
  taken, and the frame count is chosen by error diffusion against the actual segment length, so
  the A/V error at every cut stays within half an audio frame and never accumulates. Audio is
  released no further than the video already written, which keeps the file interleaved.
- Only the first video and first AAC track are copied; container metadata (e.g. GPS) is not.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction

import av

SNAP_TOL = 0.05  # seconds: a boundary within 50 ms of a keyframe means that keyframe


def hevc_eos(extradata: bytes | None) -> bytes:
    """End-of-sequence NAL (type 36, temporal id 1) with the stream's NAL length prefix size."""
    nal_len_size = (extradata[21] & 0x3) + 1 if extradata and len(extradata) > 21 else 4
    return (2).to_bytes(nal_len_size, "big") + b"\x48\x01"


@dataclass
class Segment:
    file_index: int
    start_s: float  # seconds from the video stream start
    end_s: float


@dataclass
class RemuxReport:
    segments: int = 0
    video_packets: int = 0
    audio_packets: int = 0
    dropped_leading: int = 0
    dts_bumps: int = 0
    eos_inserted: int = 0
    video_duration_s: float = 0.0
    audio_duration_s: float = 0.0
    max_cut_av_error_s: float = 0.0
    wall_s: float = 0.0
    per_segment: list[dict] = field(default_factory=list)


def _aac_stream(container):
    return next((s for s in container.streams.audio if s.codec_context.name == "aac"), None)


def _with_data(pkt: av.Packet, data: bytes) -> av.Packet:
    out = av.Packet(data)
    out.pts, out.dts, out.duration = pkt.pts, pkt.dts, pkt.duration
    out.time_base = pkt.time_base
    out.is_keyframe = pkt.is_keyframe
    return out


def remux(sources: list[str], segments: list[Segment], out_path: str) -> RemuxReport:
    rep = RemuxReport()
    wall0 = time.perf_counter()
    inputs = [av.open(p) for p in sources]
    try:
        v0 = inputs[0].streams.video[0]
        a0 = _aac_stream(inputs[0])
        is_hevc = v0.codec_context.name == "hevc"
        eos = hevc_eos(v0.codec_context.extradata) if is_hevc else b""
        out = av.open(out_path, "w", format="mp4")
        ov = out.add_stream_from_template(v0)
        oa = out.add_stream_from_template(a0) if a0 is not None else None
        tb: Fraction = v0.time_base
        sr = a0.codec_context.sample_rate if a0 is not None else 0
        flen = (a0.codec_context.frame_size or 1024) if a0 is not None else 0
        frame_s = Fraction(flen, sr) if sr else Fraction(0)

        cum_v = Fraction(0)  # seconds of video written
        frames_a = 0         # audio frames written
        last_out_dts: int | None = None

        for si, seg in enumerate(segments):
            c = inputs[seg.file_index]
            vs = c.streams.video[0]
            as_ = _aac_stream(c)
            if vs.time_base != tb:
                raise ValueError("all sources must share the video time base")
            vstart = vs.start_time or 0
            origin = Fraction(vstart) * tb  # common time origin (s) for video and audio
            atb = as_.time_base if as_ is not None else None
            end_pts = vstart + int(round(Fraction(seg.end_s - SNAP_TOL) / tb))
            c.seek(vstart + int(round(Fraction(seg.start_s + SNAP_TOL) / tb)), stream=vs, backward=True, any_frame=False)

            info: dict = {"file": seg.file_index, "start": seg.start_s, "end": seg.end_s}
            key_pts = None
            stop_pts = None
            shift = 0
            last_pts_end = None
            held: av.Packet | None = None  # last video packet, held back for a possible EOS
            pending: deque[av.Packet] = deque()
            a_taken = 0
            a_from = Fraction(0)
            a_lead = Fraction(0)

            def emit_video(pkt: av.Packet) -> None:
                nonlocal last_out_dts, shift
                out_dts = pkt.dts + shift
                if last_out_dts is not None and out_dts <= last_out_dts:
                    bump = last_out_dts + 1 - out_dts
                    shift += bump
                    out_dts += bump
                    rep.dts_bumps += 1
                pkt.pts += shift
                pkt.dts = out_dts
                last_out_dts = out_dts
                pkt.stream = ov
                out.mux(pkt)
                rep.video_packets += 1

            def flush_audio(limit_frames: int) -> None:
                nonlocal a_taken
                while pending and a_taken < limit_frames:
                    ap = pending.popleft()
                    if a_taken == 0:
                        t = Fraction(ap.pts) * atb - origin
                        info["audio_start_err_ms"] = round(float(t - (a_from + frame_s / 2)) * 1000, 2)
                    ap.pts = ap.dts = int(round(Fraction(frames_a + a_taken) * frame_s / atb))
                    ap.stream = oa
                    out.mux(ap)
                    a_taken += 1
                    rep.audio_packets += 1

            def audio_allowed() -> int:
                # frames whose output start lies before the video written so far in this segment
                if key_pts is None or last_pts_end is None:
                    return 0
                written = cum_v + Fraction(last_pts_end - key_pts) * tb
                return max(0, int(written / frame_s) - frames_a)

            streams = [vs] + ([as_] if as_ is not None else [])
            for pkt in c.demux(*streams):
                if pkt.dts is None:
                    continue
                if pkt.stream is vs:
                    if stop_pts is not None:
                        continue
                    if key_pts is None:
                        if not pkt.is_keyframe:
                            continue
                        key_pts = pkt.pts
                        shift = int(round(cum_v / tb)) - key_pts
                        if as_ is not None:
                            a_lead = Fraction(frames_a) * frame_s - cum_v  # audio ahead of video (s)
                            a_from = Fraction(key_pts - vstart) * tb + a_lead - frame_s / 2
                            pending = deque(p for p in pending if Fraction(p.pts) * atb - origin >= a_from)
                    elif pkt.is_keyframe and pkt.pts >= end_pts:
                        stop_pts = pkt.pts
                        continue
                    if pkt.pts < key_pts:
                        rep.dropped_leading += 1
                        continue
                    pend = pkt.pts + (pkt.duration or 0)
                    last_pts_end = pend if last_pts_end is None else max(last_pts_end, pend)
                    if held is not None:
                        emit_video(held)
                    held = pkt
                    if as_ is not None:
                        flush_audio(audio_allowed())
                elif as_ is not None:
                    t = Fraction(pkt.pts) * atb - origin
                    if key_pts is None or t >= a_from:
                        pending.append(pkt)
                    flush_audio(audio_allowed())
                if stop_pts is not None and as_ is None:
                    break
                if stop_pts is not None and as_ is not None:
                    seg_end = stop_pts
                    need = int(round((cum_v + Fraction(seg_end - key_pts) * tb) / frame_s)) - frames_a
                    if a_taken + len(pending) >= need:
                        break

            if key_pts is None:
                info["skipped"] = "no keyframe at/after start"
                rep.per_segment.append(info)
                continue
            seg_end = stop_pts if stop_pts is not None else last_pts_end
            seg_dur = Fraction(seg_end - key_pts) * tb
            if held is not None:
                if is_hevc and si < len(segments) - 1:
                    held = _with_data(held, bytes(held) + eos)
                    rep.eos_inserted += 1
                emit_video(held)
            if as_ is not None:
                need = int(round((cum_v + seg_dur) / frame_s)) - frames_a
                flush_audio(need)
                frames_a += a_taken
                err = abs(Fraction(frames_a) * frame_s - (cum_v + seg_dur))
                rep.max_cut_av_error_s = max(rep.max_cut_av_error_s, float(err))
            cum_v += seg_dur
            rep.segments += 1
            info.update(key_start=float(Fraction(key_pts - vstart) * tb), key_end=float(Fraction(seg_end - vstart) * tb),
                        dur=float(seg_dur), audio_frames=a_taken)
            rep.per_segment.append(info)

        out.close()
        rep.video_duration_s = float(cum_v)
        rep.audio_duration_s = float(frames_a * frame_s) if sr else 0.0
    finally:
        for c in inputs:
            c.close()
    rep.wall_s = time.perf_counter() - wall0
    return rep
