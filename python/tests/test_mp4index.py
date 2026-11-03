import struct

import pytest

from futsalcut.mp4index import read_index
from futsalcut.probe import probe
from futsalcut.sampler import display_rotation
from futsalcut.verify import _rotation

# Many files carry a second hdlr, the data handler, in minf after the media handler in mdia; a reader must not
# take it for the track's handler. A layout is (component type of the media hdlr, component type and subtype
# of the data hdlr); None is the plain ISO layout without one.
DATA_HANDLER_LAYOUTS = {
    "quicktime-alis": (b"mhlr", b"dhlr", b"alis"),
    "quicktime-url": (b"mhlr", b"dhlr", b"url "),  # what ffmpeg's mov muxer writes
    "zero-component-alis": (bytes(4), bytes(4), b"alis"),  # zero component types, as one mp4 encoder writes them
}
LAYOUTS = [None, *DATA_HANDLER_LAYOUTS.values()]
LAYOUT_IDS = ["iso", *DATA_HANDLER_LAYOUTS]


def box(typ: bytes, *parts: bytes) -> bytes:
    payload = b"".join(parts)
    return struct.pack(">I4s", 8 + len(payload), typ) + payload


def full_box(typ: bytes, *parts: bytes, flags: int = 0) -> bytes:
    return box(typ, struct.pack(">I", flags), *parts)


def hdlr(component: bytes, subtype: bytes) -> bytes:
    return full_box(b"hdlr", component, subtype, bytes(12), b"\0")


def tkhd(track_id: int, rotation: int) -> bytes:
    a, b = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[rotation]
    matrix = struct.pack(">9i", a << 16, b << 16, 0, -b << 16, a << 16, 0, 0, 0, 1 << 30)
    return full_box(b"tkhd", struct.pack(">IIIII", 0, 0, track_id, 0, 0), bytes(16), matrix, bytes(8))


def mdhd(timescale: int, duration: int) -> bytes:
    return full_box(b"mdhd", struct.pack(">IIII", 0, 0, timescale, duration), struct.pack(">HH", 0x55C4, 0))


def video_entry(width: int, height: int) -> bytes:
    fields = (bytes(6) + struct.pack(">H", 1) + bytes(16) + struct.pack(">HH", width, height)
              + struct.pack(">IIIH", 0x480000, 0x480000, 0, 1) + bytes(32) + struct.pack(">Hh", 24, -1))
    return box(b"avc1", fields, box(b"avcC", b"\x01\x64\x00\x1f"))


def audio_entry(channels: int, sample_rate: int) -> bytes:
    fields = (bytes(6) + struct.pack(">H", 1) + bytes(8) + struct.pack(">HHHH", channels, 16, 0, 0)
              + struct.pack(">I", sample_rate << 16))
    return box(b"mp4a", fields)


def sample_tables(entry: bytes, count: int, delta: int, sync: list[int] | None) -> bytes:
    parts = [full_box(b"stsd", struct.pack(">I", 1), entry),
             full_box(b"stts", struct.pack(">III", 1, count, delta)),
             full_box(b"stsz", struct.pack(">II", 0, count), struct.pack(f">{count}I", *[100] * count))]
    if sync:
        parts.append(full_box(b"stss", struct.pack(f">I{len(sync)}I", len(sync), *sync)))
    return box(b"stbl", *parts)


def track(track_id: int, subtype: bytes, rotation: int, layout: tuple[bytes, bytes, bytes] | None,
          media_handler: bool) -> bytes:
    if subtype == b"vide":
        header = full_box(b"vmhd", bytes(8), flags=1)
        entry, timescale, delta, sync = video_entry(1920, 1080), 1000, 40, [1, 3]
    else:
        header = full_box(b"smhd", bytes(4))
        entry, timescale, delta, sync = audio_entry(2, 48000), 48000, 1024, None
    dinf = box(b"dinf", full_box(b"dref", struct.pack(">I", 1), full_box(b"url ", flags=1)))
    data = hdlr(layout[1], layout[2]) if layout else b""
    minf = box(b"minf", header, data, dinf, sample_tables(entry, 4, delta, sync))
    media = hdlr(layout[0] if layout else bytes(4), subtype) if media_handler else b""
    return box(b"trak", tkhd(track_id, rotation), box(b"mdia", mdhd(timescale, 4 * delta), media, minf))


def write_movie(path, layout=None, media_handler: bool = True) -> str:
    """Movie header only (a rotated 1920x1080 video track and a stereo audio track), no media data."""
    brand = b"qt  " if layout and layout[0] == b"mhlr" else b"isom"
    mvhd = full_box(b"mvhd", struct.pack(">IIII", 3_850_000_000, 3_850_000_000, 1000, 160), bytes(80))
    moov = box(b"moov", mvhd, track(1, b"vide", 90, layout, media_handler),
               track(2, b"soun", 0, layout, media_handler))
    path.write_bytes(box(b"ftyp", brand, bytes(4), brand) + moov)
    return str(path)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_media_handler_comes_from_mdia(tmp_path, layout):
    idx = read_index(write_movie(tmp_path / "m.mov", layout))
    assert [t.handler for t in idx.tracks] == ["vide", "soun"]
    v, a = idx.video, idx.audio
    assert (v.codec, v.width, v.height, v.rotation) == ("avc1", 1920, 1080, 90)
    assert (a.codec, a.channels, a.sample_rate) == ("mp4a", 2, 48000)


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_callers_see_the_video_track(tmp_path, layout):
    path = write_movie(tmp_path / "m.mov", layout)
    assert display_rotation(path) == 90
    assert _rotation(path) == 90.0
    p = probe(path)
    assert (p.video_codec, p.width, p.height, p.rotation) == ("avc1", 1920, 1080, 90)
    assert (p.audio_codec, p.audio_sample_rate, p.audio_channels) == ("mp4a", 48000, 2)


def test_probe_does_not_depend_on_the_layout(tmp_path):
    def summary(i, layout):
        d = probe(write_movie(tmp_path / f"{i}.mov", layout)).to_dict()
        return {k: v for k, v in d.items() if k not in ("path", "file_size")}
    assert all(summary(i, layout) == summary(0, None) for i, layout in enumerate(LAYOUTS[1:], 1))


@pytest.mark.parametrize("layout", list(DATA_HANDLER_LAYOUTS.values()), ids=list(DATA_HANDLER_LAYOUTS))
def test_data_handler_alone_is_not_a_media_handler(tmp_path, layout):
    idx = read_index(write_movie(tmp_path / "m.mov", layout, media_handler=False))
    assert [t.handler for t in idx.tracks] == ["", ""]
    assert idx.video is None and idx.audio is None
