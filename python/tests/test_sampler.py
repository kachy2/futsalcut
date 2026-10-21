import numpy as np
import pytest

from futsalcut.sampler import box_downscale, target_size


def _naive_box(y: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    H, W = y.shape
    out = np.zeros((out_h, out_w), np.uint8)
    for r in range(out_h):
        r0, r1 = r * H // out_h, (r + 1) * H // out_h
        for c in range(out_w):
            c0, c1 = c * W // out_w, (c + 1) * W // out_w
            block = y[r0:r1, c0:c1].astype(np.int64)
            n = block.size
            out[r, c] = (int(block.sum()) + n // 2) // n
    return out


@pytest.mark.parametrize("shape,out", [((36, 64), (16, 9)), ((37, 61), (16, 9)), ((30, 40), (40, 30)),
                                       ((90, 160), (64, 36))])
def test_box_downscale_matches_naive(shape, out):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 256, size=shape, dtype=np.uint8)
    w, h = out
    np.testing.assert_array_equal(box_downscale(y, w, h), _naive_box(y, w, h))


def test_box_downscale_rounds_half_up():
    y = np.array([[0, 1], [0, 1]], np.uint8)  # mean 0.5 -> 1
    assert box_downscale(y, 1, 1)[0, 0] == 1


def test_box_downscale_rejects_upscale():
    with pytest.raises(ValueError):
        box_downscale(np.zeros((4, 4), np.uint8), 8, 8)


@pytest.mark.parametrize("src,expected", [((3840, 2160), (640, 360)), ((1920, 1080), (640, 360)),
                                          ((2160, 3840), (360, 640)), ((4032, 3024), (640, 480))])
def test_target_size(src, expected):
    assert target_size(*src) == expected


def test_rotate_cw():
    from futsalcut.sampler import rotate_cw
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)          # [[0,1,2],[3,4,5]]
    np.testing.assert_array_equal(rotate_cw(a, 90), [[3, 0], [4, 1], [5, 2]])
    np.testing.assert_array_equal(rotate_cw(a, 180), [[5, 4, 3], [2, 1, 0]])
    np.testing.assert_array_equal(rotate_cw(a, 270), [[2, 5], [1, 4], [0, 3]])
    np.testing.assert_array_equal(rotate_cw(a, 0), a)
