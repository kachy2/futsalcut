import numpy as np

from futsalcut.select import PostParams, kept_seconds, select, smooth, threshold_for_budget


def test_smooth_two_of_three():
    d = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0], bool)
    assert smooth(d, 3, 2).astype(int).tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_select_merge_drop_pad():
    t = np.arange(60, dtype=float)
    s = np.zeros(60)
    s[10:16] = 1   # 6 s run
    s[19:22] = 1   # 3 s run, gap 3 s -> merged with the first
    s[40:42] = 1   # 2 s run -> dropped (min_len 4)
    p = PostParams(smooth_window=1, smooth_need=1, merge_gap=5, min_len=4, pad_before=2, pad_after=2)
    assert select(s, t, 0.5, p) == [(8.0, 24.0)]


def test_budget_monotone():
    rng = np.random.default_rng(0)
    t = np.arange(600, dtype=float)
    s = rng.random(600)
    p = PostParams()
    thr, iv = threshold_for_budget(s, t, 120.0, p)
    assert kept_seconds(iv) <= 120.0
    lower = select(s, t, float(np.nextafter(thr, -np.inf)), p)
    assert kept_seconds(lower) >= kept_seconds(iv)


def test_snap_to_keyframe_grid_with_jitter():
    from futsalcut.select import snap_to_samples
    t = np.array([0.0, 0.9997, 1.9993, 3.0326, 4.0322, 5.0319, 6.0315])  # jittery ~1 s keyframes
    # boundaries computed as sample time +/- 2.0 s land a few ms off the grid
    iv = snap_to_samples([(3.0326 - 2.0, 4.0322 + 2.0)], t)
    assert iv == [(0.9997, 6.0315)]


def test_select_output_is_on_sample_grid():
    t = np.cumsum(np.r_[0, np.full(99, 0.9997)])
    s = np.zeros(100)
    s[40:50] = 1
    iv = select(s, t, 0.5, PostParams())
    starts = {round(x, 6) for x in t}
    ends = {round(x, 6) for x in np.r_[t[1:], t[-1] + 1]}
    assert all(round(a, 6) in starts and round(b, 6) in ends for a, b in iv)
