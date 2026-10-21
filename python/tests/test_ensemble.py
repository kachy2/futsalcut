import numpy as np

from futsalcut.ensemble import Calibrator, Ensemble, pick_questions, quick_calibrate


def test_calibrator_is_monotone_bounded_and_invertible():
    rng = np.random.default_rng(0)
    s = rng.random(500) * 10
    y = (rng.random(500) < s / 10).astype(int)
    c = Calibrator.fit(s, y)
    grid = np.linspace(0, 10, 101)
    p = c(grid)
    assert np.all(np.diff(p) >= -1e-12)
    assert 0 < p.min() and p.max() < 1
    assert abs(c(np.array([c.threshold_for(0.5)]))[0] - 0.5) < 1e-9


def test_ensemble_beats_noise_feature():
    rng = np.random.default_rng(1)
    y = (rng.random(800) < 0.2).astype(int)
    good = np.maximum(0, y * 2 + rng.normal(0, 0.5, 800))
    noise = np.abs(rng.normal(0, 1, 800))
    e = Ensemble.fit({"good": good, "noise": noise}, y, ["good", "noise"])
    assert abs(e.weights[0]) > 3 * abs(e.weights[1])
    p = e.prob({"good": good, "noise": noise})
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3


def test_questions_are_spread_and_quick_calibration_follows_answers():
    t = np.arange(300, dtype=float)
    a = np.where(t < 150, 0.9, 0.1)
    b = np.where(t < 150, 0.1, 0.9)  # disagrees everywhere
    q = pick_questions({"a": a, "b": b}, t, k=5, min_gap_s=20)
    assert len(q) == 5 and min(np.diff(q)) >= 20
    answers = {i: bool(t[i] >= 150) for i in q}  # the user agrees with b
    name, thr = quick_calibrate({"a": a, "b": b}, answers, default="a")
    assert name == "b"
