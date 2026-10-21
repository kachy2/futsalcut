import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from metrics import binary_gt, classifier_prf, edit_metrics, intervals_to_mask, runs  # noqa: E402


def test_binary_modes():
    labels = list("02M1F?b")
    assert binary_gt(labels, "strict").tolist() == [0, 1, 0, 0, 0, -1, 0]
    assert binary_gt(labels, "half").tolist() == [0, 1, 1, 0, 0, -1, -1]


def test_classifier_prf_ignores_unknown():
    gt = np.array([1, 1, 0, 0, -1], np.int8)
    pred = np.array([1, 0, 1, 0, 1], bool)
    r = classifier_prf(gt, pred)
    assert (r.tp, r.fp, r.fn) == (1, 1, 1)


def test_runs_and_intervals():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    t = np.arange(10, dtype=float)
    m = intervals_to_mask([(2.0, 5.0)], t)
    assert m.nonzero()[0].tolist() == [2, 3, 4]


def test_edit_band_does_not_penalise_padding():
    t = np.arange(30, dtype=float)
    gt = np.zeros(30, np.int8)
    gt[10:18] = 1                       # 8 s episode
    em = edit_metrics(gt, [(8.0, 20.0)], t, 2.0, 2.0)   # padded by 2 s each side
    assert em.prf_band.precision == 1.0 and em.prf_band.recall == 1.0
    assert em.event_recall == 1.0 and em.false_scene_rate == 0.0
