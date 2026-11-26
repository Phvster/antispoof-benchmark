import numpy as np
import pytest

from common.metrics import compute_eer, eer_from_labels


def test_perfect_separation_is_zero():
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.where(labels == 1, 5.0, -5.0) + np.random.default_rng(0).normal(0, 0.1, 100)
    eer, thr = eer_from_labels(scores, labels)
    assert eer == 0.0
    assert -5.5 < thr < 5.5


def test_random_is_about_50():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 20000)
    eer, _ = eer_from_labels(rng.normal(size=20000), labels)
    assert 47 < eer < 53


def test_flipped_sign_gives_100_minus_eer():
    rng = np.random.default_rng(2)
    labels = np.array([0] * 1000 + [1] * 1000)
    scores = labels * 1.0 + rng.normal(0, 1, 2000)  # cao = giả
    eer, _ = eer_from_labels(scores, labels)
    eer_flip, _ = eer_from_labels(-scores, labels)
    assert eer < 40
    assert abs((eer + eer_flip) - 100) < 1.0


def test_matches_clovaai_convention():
    """Hàm gốc: điểm cao = bona fide. eer_from_labels(s) == compute_eer(-s_bona, -s_spoof)."""
    rng = np.random.default_rng(3)
    bona, spoof = rng.normal(-1, 1, 300), rng.normal(1, 1, 700)
    eer_ref, _ = compute_eer(-bona, -spoof)
    eer, _ = eer_from_labels(np.r_[bona, spoof], np.r_[np.zeros(300), np.ones(700)])
    assert eer == pytest.approx(eer_ref * 100)


def test_rejects_nan_and_single_class():
    with pytest.raises(ValueError):
        eer_from_labels([0.1, np.nan], [0, 1])
    with pytest.raises(ValueError):
        eer_from_labels([0.1, 0.2], [0, 0])
