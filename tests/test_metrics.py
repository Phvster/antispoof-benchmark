"""Quy ước (chốt 26/11/2026): nhãn 1 = thật, 0 = giả; điểm cao = THẬT; không đổi dấu."""
import numpy as np
import pytest

from common.metrics import compute_eer, eer_from_labels


def test_high_score_is_bonafide():
    """Điểm cao = thật: thật được điểm cao, giả điểm thấp -> EER = 0; đảo dấu -> EER = 100."""
    labels = np.array([1] * 50 + [0] * 50)
    scores = np.where(labels == 1, 5.0, -5.0) + np.random.default_rng(0).normal(0, 0.1, 100)
    eer, thr = eer_from_labels(scores, labels)
    assert eer == 0.0
    assert -5.5 < thr < 5.5
    eer_flip, _ = eer_from_labels(-scores, labels)
    assert eer_flip == 100.0


def test_random_is_about_50():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 20000)
    eer, _ = eer_from_labels(rng.normal(size=20000), labels)
    assert 47 < eer < 53


def test_flipped_sign_gives_100_minus_eer():
    rng = np.random.default_rng(2)
    labels = np.array([1] * 1000 + [0] * 1000)
    scores = labels * 1.0 + rng.normal(0, 1, 2000)  # cao = thật
    eer, _ = eer_from_labels(scores, labels)
    eer_flip, _ = eer_from_labels(-scores, labels)
    assert eer < 40
    assert abs((eer + eer_flip) - 100) < 1.0


def test_same_as_clovaai_no_sign_flip():
    """eer_from_labels == compute_eer(bona, spoof) của clovaai, không đổi dấu."""
    rng = np.random.default_rng(3)
    bona, spoof = rng.normal(1, 1, 300), rng.normal(-1, 1, 700)
    eer_ref, thr_ref = compute_eer(bona, spoof)
    eer, thr = eer_from_labels(np.r_[bona, spoof], np.r_[np.ones(300), np.zeros(700)])
    assert eer == pytest.approx(eer_ref * 100)
    assert thr == pytest.approx(thr_ref)


def test_rejects_nan_and_single_class():
    with pytest.raises(ValueError):
        eer_from_labels([0.1, np.nan], [0, 1])
    with pytest.raises(ValueError):
        eer_from_labels([0.1, 0.2], [1, 1])


def test_class_weights_follow_labels():
    """CE [0.1, 0.9]: nhãn 0 (giả) trọng số 0.1, nhãn 1 (thật, lớp ít) trọng số 0.9."""
    import torch
    from common.data import BONAFIDE, SPOOF
    from common.train import CLASS_WEIGHTS, make_criterion
    assert (BONAFIDE, SPOOF) == (1, 0)
    assert CLASS_WEIGHTS == [0.1, 0.9]
    crit = make_criterion("cpu")
    assert crit.weight[SPOOF].item() == pytest.approx(0.1)
    assert crit.weight[BONAFIDE].item() == pytest.approx(0.9)
    # 1 mẫu thật đoán sai nặng + 1 mẫu giả đoán đúng: loss do mẫu thật chiếm trọng số 0.9
    logits = torch.tensor([[5.0, -5.0], [5.0, -5.0]])  # cả hai nghiêng về "giả"
    y = torch.tensor([BONAFIDE, SPOOF])
    per = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    expected = (0.9 * per[0] + 0.1 * per[1]) / (0.9 + 0.1)
    assert crit(logits, y).item() == pytest.approx(expected.item())
