"""Tính EER — DÙNG CHUNG (A và B).

compute_det_curve / compute_eer: port NGUYÊN VĂN từ clovaai/aasist evaluation.py
(https://github.com/clovaai/aasist, commit a04c986, MIT, Copyright (c) 2021-present NAVER Corp.).
Hàm gốc coi "target" = bona fide và điểm CAO = bona fide.

Quy ước chung của repo: điểm CAO = GIẢ. eer_from_labels() đổi dấu ĐÚNG MỘT CHỖ trước khi
gọi hàm gốc. min t-DCF và chấm file điểm nằm trong eval/score.py (B phụ trách).
"""
from __future__ import annotations

import numpy as np


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def eer_from_labels(scores, labels) -> tuple[float, float]:
    """EER (%) theo quy ước chung: labels 0 = thật, 1 = giả; điểm cao = giả.

    Trả (eer_percent, threshold) với threshold theo dấu của `scores`
    (điểm > threshold -> coi là giả, xấp xỉ tại điểm EER).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} != labels {labels.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores có NaN/inf")
    bona, spoof = scores[labels == 0], scores[labels == 1]
    if bona.size == 0 or spoof.size == 0:
        raise ValueError("cần cả mẫu thật và mẫu giả để tính EER")
    # Đổi dấu: hàm gốc cần điểm cao = bona fide.
    eer, thr = compute_eer(-bona, -spoof)
    return float(eer * 100.0), float(-thr)
