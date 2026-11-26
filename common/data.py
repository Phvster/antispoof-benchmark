"""Đọc protocol chung và Dataset train/dev/eval — DÙNG CHUNG (A và B).

Định dạng protocol (CSV, có header):
    file_id,speaker_id,label,attack_id,subset
- label: 0 = thật (bona fide), 1 = giả (spoof).
- attack_id: A01..A19 (hoặc mã tấn công của bộ khác); file thật ghi "-".
- subset: train | dev | eval.

Tính ngẫu nhiên của cắt train (pad_random):
- Mỗi worker có rng riêng = default_rng([seed, worker_id, epoch]), tạo trong worker_init_fn.
- num_workers=0 dùng đúng rng của "worker 0" -> cùng seed/epoch thì cùng crop.
- Gọi dataset.set_epoch(epoch) TRƯỚC khi tạo iterator mỗi epoch (không dùng
  persistent_workers, để worker được tạo lại và nhận epoch mới).
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.audio import load_audio, prepare

COLUMNS = ["file_id", "speaker_id", "label", "attack_id", "subset"]


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_protocol(path: str) -> pd.DataFrame:
    """Đọc + kiểm tra protocol: đúng cột, nhãn ∈ {0,1}, file_id không trùng."""
    df = pd.read_csv(path, dtype={"file_id": str, "speaker_id": str,
                                  "attack_id": str, "subset": str},
                     keep_default_na=False)
    if list(df.columns) != COLUMNS:
        raise ValueError(f"{path}: cột {list(df.columns)} != {COLUMNS}")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError(f"{path}: nhãn phải là 0/1, có {sorted(df['label'].unique())}")
    dup = df["file_id"].duplicated()
    if dup.any():
        raise ValueError(f"{path}: file_id trùng, ví dụ {df.loc[dup, 'file_id'].iloc[0]}")
    return df


class AntiSpoofDataset(Dataset):
    """Trả (wave: FloatTensor[T], label: int, file_id: str).

    mode="train": cắt ngẫu nhiên; "window": đoạn đầu; "full": cả file (dùng batch_size=1).
    ext: ".flac" (đọc FLAC) hoặc ".npy" (int16 đã giải mã sẵn).
    """

    def __init__(self, protocol: str | pd.DataFrame, audio_dir: str, ext: str = ".flac",
                 mode: str = "window", seed: int = 0):
        self.df = read_protocol(protocol) if isinstance(protocol, str) else protocol
        self.audio_dir = audio_dir
        self.ext = ext
        self.mode = mode
        self.seed = seed
        self.epoch = 0
        self.file_ids = self.df["file_id"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.init_rng(worker_id=0)

    def init_rng(self, worker_id: int) -> None:
        self.rng = np.random.default_rng([self.seed, worker_id, self.epoch])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.init_rng(worker_id=0)  # cho num_workers=0; worker thật tự gọi lại init_rng

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, i: int):
        fid = self.file_ids[i]
        x = load_audio(os.path.join(self.audio_dir, fid + self.ext))
        x = prepare(x, self.mode, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x)), self.labels[i], fid


def worker_init_fn(worker_id: int) -> None:
    """rng riêng cho mỗi worker: default_rng([seed, worker_id, epoch]).

    Seed thêm numpy/random toàn cục (phòng khi augment sau này dùng) từ torch.initial_seed().
    """
    info = torch.utils.data.get_worker_info()
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
    ds = info.dataset
    if hasattr(ds, "init_rng"):
        ds.init_rng(worker_id)
