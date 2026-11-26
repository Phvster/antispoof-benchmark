"""Nạp và chuẩn hóa audio — DÙNG CHUNG cho mọi model (A và B).

Quy ước (CONVENTIONS.md, mục "Tiền xử lý"):
- 16 kHz, mono, PCM 16-bit chia 32768 về [-1, 1].
- Train: cắt ngẫu nhiên NB_SAMP mẫu, lặp nếu ngắn  -> pad_random / mode="train".
- Chấm chế độ cửa sổ: NB_SAMP mẫu đầu, lặp nếu ngắn -> pad        / mode="window".
- Chấm chế độ cả file: toàn bộ file                 ->              mode="full".

pad() và pad_random() port từ clovaai/aasist data_utils.py
(https://github.com/clovaai/aasist, commit a04c986, MIT, Copyright (c) 2021-present NAVER Corp.).

KHÔNG viết số 64000/64600 ở nơi khác trong repo: luôn import NB_SAMP từ đây.
"""
from __future__ import annotations

import warnings

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000

# 64.600 mẫu (~4,04 s) = nb_samp của code AASIST — đã chốt trong tài liệu quy ước.
NB_SAMP = 64600

MODES = ("train", "window", "full")


def load_audio(path: str) -> np.ndarray:
    """Đọc file audio -> np.float32 1-D, 16 kHz, mono, trong [-1, 1].

    - File âm thanh (flac/wav...): đọc dạng int16 rồi chia 32768 (đúng câu chữ quy ước).
      Nhiều kênh -> trung bình. Sai sample rate -> resample_poly (scipy).
    - File .npy: mảng int16 1-D đã giải mã sẵn bằng tools/decode_to_npy.py
      (tool đó chỉ nhận file 16 kHz mono), chia 32768 y hệt -> kết quả giống hệt FLAC.
    """
    if str(path).endswith(".npy"):
        x = np.load(path, allow_pickle=False)
        if x.dtype != np.int16 or x.ndim != 1:
            raise ValueError(f"{path}: .npy phải là int16 1-D, nhận {x.dtype} {x.shape}")
        return x.astype(np.float32) / 32768.0
    info = sf.info(path)
    if info.subtype != "PCM_16":
        warnings.warn(f"{path}: subtype {info.subtype} khác PCM_16, vẫn quy về int16.")
    x, sr = sf.read(path, dtype="int16", always_2d=True)
    x = x.astype(np.float32) / 32768.0
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
        x = np.clip(x, -1.0, 1.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def pad(x: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """Chế độ cửa sổ: lấy max_len mẫu đầu; ngắn hơn thì lặp (tile) rồi cắt.

    Port nguyên logic clovaai `data_utils.pad`.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def pad_random(x: np.ndarray, max_len: int = NB_SAMP,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Train: cắt ngẫu nhiên max_len mẫu; ngắn hơn thì lặp (tile) rồi cắt.

    Port từ clovaai `data_utils.pad_random`, khác hai điểm:
    - dùng `rng` truyền vào thay cho np.random toàn cục (Dataset truyền
      default_rng([seed, epoch, idx]) theo quy ước);
    - x_len == max_len: bản gốc gọi np.random.randint(0) và lỗi; ở đây trả nguyên x.
    """
    rng = rng if rng is not None else np.random.default_rng()
    x_len = x.shape[0]
    if x_len > max_len:
        stt = int(rng.integers(x_len - max_len))  # như randint(x_len - max_len) gốc
        return x[stt:stt + max_len]
    if x_len == max_len:
        return x
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def prepare(x: np.ndarray, mode: str, rng: np.random.Generator | None = None,
            max_len: int = NB_SAMP) -> np.ndarray:
    """Áp cắt/lặp theo chế độ: "train" | "window" | "full"."""
    if mode == "train":
        return pad_random(x, max_len, rng)
    if mode == "window":
        return pad(x, max_len)
    if mode == "full":
        return x
    raise ValueError(f"mode phải thuộc {MODES}, nhận {mode!r}")
