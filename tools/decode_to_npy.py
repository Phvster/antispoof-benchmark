"""Giải mã FLAC -> .npy int16 1-D (một lần), để train không tốn CPU giải mã FLAC.

- Chỉ nhận file 16 kHz mono (đúng ASVspoof 2019 LA); khác thì báo lỗi, KHÔNG resample,
  để load_audio(.npy) cho kết quả giống hệt load_audio(.flac).
- Giá trị int16 giữ nguyên; load_audio chia 32768 như với FLAC.
- File đã có thì bỏ qua (chạy lại được nếu bị ngắt).

Ví dụ:
  python -m tools.decode_to_npy --protocol protocols/asvspoof2019_la_train.csv \
      --audio_dir /kaggle/input/<ds>/LA/ASVspoof2019_LA_train/flac --out_dir /tmp/la_npy/train
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

from common.audio import SAMPLE_RATE
from common.data import read_protocol


def decode_one(src: str, dst: str) -> str:
    if os.path.exists(dst):
        return "skip"
    x, sr = sf.read(src, dtype="int16", always_2d=True)
    if sr != SAMPLE_RATE or x.shape[1] != 1:
        raise ValueError(f"{src}: cần {SAMPLE_RATE} Hz mono, nhận {sr} Hz, {x.shape[1]} kênh")
    tmp = dst + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(x[:, 0]))
    os.replace(tmp, dst)
    return "ok"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Giải mã FLAC -> .npy int16")
    ap.add_argument("--protocol", required=True, nargs="+")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--ext", default=".flac")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    fids = [f for p in args.protocol for f in read_protocol(p)["file_id"]]
    srcs = [os.path.join(args.audio_dir, f + args.ext) for f in fids]
    dsts = [os.path.join(args.out_dir, f + ".npy") for f in fids]
    if args.workers <= 1:
        res = [decode_one(s, d) for s, d in zip(srcs, dsts)]
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(decode_one, srcs, dsts, chunksize=64))
    print(f"{len(res)} file: {res.count('ok')} giải mã, {res.count('skip')} bỏ qua -> {args.out_dir}")


if __name__ == "__main__":
    main()
