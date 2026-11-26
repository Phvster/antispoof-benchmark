"""Chuyển protocol CM của ASVspoof 2019 LA (BTC) sang định dạng chung — DÙNG CHUNG.

Input (thư mục ASVspoof2019_LA_cm_protocols), mỗi dòng: `speaker file_id - attack key`
  vd: `LA_0079 LA_T_1138215 - - bonafide`, `LA_0039 LA_E_2834763 - A11 spoof`
Output: <out_dir>/asvspoof2019_la_{train,dev,eval}.csv với cột
  file_id,speaker_id,label,attack_id,subset   (label: bonafide -> 0, spoof -> 1)

Ví dụ:
  python -m protocols.convert_asvspoof2019_la \
      --src /kaggle/input/<asvspoof2019-la>/LA/ASVspoof2019_LA_cm_protocols --out_dir protocols
"""
from __future__ import annotations

import argparse
import csv
import os

from common.data import COLUMNS, md5sum, read_protocol

SPLITS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", 25380),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", 24844),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", 71237),
}
LABEL_MAP = {"bonafide": 0, "spoof": 1}  # quy ước chung: 0 = thật, 1 = giả


def convert_file(src: str, dst: str, subset: str) -> list[dict]:
    rows = []
    with open(src, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{src}:{n}: cần 5 cột, có {len(parts)}: {line!r}")
            spk, fid, _, attack, key = parts
            if key not in LABEL_MAP:
                raise ValueError(f"{src}:{n}: key lạ {key!r}")
            label = LABEL_MAP[key]
            if label == 0 and attack != "-":
                raise ValueError(f"{src}:{n}: bonafide nhưng attack={attack}")
            rows.append({"file_id": fid, "speaker_id": spk, "label": label,
                         "attack_id": attack, "subset": subset})
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Chuyển protocol ASVspoof 2019 LA sang định dạng chung")
    ap.add_argument("--src", required=True, help="thư mục ASVspoof2019_LA_cm_protocols")
    ap.add_argument("--out_dir", default="protocols")
    ap.add_argument("--no_count_check", action="store_true",
                    help="bỏ kiểm tra số dòng chính thức (chỉ dùng cho dữ liệu giả)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    ids, spks, md5 = {}, {}, {}
    for subset, (fname, expected) in SPLITS.items():
        src = os.path.join(args.src, fname)
        dst = os.path.join(args.out_dir, f"asvspoof2019_la_{subset}.csv")
        rows = convert_file(src, dst, subset)
        if not args.no_count_check and len(rows) != expected:
            raise ValueError(f"{subset}: {len(rows)} dòng, kỳ vọng {expected}")
        df = read_protocol(dst)  # kiểm tra lại file vừa ghi
        ids[subset] = set(df["file_id"])
        spks[subset] = set(df["speaker_id"])
        md5[subset] = md5sum(dst)
        n_bona = int((df["label"] == 0).sum())
        print(f"{subset:5s} rows={len(df):6d} bona={n_bona:6d} spoof={len(df) - n_bona:6d} "
              f"speakers={df['speaker_id'].nunique():4d} "
              f"attacks={sorted(set(df['attack_id']) - {'-'})}")
        print(f"      md5 src {md5sum(src)}  {src}")
        print(f"      md5 out {md5[subset]}  {dst}")
    for a, b in (("train", "dev"), ("train", "eval"), ("dev", "eval")):
        inter = ids[a] & ids[b]
        if inter:
            raise ValueError(f"file_id trùng giữa {a} và {b}: {sorted(inter)[:5]}")
        inter = spks[a] & spks[b]
        if inter:
            raise ValueError(f"rò người nói giữa {a} và {b}: {sorted(inter)[:5]}")
    print("OK: không có file_id hay speaker_id trùng giữa các tập.")
    return md5


if __name__ == "__main__":
    main()
