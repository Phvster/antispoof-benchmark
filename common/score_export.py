"""Xuất file điểm theo quy ước — DÙNG CHUNG (A và B).

Tên file: <out_dir>/<mode>/<model>_<train>_<test>_<aug>_seed<k>.txt, mỗi dòng `file_id score`,
điểm = logit[:,1] - logit[:,0] = logit_thật - logit_giả (cao = thật), thứ tự dòng = thứ tự protocol.
Thư mục con <mode> = window | full (đã chốt trong tài liệu).

Ví dụ:
  python -m common.score_export --checkpoint runs/aasist_l_seed0/best.pt \
      --protocol protocols/asvspoof2019_la_eval.csv --audio_dir <...>/eval --ext .flac \
      --mode window --train_name la19 --test_name la19eval --aug A0 --seed 0
  # model pretrained gốc (model cần có load_pretrained):
  python -m common.score_export --config models/aasist_l/config.yaml --pretrained AASIST-L.pth ...
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import AntiSpoofDataset, read_protocol
from common.train import predict
from common.utils import build_model, count_params, load_yaml


def score_path(out_dir: str, mode: str, model: str, train: str, test: str, aug: str, seed) -> str:
    return os.path.join(out_dir, mode, f"{model}_{train}_{test}_{aug}_seed{seed}.txt")


def write_scores(path: str, fids, scores) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for fid, s in zip(fids, scores):
            f.write(f"{fid} {s:.6f}\n")
    os.replace(tmp, path)


def check_scores(path: str, protocol_ids: list[str]) -> None:
    """Số dòng = protocol, cùng tập file_id, cùng thứ tự, không NaN."""
    with open(path, encoding="utf-8") as f:
        rows = [ln.split() for ln in f if ln.strip()]
    if any(len(r) != 2 for r in rows):
        raise ValueError(f"{path}: mỗi dòng phải là `file_id score`")
    ids = [r[0] for r in rows]
    vals = np.array([float(r[1]) for r in rows])
    if ids != list(protocol_ids):
        raise ValueError(f"{path}: file_id không khớp protocol ({len(ids)} vs {len(protocol_ids)})")
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"{path}: có NaN/inf")


def main(argv=None) -> str:
    ap = argparse.ArgumentParser(description="Xuất file điểm `file_id score`")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="best.pt do common.train sinh ra")
    src.add_argument("--pretrained", help="checkpoint gốc, nạp qua model.load_pretrained()")
    ap.add_argument("--config", help="mặc định lấy config trong --checkpoint")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--ext", default=".flac")
    ap.add_argument("--mode", choices=["window", "full"], required=True)
    ap.add_argument("--model_name", default=None, help="mặc định = model.name trong config")
    ap.add_argument("--train_name", required=True)
    ap.add_argument("--test_name", required=True)
    ap.add_argument("--aug", required=True, choices=["A0", "A1"])
    ap.add_argument("--seed", required=True, choices=["0", "1", "2", "pretrained"],
                    help="0/1/2, hoặc 'pretrained' cho checkpoint gốc (tên ..._seedpretrained.txt)")
    ap.add_argument("--out_dir", default="scores")
    ap.add_argument("--batch_size", type=int, default=32, help="chế độ full luôn dùng 1")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--logits_out", default=None, help="tùy chọn: lưu logits [N,2] (.npy) để chẩn đoán")
    ap.add_argument("--limit_batches", type=int, default=None, help="chỉ để smoke test")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = None
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_yaml(args.config) if args.config else ck["config"]
    model = build_model(cfg["model"])
    if ck is not None:
        model.load_state_dict(ck["model"])
    else:
        if not callable(getattr(model, "load_pretrained", None)):
            raise TypeError("model không có load_pretrained()")
        model.load_pretrained(args.pretrained)
    model.to(device)

    df = read_protocol(args.protocol)
    ds = AntiSpoofDataset(df, args.audio_dir, args.ext, mode=args.mode)
    bs = 1 if args.mode == "full" else args.batch_size
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    fids, labels, scores, logits = predict(model, loader, device, args.limit_batches)

    name = args.model_name or cfg["model"]["name"]
    out = score_path(args.out_dir, args.mode, name, args.train_name, args.test_name, args.aug, args.seed)
    write_scores(out, fids, scores)
    expected = df["file_id"].tolist()
    if args.limit_batches is not None:
        expected = expected[:len(fids)]
    check_scores(out, expected)
    if args.logits_out:
        np.save(args.logits_out, logits)
    print(f"{len(fids)} điểm -> {out}  (params={count_params(model)})")
    return out


if __name__ == "__main__":
    main()
