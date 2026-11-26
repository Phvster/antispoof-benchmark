"""Vòng train chung — DÙNG CHUNG cho mọi model (A và B).

Quy ước cố định (CONVENTIONS.md, mục "Huấn luyện") — là hằng số ở đây, KHÔNG đặt trong config model:
- Weighted CE [0.1, 0.9] theo nhãn (0 = giả, 1 = thật), giống clovaai -> CLASS_WEIGHTS
- Chọn checkpoint theo dev EER (chế độ cửa sổ), tối đa 100 epoch, dừng sớm sau 10 epoch
  dev EER không giảm                                     -> MAX_EPOCHS, PATIENCE
- Seed 0, 1, 2                                          -> SEEDS
Khác clovaai/aasist main.py (cố ý): không SWA, không chấm eval trong lúc train.

Resume: mỗi epoch ghi last.pt (model, optimizer, scheduler, RNG, generator, lịch sử log).
`--resume auto` nạp last.pt nếu có. `--max_minutes` dừng sạch trước khi hết phiên Kaggle;
`--stop_after_epochs N` (chạy thử) tạm dừng sau N epoch của phiên này — không đổi MAX_EPOCHS,
nên lịch cosine và kết quả giống hệt chạy liền.
(trạng thái "paused" trong run_status.json), chạy lại cùng lệnh để đi tiếp.

Ví dụ:
  python -m common.train --config models/aasist_l/config.yaml --seed 0 --out_dir runs/aasist_l_seed0 \
      --train_protocol protocols/asvspoof2019_la_train.csv --train_audio_dir <...>/train \
      --dev_protocol protocols/asvspoof2019_la_dev.csv --dev_audio_dir <...>/dev --ext .flac
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.audio import NB_SAMP
from common.data import AntiSpoofDataset, worker_init_fn
from common.metrics import eer_from_labels
from common.utils import (atomic_torch_save, build_model, count_params, env_info, load_yaml,
                          save_yaml, set_seed, write_json)

CLASS_WEIGHTS = [0.1, 0.9]  # [giả (0), thật (1)] — đã chốt, giống clovaai/aasist main.py
MAX_EPOCHS = 100
PATIENCE = 10
SEEDS = (0, 1, 2)

LOG_FIELDS = ["epoch", "train_loss", "dev_loss", "dev_eer", "lr", "epoch_sec",
              "best_dev_eer", "is_best", "bad_epochs"]
DATA_KEYS = ["train_protocol", "train_audio_dir", "dev_protocol", "dev_audio_dir", "ext"]


# ----------------------------------------------------------------------------- optim
def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Port từ clovaai/aasist utils.py."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def create_optimizer(params, ocfg: dict, total_steps: int):
    """Adam + cosine theo step như clovaai/aasist utils.py (_get_optimizer/_get_scheduler)."""
    name = ocfg.get("optimizer", "adam")
    if name == "adam":
        opt = torch.optim.Adam(params, lr=ocfg["base_lr"], betas=tuple(ocfg.get("betas", (0.9, 0.999))),
                               weight_decay=ocfg.get("weight_decay", 0.0),
                               amsgrad=bool(ocfg.get("amsgrad", False)))
    elif name == "sgd":
        opt = torch.optim.SGD(params, lr=ocfg["base_lr"], momentum=ocfg.get("momentum", 0.9),
                              weight_decay=ocfg.get("weight_decay", 0.0),
                              nesterov=bool(ocfg.get("nesterov", False)))
    else:
        raise ValueError(f"optimizer không hỗ trợ: {name}")
    sname = ocfg.get("scheduler")
    if sname == "cosine":
        ratio = ocfg["lr_min"] / ocfg["base_lr"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda step: cosine_annealing(step, total_steps, 1, ratio))
    elif sname in (None, "none"):
        sched = None
    else:
        raise ValueError(f"scheduler không hỗ trợ: {sname}")
    return opt, sched


# ----------------------------------------------------------------------------- loops
def make_criterion(device) -> nn.Module:
    return nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device))


def train_one_epoch(model, loader, opt, sched, crit, device, limit_batches=None) -> float:
    model.train()
    total, n = 0.0, 0
    for i, (x, y, _) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.view(-1).long().to(device, non_blocking=True)
        loss = crit(model.logits(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, limit_batches=None):
    """Trả (file_ids, labels, scores, logits). score = logit[:,1] - logit[:,0]
    = logit_thật - logit_giả (cao = thật)."""
    model.eval()
    fids, labels, logits = [], [], []
    for i, (x, y, fid) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        lg = model.logits(x.to(device, non_blocking=True)).float().cpu()
        logits.append(lg)
        labels.extend(int(v) for v in y)
        fids.extend(fid)
    logits = torch.cat(logits) if logits else torch.zeros(0, 2)
    scores = (logits[:, 1] - logits[:, 0]).numpy()
    return fids, np.asarray(labels), scores, logits.numpy()


# ----------------------------------------------------------------------------- run
def rng_state(gen: torch.Generator) -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "loader_gen": gen.get_state(),
    }


def set_rng_state(st: dict, gen: torch.Generator) -> None:
    # Các hàm set_*_state chỉ nhận ByteTensor trên CPU.
    torch.set_rng_state(st["torch"].cpu())
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    gen.set_state(st["loader_gen"].cpu())


def write_log(history: list[dict], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(history)
    os.replace(tmp, path)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train chung (weighted CE, chọn theo dev EER, resume)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", choices=["auto", "none"], default="auto")
    ap.add_argument("--stop_after_epochs", type=int, default=None,
                    help="chạy thử: tạm dừng (paused) sau N epoch của phiên này; resume đi tiếp bình thường")
    ap.add_argument("--max_minutes", type=float, default=None,
                    help="ngân sách thời gian của phiên này; không bắt đầu epoch mới nếu không kịp")
    for k in DATA_KEYS:
        ap.add_argument(f"--{k}", default=None, help="ghi đè data.%s trong config" % k)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="chỉ để smoke test; ghi vào config_used")
    ap.add_argument("--limit_batches", type=int, default=None, help="chỉ để smoke test")
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    session_start = time.time()
    cfg = load_yaml(args.config)
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    for k in DATA_KEYS:
        if getattr(args, k) is not None:
            cfg["data"][k] = getattr(args, k)
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.limit_batches is not None:
        cfg["train"]["limit_batches"] = args.limit_batches
    missing = [k for k in DATA_KEYS if not cfg["data"].get(k)]
    if missing:
        raise ValueError(f"thiếu đường dẫn dữ liệu: {missing} (đặt trong config data: hoặc CLI)")
    if args.seed not in SEEDS:
        warnings.warn(f"seed {args.seed} không thuộc quy ước {SEEDS}")

    dcfg, tcfg, ocfg = cfg["data"], cfg["train"], cfg["optim"]
    bs = int(tcfg["batch_size"])
    nw = int(tcfg.get("num_workers", 0))
    limit = tcfg.get("limit_batches")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    p = lambda name: os.path.join(args.out_dir, name)  # noqa: E731

    set_seed(args.seed)
    model = build_model(cfg["model"]).to(device)
    n_params = count_params(model)

    train_ds = AntiSpoofDataset(dcfg["train_protocol"], dcfg["train_audio_dir"], dcfg["ext"],
                                mode="train", seed=args.seed)
    dev_ds = AntiSpoofDataset(dcfg["dev_protocol"], dcfg["dev_audio_dir"], dcfg["ext"], mode="window")
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True,
                              num_workers=nw, pin_memory=pin, worker_init_fn=worker_init_fn,
                              generator=gen, persistent_workers=False)
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False, drop_last=False,
                            num_workers=nw, pin_memory=pin, persistent_workers=False)

    steps_per_epoch = len(train_loader) if limit is None else min(limit, len(train_loader))
    opt, sched = create_optimizer(model.parameters(), ocfg, MAX_EPOCHS * steps_per_epoch)
    crit = make_criterion(device)

    # ---- trạng thái mặc định / resume
    start_epoch, best_eer, best_epoch, bad_epochs = 0, math.inf, -1, 0
    history, train_time = [], 0.0
    if args.resume == "auto" and os.path.exists(p("last.pt")):
        # map_location="cpu": RNG state phải ở CPU; load_state_dict tự đưa tensor về device
        ck = torch.load(p("last.pt"), map_location="cpu", weights_only=False)
        if ck["seed"] != args.seed:
            raise ValueError(f"last.pt có seed {ck['seed']} != --seed {args.seed}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck["scheduler"] is not None:
            sched.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"], gen)
        start_epoch = ck["epoch"] + 1
        best_eer, best_epoch, bad_epochs = ck["best_eer"], ck["best_epoch"], ck["bad_epochs"]
        history, train_time = ck["history"], ck["train_time"]
        if ck.get("config") != cfg:
            warnings.warn("config hiện tại khác config trong last.pt (xem config_used.yaml)")
        print(f"[resume] từ epoch {ck['epoch']} (best dev EER {best_eer:.4f}% @ {best_epoch})")

    save_yaml(cfg, p("config_used.yaml"))
    write_json({**env_info(), "device": str(device), "seed": args.seed, "batch_size": bs,
                "lr": ocfg["base_lr"], "optim": ocfg, "num_workers": nw, "n_params": n_params,
                "NB_SAMP": NB_SAMP, "CLASS_WEIGHTS": CLASS_WEIGHTS, "MAX_EPOCHS": MAX_EPOCHS,
                "PATIENCE": PATIENCE, "steps_per_epoch": steps_per_epoch,
                "n_train": len(train_ds), "n_dev": len(dev_ds)}, p("env.json"))
    print(f"model={cfg['model'].get('name')} params={n_params} device={device} "
          f"train={len(train_ds)} dev={len(dev_ds)} bs={bs} steps/epoch={steps_per_epoch}")

    def status(state: str, epoch: int) -> dict:
        st = {"state": state, "last_epoch": epoch, "best_epoch": best_epoch,
              "best_dev_eer": best_eer, "bad_epochs": bad_epochs, "train_time_sec": train_time}
        write_json(st, p("run_status.json"))
        with open(p("train_time.txt"), "w") as f:
            f.write(f"{train_time:.1f}\n")
        return st

    if bad_epochs >= PATIENCE or start_epoch >= MAX_EPOCHS:
        print("Run đã kết thúc trước đó, không train thêm.")
        return status("early_stopped" if bad_epochs >= PATIENCE else "finished", start_epoch - 1)

    epoch_secs = []
    epoch = start_epoch - 1
    for epoch in range(start_epoch, MAX_EPOCHS):
        if args.max_minutes is not None and epoch_secs:
            elapsed = time.time() - session_start
            if elapsed + 1.2 * float(np.mean(epoch_secs)) > args.max_minutes * 60:
                print(f"[pause] hết ngân sách {args.max_minutes} phút trước epoch {epoch}; "
                      f"chạy lại cùng lệnh để resume.")
                return status("paused", epoch - 1)
        t0 = time.time()
        train_ds.set_epoch(epoch)
        lr = opt.param_groups[0]["lr"]
        tr_loss = train_one_epoch(model, train_loader, opt, sched, crit, device, limit)
        fids, labels, scores, logits = predict(model, dev_loader, device, limit)
        dev_loss = float(crit(torch.from_numpy(logits).to(device),
                              torch.from_numpy(labels).long().to(device)).item())
        dev_eer, _ = eer_from_labels(scores, labels)
        is_best = dev_eer < best_eer
        if is_best:
            best_eer, best_epoch, bad_epochs = dev_eer, epoch, 0
            atomic_torch_save({"model": model.state_dict(), "epoch": epoch, "dev_eer": dev_eer,
                               "seed": args.seed, "config": cfg}, p("best.pt"))
        else:
            bad_epochs += 1
        dt = time.time() - t0
        epoch_secs.append(dt)
        train_time += dt
        history.append({"epoch": epoch, "train_loss": f"{tr_loss:.6f}", "dev_loss": f"{dev_loss:.6f}",
                        "dev_eer": f"{dev_eer:.4f}", "lr": f"{lr:.3e}", "epoch_sec": f"{dt:.1f}",
                        "best_dev_eer": f"{best_eer:.4f}", "is_best": int(is_best),
                        "bad_epochs": bad_epochs})
        atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                           "scheduler": sched.state_dict() if sched is not None else None,
                           "rng": rng_state(gen), "epoch": epoch, "best_eer": best_eer,
                           "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                           "history": copy.deepcopy(history), "train_time": train_time,
                           "seed": args.seed, "config": cfg}, p("last.pt"))
        write_log(history, p("log.csv"))
        print(f"epoch {epoch:3d} loss {tr_loss:.4f} dev_loss {dev_loss:.4f} dev_EER {dev_eer:.3f}% "
              f"best {best_eer:.3f}%@{best_epoch} bad {bad_epochs} ({dt:.0f}s)")
        if bad_epochs >= PATIENCE:
            print(f"[early stop] {PATIENCE} epoch dev EER không giảm.")
            return status("early_stopped", epoch)
        if args.stop_after_epochs is not None and epoch - start_epoch + 1 >= args.stop_after_epochs                 and epoch + 1 < MAX_EPOCHS:
            print(f"[pause] đã chạy {args.stop_after_epochs} epoch trong phiên này (--stop_after_epochs).")
            return status("paused", epoch)
    return status("finished", epoch)


if __name__ == "__main__":
    main()
