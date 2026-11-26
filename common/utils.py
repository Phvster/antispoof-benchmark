"""Tiện ích chung: seed, nạp config/model, ghi file an toàn, thông tin môi trường."""
from __future__ import annotations

import importlib
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def build_model(model_cfg: dict) -> torch.nn.Module:
    """model_cfg = {"name": ..., "class": "pkg.module:Class", "args": {...}}.

    Model phải có:
      logits(wave[B,T]) -> [B,2], cột 0 = giả, cột 1 = thật (nhãn chung, giống clovaai);
      forward(wave[B,T]) -> score[B] = logits[:,1] - logits[:,0] (cao = thật).
    """
    mod_name, cls_name = model_cfg["class"].split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    model = cls(**(model_cfg.get("args") or {}))
    if not callable(getattr(model, "logits", None)):
        raise TypeError(f"{model_cfg['class']} thiếu logits(wave) -> [B,2] (cột 0 thật, 1 giả)")
    return model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def env_info() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "git_commit": git_commit(),
    }


def write_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
