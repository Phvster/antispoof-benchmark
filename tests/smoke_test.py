"""Smoke test khung chung trên CPU: protocol giả -> train (1 epoch, vài batch) -> resume
-> file điểm cả hai chế độ. Dùng model giả TinyModel; smoke AASIST-L thêm ở PR2.
Resume dùng num_workers=0 để so khớp tuyệt đối.
"""
import csv
import json
import os

import numpy as np
import pytest
import yaml

import common.train as train_mod
from common.data import read_protocol
from common.metrics import eer_from_labels
from common.score_export import main as export_main
from protocols.convert_asvspoof2019_la import main as convert_main

TINY_CFG = {
    "model": {"name": "tiny", "class": "tests.dummy_model:TinyModel", "args": {"channels": 4}},
    "train": {"batch_size": 4, "num_workers": 0},
    "optim": {"optimizer": "adam", "base_lr": 1.0e-3, "lr_min": 1.0e-5, "betas": [0.9, 0.999],
              "weight_decay": 1.0e-4, "amsgrad": False, "scheduler": "cosine"},
}


@pytest.fixture(scope="module")
def setup(fake_la, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    proto = tmp / "protocols"
    convert_main(["--src", fake_la["proto_dir"], "--out_dir", str(proto), "--no_count_check"])
    cfg_path = tmp / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY_CFG))
    data_args = ["--train_protocol", str(proto / "asvspoof2019_la_train.csv"),
                 "--train_audio_dir", fake_la["audio_dir"]["train"],
                 "--dev_protocol", str(proto / "asvspoof2019_la_dev.csv"),
                 "--dev_audio_dir", fake_la["audio_dir"]["dev"], "--ext", ".flac"]
    return {"tmp": tmp, "proto": proto, "cfg": str(cfg_path), "data": data_args, "la": fake_la}


def read_log(path, drop=("epoch_sec",)):
    with open(path, newline="") as f:
        return [{k: v for k, v in r.items() if k not in drop} for r in csv.DictReader(f)]


def run(setup, out, *extra):
    return train_mod.main(["--config", setup["cfg"], "--seed", "0", "--out_dir", str(out),
                           "--limit_batches", "3", "--num_workers", "0", *setup["data"], *extra])


def test_train_resume_matches_continuous(setup, monkeypatch):
    monkeypatch.setattr(train_mod, "MAX_EPOCHS", 2)
    a = setup["tmp"] / "run_continuous"
    b = setup["tmp"] / "run_resumed"

    st = run(setup, a)
    assert st["state"] == "finished"

    st1 = run(setup, b, "--max_minutes", "1e-6")  # 1 epoch rồi pause
    assert st1["state"] == "paused" and st1["last_epoch"] == 0
    assert len(read_log(b / "log.csv")) == 1
    st2 = run(setup, b)  # resume
    assert st2["state"] == "finished" and st2["last_epoch"] == 1

    la, lb = read_log(a / "log.csv"), read_log(b / "log.csv")
    assert len(la) == len(lb) == 2
    assert la == lb, f"resume lệch so với chạy liền:\n{la}\n{lb}"
    for name in ("best.pt", "last.pt", "config_used.yaml", "env.json", "train_time.txt",
                 "run_status.json"):
        assert (b / name).exists(), name
    env = json.loads((b / "env.json").read_text(encoding="utf-8"))
    assert env["CLASS_WEIGHTS"] == [0.1, 0.9] and env["batch_size"] == 4 and "torch" in env


def test_early_stopping(setup, monkeypatch):
    monkeypatch.setattr(train_mod, "MAX_EPOCHS", 5)
    monkeypatch.setattr(train_mod, "PATIENCE", 2)
    monkeypatch.setattr(train_mod, "eer_from_labels", lambda s, l: (50.0, 0.0))  # dev EER đứng yên
    out = setup["tmp"] / "run_early"
    st = run(setup, out)
    assert st["state"] == "early_stopped"
    log = read_log(out / "log.csv")
    assert [int(r["epoch"]) for r in log] == [0, 1, 2]
    assert [int(r["is_best"]) for r in log] == [1, 0, 0]
    assert st["best_epoch"] == 0
    # chạy lại sau khi đã dừng sớm: không train thêm
    assert run(setup, out)["state"] == "early_stopped"
    assert len(read_log(out / "log.csv")) == 3


def test_resume_rejects_other_seed(setup, monkeypatch):
    monkeypatch.setattr(train_mod, "MAX_EPOCHS", 1)
    out = setup["tmp"] / "run_seed"
    run(setup, out)
    with pytest.raises(ValueError):
        train_mod.main(["--config", setup["cfg"], "--seed", "1", "--out_dir", str(out),
                        "--limit_batches", "3", *setup["data"]])


@pytest.mark.parametrize("mode", ["window", "full"])
def test_score_export(setup, monkeypatch, mode):
    monkeypatch.setattr(train_mod, "MAX_EPOCHS", 1)
    run_dir = setup["tmp"] / "run_export"
    if not (run_dir / "best.pt").exists():
        run(setup, run_dir)
    eval_proto = str(setup["proto"] / "asvspoof2019_la_eval.csv")
    out_dir = setup["tmp"] / "scores"
    logits = setup["tmp"] / f"logits_{mode}.npy"
    path = export_main(["--checkpoint", str(run_dir / "best.pt"), "--protocol", eval_proto,
                        "--audio_dir", setup["la"]["audio_dir"]["eval"], "--ext", ".flac",
                        "--mode", mode, "--train_name", "la19", "--test_name", "la19eval",
                        "--aug", "A0", "--seed", "0", "--out_dir", str(out_dir),
                        "--batch_size", "4", "--logits_out", str(logits)])
    assert os.path.normpath(path) == os.path.normpath(
        str(out_dir / mode / "tiny_la19_la19eval_A0_seed0.txt"))
    rows = [ln.split() for ln in open(path, encoding="utf-8")]
    df = read_protocol(eval_proto)
    assert [r[0] for r in rows] == df["file_id"].tolist()
    scores = np.array([float(r[1]) for r in rows])
    assert np.isfinite(scores).all()
    lg = np.load(logits)
    np.testing.assert_allclose(scores, lg[:, 1] - lg[:, 0], atol=1e-5)
    eer, _ = eer_from_labels(scores, df["label"].values)
    assert 0.0 <= eer <= 100.0


def test_resume_checkpoint_rng_restorable_from_cpu(setup, monkeypatch):
    """last.pt phải nạp bằng map_location='cpu' và set_rng_state nhận được (lỗi cũ: map_location=cuda)."""
    import torch
    monkeypatch.setattr(train_mod, "MAX_EPOCHS", 1)
    out = setup["tmp"] / "run_rng"
    run(setup, out)
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    g = torch.Generator()
    train_mod.set_rng_state(ck["rng"], g)
    assert torch.equal(g.get_state(), ck["rng"]["loader_gen"])
    src = open(train_mod.__file__, encoding="utf-8").read()
    assert 'torch.load(p("last.pt"), map_location="cpu"' in src


def test_high_score_means_bonafide_end_to_end(setup):
    """Toàn chuỗi protocol -> dataset -> score_export -> EER: file thật (nhãn 1) phải có điểm cao."""
    cfg = setup["tmp"] / "oracle.yaml"
    cfg.write_text(yaml.safe_dump({"model": {"name": "oracle", "class": "tests.dummy_model:OracleModel"}}))
    eval_proto = str(setup["proto"] / "asvspoof2019_la_eval.csv")
    path = export_main(["--config", str(cfg), "--pretrained", "none", "--protocol", eval_proto,
                        "--audio_dir", setup["la"]["audio_dir"]["eval"], "--mode", "window",
                        "--train_name", "x", "--test_name", "la19eval", "--aug", "A0",
                        "--seed", "pretrained", "--out_dir", str(setup["tmp"] / "scores_oracle")])
    scores = np.array([float(ln.split()[1]) for ln in open(path, encoding="utf-8")])
    labels = read_protocol(eval_proto)["label"].values
    assert scores[labels == 1].min() > scores[labels == 0].max()
    eer, _ = eer_from_labels(scores, labels)
    assert eer == 0.0
