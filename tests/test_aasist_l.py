"""Test AASIST-L: số tham số, chiều nhãn (khớp output gốc clovaai), sinc buffer, nạp pretrained,
smoke train + file điểm trên CPU."""
import copy
import os

import numpy as np
import pytest
import torch

import common.train as train_mod
from common.audio import NB_SAMP
from common.score_export import main as export_main
from common.utils import build_model, count_params, load_yaml
from models.aasist_l.model import AASISTL
from tests.reference import aasist_clovaai as ref

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "aasist_l", "config.yaml")
CFG = load_yaml(CFG_PATH)


def ref_model_with_same_weights(ours: AASISTL) -> torch.nn.Module:
    """Model gốc clovaai, cùng trọng số với `ours` (state_dict gốc = state_dict của ours.model)."""
    torch.manual_seed(123)
    m = ref.Model(copy.deepcopy(ours.model.d_args))
    m.load_state_dict(ours.model.state_dict(), strict=True)
    return m


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(CFG["model"]).eval()


def test_param_count(model):
    assert count_params(model) == AASISTL.EXPECTED_PARAMS == 85306


def test_hyperparams_match_original_conf():
    """Giá trị copy từ config/AASIST-L.conf gốc (commit a04c986)."""
    a = CFG["model"]["args"]
    assert a["first_conv"] == 128
    assert a["filts"] == [70, [1, 32], [32, 32], [32, 24], [24, 24]]
    assert a["gat_dims"] == [24, 32]
    assert a["pool_ratios"] == [0.4, 0.5, 0.7, 0.5]
    assert a["temperatures"] == [2.0, 2.0, 100.0, 100.0]
    assert CFG["train"]["batch_size"] == 24
    o = CFG["optim"]
    assert (o["base_lr"], o["lr_min"], o["weight_decay"], o["scheduler"]) == (1e-4, 5e-6, 1e-4, "cosine")


def test_registry_name_only():
    m = build_model({"name": "aasist_l", "args": CFG["model"]["args"]})
    assert isinstance(m, AASISTL)


def test_forward_is_logit_difference(model):
    x = torch.randn(3, NB_SAMP) * 0.1
    with torch.no_grad():
        lg = model.logits(x)
        s = model(x)
    assert lg.shape == (3, 2)
    torch.testing.assert_close(s, lg[:, 1] - lg[:, 0], atol=1e-6, rtol=0)


def test_label_direction_logits_equal_clovaai_output(model):
    """Không đảo cột: logits(x) == output gốc clovaai (cột 0 = giả, cột 1 = thật như clovaai)."""
    m_ref = ref_model_with_same_weights(model).eval()
    x = torch.randn(2, NB_SAMP) * 0.1
    with torch.no_grad():
        _, out_ref = m_ref(x)
        out = model.logits(x)
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=0)


def test_sinc_buffer_matches_original(model):
    """band_pass: giá trị trùng bản gốc, là buffer không persistent (không vào state_dict)."""
    m_ref = ref_model_with_same_weights(model)
    ours = model.model.conv_time
    torch.testing.assert_close(ours.band_pass, m_ref.conv_time.band_pass, atol=1e-6, rtol=0)
    assert "band_pass" in dict(ours.named_buffers())
    assert not any("band_pass" in k for k in model.state_dict())
    x = torch.randn(2, 1, NB_SAMP) * 0.1
    torch.testing.assert_close(ours(x), m_ref.conv_time(x), atol=1e-6, rtol=0)
    # nhánh mask (Freq_aug) không được sửa buffer tại chỗ
    before = ours.band_pass.clone()
    ours(x, mask=True)
    torch.testing.assert_close(ours.band_pass, before, atol=0, rtol=0)


def test_train_mode_matches_original(model):
    """Chế độ train (dropout), cùng seed -> output trùng bản gốc tới 1e-6."""
    ours = copy.deepcopy(model).train()
    m_ref = ref_model_with_same_weights(ours).train()
    x = torch.randn(2, NB_SAMP) * 0.1
    torch.manual_seed(7)
    out = ours.logits(x)
    torch.manual_seed(7)
    _, out_ref = m_ref(x, Freq_aug=False)
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=0)


@pytest.mark.parametrize("n", [NB_SAMP // 3, NB_SAMP * 2 + 123])
def test_full_mode_variable_length(model, n):
    with torch.no_grad():
        s = model(torch.randn(1, n) * 0.1)
    assert s.shape == (1,) and torch.isfinite(s).all()


def test_load_pretrained_strict(model, tmp_path):
    """Checkpoint dạng gốc (key không có tiền tố, hoặc 'module.') nạp strict; thiếu key -> lỗi."""
    sd = ref_model_with_same_weights(model).state_dict()
    p = tmp_path / "orig.pth"
    torch.save(sd, p)
    m = AASISTL(**CFG["model"]["args"]).eval()
    m.load_pretrained(str(p))
    x = torch.randn(1, NB_SAMP) * 0.1
    with torch.no_grad():
        torch.testing.assert_close(m.logits(x), model.logits(x), atol=1e-6, rtol=0)
    p2 = tmp_path / "dp.pth"
    torch.save({"module." + k: v for k, v in sd.items()}, p2)
    m.load_pretrained(str(p2))
    p3 = tmp_path / "bad.pth"
    torch.save({k: v for k, v in list(sd.items())[1:]}, p3)
    with pytest.raises(RuntimeError):
        m.load_pretrained(str(p3))


@pytest.mark.skipif(not os.environ.get("AASIST_L_WEIGHTS"),
                    reason="đặt AASIST_L_WEIGHTS=<đường dẫn AASIST-L.pth gốc> để chạy")
def test_load_real_pretrained():
    m = AASISTL(**CFG["model"]["args"])
    m.load_pretrained(os.environ["AASIST_L_WEIGHTS"])
    assert count_params(m) == 85306


def test_smoke_train_and_export_aasist_l(fake_la, tmp_path, monkeypatch):
    """CPU: train AASIST-L 1 epoch (2 batch) -> best.pt -> file điểm chế độ cửa sổ."""
    from protocols.convert_asvspoof2019_la import main as convert_main
    monkeypatch.setattr(train_mod, "MAX_EPOCHS", 1)
    proto = tmp_path / "protocols"
    convert_main(["--src", fake_la["proto_dir"], "--out_dir", str(proto), "--no_count_check"])
    out = tmp_path / "run"
    st = train_mod.main([
        "--config", CFG_PATH, "--seed", "0", "--out_dir", str(out),
        "--train_protocol", str(proto / "asvspoof2019_la_train.csv"),
        "--train_audio_dir", fake_la["audio_dir"]["train"],
        "--dev_protocol", str(proto / "asvspoof2019_la_dev.csv"),
        "--dev_audio_dir", fake_la["audio_dir"]["dev"], "--ext", ".flac",
        "--batch_size", "2", "--num_workers", "0", "--limit_batches", "2", "--device", "cpu"])
    assert st["state"] == "finished" and (out / "best.pt").exists()
    path = export_main(["--checkpoint", str(out / "best.pt"),
                        "--protocol", str(proto / "asvspoof2019_la_eval.csv"),
                        "--audio_dir", fake_la["audio_dir"]["eval"], "--mode", "window",
                        "--train_name", "la19", "--test_name", "la19eval", "--aug", "A0",
                        "--seed", "0", "--out_dir", str(tmp_path / "scores"), "--batch_size", "4",
                        "--device", "cpu"])
    assert path.replace("\\", "/").endswith("scores/window/aasist_l_la19_la19eval_A0_seed0.txt")
    vals = np.array([float(l.split()[1]) for l in open(path, encoding="utf-8")])
    assert len(vals) == fake_la["n"]["eval"] and np.isfinite(vals).all()
