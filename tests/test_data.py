import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from common.audio import NB_SAMP, pad_random
from common.data import AntiSpoofDataset, read_protocol, sample_rng, worker_init_fn
from protocols.convert_asvspoof2019_la import main as convert_main
from tests.conftest import write_flac


def test_convert_protocol(fake_la, tmp_path):
    md5 = convert_main(["--src", fake_la["proto_dir"], "--out_dir", str(tmp_path), "--no_count_check"])
    assert set(md5) == {"train", "dev", "eval"}
    df = read_protocol(str(tmp_path / "asvspoof2019_la_train.csv"))
    assert len(df) == fake_la["n"]["train"]
    assert set(df["subset"]) == {"train"}
    # i chẵn = bonafide -> 1 (thật), i lẻ = spoof -> 0 (giả); bonafide có attack "-"
    assert (df["label"].values == 1 - np.arange(len(df)) % 2).all()
    assert (df.loc[df["label"] == 1, "attack_id"] == "-").all()
    assert (df.loc[df["label"] == 0, "attack_id"] != "-").all()


def test_convert_count_check_fails_on_fake(fake_la, tmp_path):
    with pytest.raises(ValueError):
        convert_main(["--src", fake_la["proto_dir"], "--out_dir", str(tmp_path)])


def test_read_protocol_validation(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("file_id,speaker_id,label,attack_id,subset\na,s,2,-,train\n")
    with pytest.raises(ValueError):
        read_protocol(str(p))
    p.write_text("file_id,speaker_id,label,attack_id,subset\na,s,0,-,train\na,s,1,A01,train\n")
    with pytest.raises(ValueError):
        read_protocol(str(p))


@pytest.fixture
def same_long_audio(tmp_path):
    """8 file cùng nội dung (ramp dài 3*NB_SAMP) -> crop chỉ khác nhau do rng."""
    x = (np.arange(3 * NB_SAMP) % 30000).astype(np.int16)
    fids = [f"R{i}" for i in range(8)]
    for f in fids:
        write_flac(str(tmp_path / f"{f}.flac"), x)
    df = pd.DataFrame({"file_id": fids, "speaker_id": "S", "label": 1, "attack_id": "-",
                       "subset": "train"})
    return df, str(tmp_path)


def crops(ds, num_workers, epoch):
    ds.set_epoch(epoch)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                    worker_init_fn=worker_init_fn)
    return [x[0].numpy().copy() for x, _, _ in dl]


def start_of(c):
    return round(float(c[0]) * 32768)  # ramp -> giá trị đầu = vị trí bắt đầu crop


def test_crop_rng_per_sample_independent_of_workers(same_long_audio):
    """Quy ước: rng = default_rng([seed, epoch, idx]) -> crop không phụ thuộc num_workers."""
    df, adir = same_long_audio
    ds = AntiSpoofDataset(df, adir, ".flac", mode="train", seed=0)
    c0 = crops(ds, 0, epoch=0)
    c2 = crops(ds, 2, epoch=0)
    for a, b in zip(c0, c2):
        np.testing.assert_array_equal(a, b)
    # các mẫu khác idx (kể cả do 2 worker khác nhau xử lý) có crop khác nhau
    assert not np.array_equal(c2[0], c2[1])
    assert len({start_of(v) for v in c2}) > 1
    # đúng công thức quy ước
    x = (np.arange(3 * NB_SAMP) % 30000).astype(np.float32) / 32768.0
    for idx in (0, 3):
        np.testing.assert_array_equal(c0[idx], pad_random(x, NB_SAMP, sample_rng(0, 0, idx)))
    # epoch khác -> crop khác
    c3 = crops(ds, 2, epoch=1)
    assert any(not np.array_equal(a, b) for a, b in zip(c0, c3))


def test_seed_changes_crops(same_long_audio):
    df, adir = same_long_audio
    a = crops(AntiSpoofDataset(df, adir, ".flac", mode="train", seed=0), 0, 0)
    b = crops(AntiSpoofDataset(df, adir, ".flac", mode="train", seed=1), 0, 0)
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


def test_window_and_full_modes(fake_la, tmp_path):
    convert_main(["--src", fake_la["proto_dir"], "--out_dir", str(tmp_path), "--no_count_check"])
    proto = str(tmp_path / "asvspoof2019_la_dev.csv")
    w = AntiSpoofDataset(proto, fake_la["audio_dir"]["dev"], mode="window")
    f = AntiSpoofDataset(proto, fake_la["audio_dir"]["dev"], mode="full")
    for i in range(len(w)):
        xw, yw, fw = w[i]
        xf, yf, ff = f[i]
        assert xw.shape == (NB_SAMP,) and xw.dtype == torch.float32
        assert fw == ff and yw == yf
        n = min(NB_SAMP, xf.shape[0])
        assert torch.equal(xw[:n], xf[:n])


def test_convert_rejects_speaker_leak(fake_la, tmp_path):
    import shutil
    src = tmp_path / "proto"
    shutil.copytree(fake_la["proto_dir"], src)
    dev = src / "ASVspoof2019.LA.cm.dev.trl.txt"
    train_spk = (src / "ASVspoof2019.LA.cm.train.trn.txt").read_text().split()[0]
    lines = dev.read_text().splitlines(keepends=True)
    lines[0] = train_spk + lines[0][lines[0].index(" "):]
    dev.write_text("".join(lines))
    with pytest.raises(ValueError, match="rò người nói"):
        convert_main(["--src", str(src), "--out_dir", str(tmp_path / "out"), "--no_count_check"])


def test_convert_catches_flipped_bona_count(fake_la, tmp_path, monkeypatch):
    """Kiểm tra số nhãn 1 (thật) chính thức bắt được lỗi đảo nhãn."""
    import protocols.convert_asvspoof2019_la as conv
    splits = {k: (v[0], fake_la["n"][k]) for k, v in conv.SPLITS.items()}
    monkeypatch.setattr(conv, "SPLITS", splits)
    right = {k: n // 2 for k, n in fake_la["n"].items()}  # i chẵn = bonafide
    monkeypatch.setattr(conv, "EXPECTED_BONA", right)
    conv.main(["--src", fake_la["proto_dir"], "--out_dir", str(tmp_path / "ok")])
    monkeypatch.setattr(conv, "LABEL_MAP", {"bonafide": 0, "spoof": 1})  # giả lập đảo nhãn
    with pytest.raises(ValueError, match="nhãn 1 \(thật\) nhưng attack.*đảo nhãn"):
        conv.main(["--src", fake_la["proto_dir"], "--out_dir", str(tmp_path / "bad")])


def test_convert_bona_count_check(fake_la, tmp_path, monkeypatch):
    """Nhánh kiểm tra số nhãn 1 chính thức (EXPECTED_BONA) tự nó báo lỗi khi số không khớp."""
    import protocols.convert_asvspoof2019_la as conv
    splits = {k: (v[0], fake_la["n"][k]) for k, v in conv.SPLITS.items()}
    monkeypatch.setattr(conv, "SPLITS", splits)
    wrong = {k: n // 2 + 1 for k, n in fake_la["n"].items()}
    monkeypatch.setattr(conv, "EXPECTED_BONA", wrong)
    with pytest.raises(ValueError, match="nhãn 1 \(thật\), kỳ vọng.*đảo nhãn"):
        conv.main(["--src", fake_la["proto_dir"], "--out_dir", str(tmp_path)])
