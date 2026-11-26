import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from common.audio import NB_SAMP
from common.data import AntiSpoofDataset, read_protocol, worker_init_fn
from protocols.convert_asvspoof2019_la import main as convert_main
from tests.conftest import write_flac


def test_convert_protocol(fake_la, tmp_path):
    md5 = convert_main(["--src", fake_la["proto_dir"], "--out_dir", str(tmp_path), "--no_count_check"])
    assert set(md5) == {"train", "dev", "eval"}
    df = read_protocol(str(tmp_path / "asvspoof2019_la_train.csv"))
    assert len(df) == fake_la["n"]["train"]
    assert set(df["subset"]) == {"train"}
    # i chẵn = bonafide -> 0, i lẻ = spoof -> 1; bonafide có attack "-"
    assert (df["label"].values == np.arange(len(df)) % 2).all()
    assert (df.loc[df["label"] == 0, "attack_id"] == "-").all()


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
    df = pd.DataFrame({"file_id": fids, "speaker_id": "S", "label": 0, "attack_id": "-",
                       "subset": "train"})
    return df, str(tmp_path)


def crops(ds, num_workers, epoch):
    ds.set_epoch(epoch)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                    worker_init_fn=worker_init_fn)
    return [x[0].numpy().copy() for x, _, _ in dl]


def test_two_workers_get_different_crops(same_long_audio):
    df, adir = same_long_audio
    ds = AntiSpoofDataset(df, adir, ".flac", mode="train", seed=0)
    c = crops(ds, 2, epoch=0)
    # batch_size=1, shuffle=False: idx 0 -> worker 0, idx 1 -> worker 1 (lần rút đầu của mỗi rng)
    assert not np.array_equal(c[0], c[1])
    assert len({round(float(v[0]) * 32768) for v in c}) > 1  # điểm bắt đầu crop khác nhau
    # tái lập: chạy lại cùng seed/epoch -> y hệt
    c2 = crops(ds, 2, epoch=0)
    for a, b in zip(c, c2):
        np.testing.assert_array_equal(a, b)
    # epoch khác -> crop khác
    c3 = crops(ds, 2, epoch=1)
    assert any(not np.array_equal(a, b) for a, b in zip(c, c3))
    # num_workers=0 dùng rng của "worker 0": crop đầu tiên trùng với worker 0
    c0 = crops(ds, 0, epoch=0)
    np.testing.assert_array_equal(c0[0], c[0])


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
