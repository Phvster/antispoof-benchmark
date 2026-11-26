"""Dữ liệu giả cho test: cây thư mục + protocol giống ASVspoof 2019 LA của BTC (rất nhỏ)."""
import os

import numpy as np
import pytest
import soundfile as sf

from common.audio import NB_SAMP, SAMPLE_RATE

SPLIT_FILES = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "T", 16),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "D", 8),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "E", 8),
}


def write_flac(path, x_int16, sr=SAMPLE_RATE):
    sf.write(path, x_int16, sr, subtype="PCM_16", format="FLAC")


@pytest.fixture(scope="session")
def fake_la(tmp_path_factory):
    """Trả dict: root, proto_dir, audio_dir[subset], n[subset]. Độ dài file 0,5–6 s
    (có file ngắn hơn và dài hơn NB_SAMP); spoof có thêm sóng sin để model học được chút."""
    root = tmp_path_factory.mktemp("LA")
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir()
    rng = np.random.default_rng(123)
    out = {"root": str(root), "proto_dir": str(proto_dir), "audio_dir": {}, "n": {}}
    for subset, (fname, letter, n) in SPLIT_FILES.items():
        adir = root / f"ASVspoof2019_LA_{subset}" / "flac"
        adir.mkdir(parents=True)
        lines = []
        for i in range(n):
            fid = f"LA_{letter}_{1000000 + i}"
            spoof = i % 2 == 1
            length = int(rng.integers(SAMPLE_RATE // 2, 6 * SAMPLE_RATE))
            if i == 0:
                length = NB_SAMP  # biên: đúng bằng NB_SAMP
            x = rng.normal(0, 0.05, length)
            if spoof:
                x += 0.1 * np.sin(2 * np.pi * 3000 * np.arange(length) / SAMPLE_RATE)
            write_flac(str(adir / f"{fid}.flac"), np.clip(x * 32767, -32768, 32767).astype(np.int16))
            spk = f"LA_00{subset[0]}{i % 3}"
            attack = ("A0%d" % (1 + i % 6)) if spoof else "-"
            lines.append(f"{spk} {fid} - {attack} {'spoof' if spoof else 'bonafide'}\n")
        (proto_dir / fname).write_text("".join(lines))
        out["audio_dir"][subset] = str(adir)
        out["n"][subset] = n
    return out
