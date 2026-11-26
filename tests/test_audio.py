import numpy as np
import pytest
import soundfile as sf

from common.audio import NB_SAMP, SAMPLE_RATE, load_audio, pad, pad_random, prepare
from tests.conftest import write_flac
from tools.decode_to_npy import main as decode_main


# --- bản tham chiếu: copy nguyên văn clovaai/aasist data_utils.pad (commit a04c986)
def ref_pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


@pytest.mark.parametrize("n", [1, 1000, NB_SAMP - 1, NB_SAMP, NB_SAMP + 1, 3 * NB_SAMP + 7])
def test_lengths_and_ref(n):
    x = np.random.default_rng(n).uniform(-1, 1, n).astype(np.float32)
    w = pad(x)
    assert w.shape == (NB_SAMP,)
    np.testing.assert_array_equal(w, ref_pad(x, NB_SAMP))
    np.testing.assert_array_equal(w, pad(x))  # cố định
    r = pad_random(x, rng=np.random.default_rng(0))
    assert r.shape == (NB_SAMP,)
    np.testing.assert_array_equal(r, pad_random(x, rng=np.random.default_rng(0)))
    assert prepare(x, "full").shape == (n,)


def test_pad_random_crop_is_contiguous_and_varies():
    x = np.arange(3 * NB_SAMP, dtype=np.float32)
    starts = {int(pad_random(x, rng=np.random.default_rng(s))[0]) for s in range(10)}
    assert len(starts) > 1
    for s in range(3):
        r = pad_random(x, rng=np.random.default_rng(s))
        np.testing.assert_array_equal(np.diff(r), 1)


def test_prepare_bad_mode():
    with pytest.raises(ValueError):
        prepare(np.zeros(10, np.float32), "eval")


def test_load_audio_range_and_scale(tmp_path):
    x = np.array([-32768, -1, 0, 1, 32767] * 100, dtype=np.int16)
    p = str(tmp_path / "a.flac")
    write_flac(p, x)
    y = load_audio(p)
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y, x.astype(np.float32) / 32768.0)
    assert y.min() >= -1.0 and y.max() < 1.0


def test_load_audio_stereo_and_resample(tmp_path):
    st = np.stack([np.full(800, 1000, np.int16), np.full(800, 3000, np.int16)], axis=1)
    p = str(tmp_path / "s.wav")
    sf.write(p, st, 8000, subtype="PCM_16")
    y = load_audio(p)
    assert y.shape == (1600,)  # 8 kHz -> 16 kHz
    np.testing.assert_allclose(y[200:1400], 2000 / 32768.0, atol=1e-3)


def test_flac_vs_npy_identical(tmp_path):
    rng = np.random.default_rng(7)
    adir = tmp_path / "flac"
    adir.mkdir()
    fids = [f"F{i}" for i in range(5)]
    for i, f in enumerate(fids):
        write_flac(str(adir / f"{f}.flac"), rng.integers(-32768, 32767, 1000 + 7919 * i, dtype=np.int16))
    proto = tmp_path / "p.csv"
    proto.write_text("file_id,speaker_id,label,attack_id,subset\n"
                     + "".join(f"{f},S,0,-,train\n" for f in fids))
    out = tmp_path / "npy"
    decode_main(["--protocol", str(proto), "--audio_dir", str(adir), "--out_dir", str(out),
                 "--workers", "1"])
    for f in fids:
        a = load_audio(str(adir / f"{f}.flac"))
        b = load_audio(str(out / f"{f}.npy"))
        assert a.dtype == b.dtype == np.float32
        np.testing.assert_array_equal(a, b)
    # chạy lại: bỏ qua file đã có
    decode_main(["--protocol", str(proto), "--audio_dir", str(adir), "--out_dir", str(out),
                 "--workers", "1"])


def test_decode_rejects_non_16k(tmp_path):
    adir = tmp_path / "flac"
    adir.mkdir()
    write_flac(str(adir / "X.flac"), np.zeros(800, np.int16), sr=8000)
    proto = tmp_path / "p.csv"
    proto.write_text("file_id,speaker_id,label,attack_id,subset\nX,S,0,-,train\n")
    with pytest.raises(ValueError):
        decode_main(["--protocol", str(proto), "--audio_dir", str(adir),
                     "--out_dir", str(tmp_path / "npy"), "--workers", "1"])


def test_npy_wrong_dtype(tmp_path):
    p = str(tmp_path / "x.npy")
    np.save(p, np.zeros(10, np.float32))
    with pytest.raises(ValueError):
        load_audio(p)


def test_sample_rate_constant():
    assert SAMPLE_RATE == 16000
