# antispoof-benchmark

Benchmark các model phát hiện giọng giả (anti-spoofing) để chọn model cho app Android.
Quy ước chung: xem `CONVENTIONS.md` (bản tóm tắt). Tài liệu quy ước gốc là nguồn chính.

## Cài đặt local (Windows, CPU — chỉ code + smoke test)

```bash
python -m venv .venv
.venv\Scripts\activate          # Git Bash: source .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q
```

`requirements.txt` cố ý KHÔNG có torch: Kaggle đã cài sẵn torch GPU, nếu `pip install torch`
ở đó sẽ ghi đè bản CUDA.

## Cấu trúc

| Đường dẫn | Nội dung | Dùng chung |
|---|---|---|
| `common/audio.py` | `load_audio()` (flac/wav hoặc .npy int16), `NB_SAMP`, `pad` / `pad_random` / `prepare(mode)` | A + B |
| `common/data.py` | `read_protocol()`, `AntiSpoofDataset`, `worker_init_fn` | A + B |
| `common/metrics.py` | EER (port clovaai/aasist), `eer_from_labels()` theo quy ước điểm cao = thật (không đổi dấu) | A + B |
| `common/train.py` | vòng train: weighted CE [0.1, 0.9], chọn theo dev EER, 100 epoch, patience 10, resume, `--max_minutes` | A + B |
| `common/score_export.py` | xuất `scores/<mode>/<model>_<train>_<test>_<aug>_seed<k>.txt` | A + B |
| `protocols/convert_asvspoof2019_la.py` | protocol BTC 2019 LA -> CSV `file_id,speaker_id,label,attack_id,subset` + md5 | A + B |
| `tools/decode_to_npy.py` | giải mã FLAC -> .npy int16 một lần (đọc nhanh hơn khi train) | A + B |
| `eval/score.py` | chấm file điểm, min t-DCF (B phụ trách) | A + B |
| `models/<tên>/` | `model.py` + `config.yaml` | từng người |

## Giao diện model

Mỗi model là một `nn.Module` có:
- `logits(wave[B, T]) -> [B, 2]`: cột 0 = giả, cột 1 = thật (nhãn chung: 1 = thật, 0 = giả, giống clovaai);
- `forward(wave[B, T]) -> score[B] = logits[:,1] - logits[:,0]` (cao = thật).

Config model khai báo `model.class: "models.<tên>.model:<Class>"` và `model.args`.

## Luồng chạy

```bash
# 1. protocol (một lần, ghi md5 vào protocols/README.md)
python -m protocols.convert_asvspoof2019_la --src <LA>/ASVspoof2019_LA_cm_protocols --out_dir protocols

# 2. (tùy chọn) giải mã sẵn sang .npy, rồi dùng --ext .npy
python -m tools.decode_to_npy --protocol protocols/asvspoof2019_la_train.csv \
    --audio_dir <LA>/ASVspoof2019_LA_train/flac --out_dir <npy>/train

# 3. train (chạy lại đúng lệnh này để resume)
python -m common.train --config models/<tên>/config.yaml --seed 0 --out_dir runs/<tên>_seed0 \
    --train_protocol protocols/asvspoof2019_la_train.csv --train_audio_dir <...> \
    --dev_protocol protocols/asvspoof2019_la_dev.csv --dev_audio_dir <...> --ext .flac --max_minutes 660

# 4. file điểm
python -m common.score_export --checkpoint runs/<tên>_seed0/best.pt \
    --protocol protocols/asvspoof2019_la_eval.csv --audio_dir <...> --ext .flac \
    --mode window --train_name la19 --test_name la19eval --aug A0 --seed 0
```

Mỗi run lưu: `config_used.yaml`, `env.json` (torch, GPU, batch size, lr, commit), `log.csv` (từng
epoch), `best.pt`, `last.pt`, `train_time.txt`, `run_status.json` (`paused` / `early_stopped` / `finished`).
