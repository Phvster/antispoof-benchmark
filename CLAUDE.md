# CLAUDE.md — Deepfake Voice Detection benchmark (người A)

## Bối cảnh
- Mục tiêu: chọn model phát hiện giọng giả (anti-spoofing) để chạy trong app Android (Kotlin, cuộc gọi qua LiveKit).
- Nhóm 2 người, chung một repo:
  - **A (tôi): LFCC-LCNN, LiteAntiSpoofNet, AASIST-L.** Việc chung: đo độ trễ và kích thước `.tflite` trên Android cho MỌI model.
  - B: RawNet2, AASIST, XLS-R + AASIST. Việc chung: script chấm chung, điểm XLS-R làm nhãn mềm để A distill.
- Nguồn quy ước duy nhất: tài liệu "Quy ước benchmark model chống giả mạo giọng nói" (Claude Docs, link trong `CONVENTIONS.md`). `CONVENTIONS.md` trong repo là bản tóm tắt; nếu khác nhau thì tài liệu thắng.
- **Đọc `CONVENTIONS.md` trước khi sửa bất kỳ thứ gì về dữ liệu, tiền xử lý, nhãn, điểm số, train hoặc metric.**

## Môi trường
- Local: Windows, VS Code + Claude Code. Chỉ code + smoke test CPU với dữ liệu giả/rất nhỏ.
- Train thật: Kaggle Notebook (GPU T4/P100). Notebook clone repo từ GitHub, dữ liệu qua Kaggle Datasets.
  - Dữ liệu: `/kaggle/input/...` (chỉ đọc). Output: `/kaggle/working/...`.
  - Phiên Kaggle có giới hạn thời gian → train PHẢI resume được từ checkpoint.
- Không hard-code đường dẫn; đặt trong `config.yaml` hoặc tham số CLI.

## Cấu trúc repo (theo quy ước)
- `common/audio.py` — `load_audio()` DÙNG CHUNG (16 kHz mono, [-1,1], pad/crop). Không ai tự viết bản riêng.
- `common/data.py` — đọc protocol (`file_id, speaker_id, label, attack_id, subset`), Dataset train/dev/eval.
- `common/train.py` — vòng train chung: weighted CE, chọn checkpoint theo dev EER, early stopping, resume, 3 seed.
- `common/augment.py` — cấu hình A0 / A1 (RawBoost 3 + Opus 8–32 kbps + MUSAN SNR 5–20 dB).
- `models/<tên>/model.py` + `models/<tên>/config.yaml` — mỗi model một thư mục, giao diện `forward(waveform) -> score` (điểm càng cao càng THẬT; `logits` cột 0 = giả, 1 = thật).
  - Của tôi: `models/lfcc_lcnn/`, `models/liteantispoofnet/`, `models/aasist_l/`.
- `common/metrics.py` — hàm EER chung (port clovaai/aasist; A viết, B review). `eval/score.py` — script chấm file điểm + min t-DCF (B phụ trách). **Không sửa logic tính EER/t-DCF.**
- `protocols/` — file protocol dùng chung + md5.
- `scores/window/`, `scores/full/` — file điểm `<model>_<train>_<test>_<aug>_seed<k>.txt`, mỗi dòng `file_id score`.
- `export/` — PyTorch → ONNX → TFLite (float32 + INT8 dynamic-range), test so khớp, script đo trên Android (tôi phụ trách).
- `kaggle/` — notebook chạy train/eval.
- `tests/` — smoke test.

## Quy tắc làm việc
1. Task lớn: lập plan ngắn, liệt kê file sẽ sửa, chờ tôi duyệt.
2. Mọi thay đổi phải qua smoke test local (CPU, audio giả, 1 epoch, vài batch, ra được file điểm) trước khi đẩy lên Kaggle.
3. Không đổi: sample rate, độ dài input, `load_audio()`, quy ước nhãn/dấu điểm, trọng số loss, cách chọn checkpoint, script chấm — trừ khi tài liệu quy ước đã được cả hai đồng ý sửa.
4. Code chung (`common/`, `eval/`, `protocols/`) đổi thì phải báo B; ưu tiên PR riêng.
5. Mỗi lần chạy lưu: config đã dùng, log từng epoch (CSV), checkpoint tốt nhất, file điểm, phiên bản PyTorch, GPU, batch size, lr, thời gian train.
6. Không commit audio, checkpoint lớn hay dữ liệu Kaggle.
7. Port model từ repo gốc (AASIST, LFCC-LCNN baseline ASVspoof 2021): giữ đúng kiến trúc; chỉ bọc lại cho giao diện chung, ghi rõ nguồn trong docstring.
8. Giải thích cho tôi bằng tiếng Việt.

## Subagent
- `conventions-guard`: sau khi sửa `common/`, config, protocol, `eval/` hoặc bất cứ thứ gì ảnh hưởng tới so sánh với B.
- `experiment-auditor`: khi tôi đưa log/file điểm từ Kaggle về, trước khi điền bảng kết quả.
- `export-checker`: khi làm export ONNX/TFLite, INT8, hoặc đo độ trễ Android.

## Mốc kiểm chứng
- AASIST-L pretrained (clovaai/aasist) trên ASVspoof 2019 LA eval, chấm **chế độ cửa sổ 64.600 mẫu đầu** (đúng cách clovaai chấm, `pad()` trong `data_utils.py`; không phải cả file): EER ≈ 0,99%, min t-DCF ≈ 0,0309, 85.306 tham số. Lệch xa → pipeline sai (thường do dấu điểm hoặc nhãn). Quy ước nhãn chốt 26/11/2026: 1 = thật, 0 = giả, điểm cao = thật, CE `[0.1, 0.9]` — giống clovaai, không đổi dấu; chỉ đổi sang `p_fake` khi export Android.
