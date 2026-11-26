# CONVENTIONS.md — tóm tắt quy ước chung (bản trong repo)

Nguồn chính thức: https://claude.ai/artifact/WLWq2e6Ymw3TL58PBT445X
Nếu file này khác tài liệu đó → tài liệu thắng; cập nhật lại file này.

## Phân công
- A: LFCC-LCNN, LiteAntiSpoofNet, AASIST-L — đo độ trễ & kích thước .tflite trên Android.
- B: RawNet2, AASIST, XLS-R + AASIST — script chấm chung, điểm XLS-R làm nhãn mềm cho A distill.

## Dữ liệu
| Bộ | Train | Dev | Test |
|---|---|---|---|
| ASVspoof 2019 LA | train chính thức | dev chính thức | eval chính thức (71.237 file) |
| VSASV (ghi rõ bản HF 2024 hay VLSP 2025) | train trừ dev | 10% người nói của train, seed 42 | test chính thức |
| In-the-Wild | – | – | toàn bộ (chỉ thử tổng quát) |
| LiveKit tự thu | tùy thí nghiệm | tách theo người nói | tách theo người nói |

- Nhãn: `0 = thật (bona fide)`, `1 = giả (spoof)`. **Điểm càng cao càng giả.**
- Protocol: cột `file_id, speaker_id, label, attack_id, subset`; ghi md5 từng file protocol vào bảng kết quả.
- Không dùng test để chọn epoch, ngưỡng hay siêu tham số.

## Tiền xử lý
- 16 kHz, mono, PCM 16-bit chia 32768 về [-1, 1]; dùng chung `load_audio()`.
- Độ dài: tài liệu hiện vẫn ghi 64.000; đề xuất sửa → 64.600 mẫu cho khớp code AASIST là suggestion CHƯA được chấp nhận (chờ cả hai duyệt). Code đang để `NB_SAMP = 64600`. Dùng hằng số `NB_SAMP` trong `common/audio.py`, không viết số cứng ở nơi khác.
- Train: cắt ngẫu nhiên, lặp nếu ngắn. Chấm chế độ cửa sổ: đoạn đầu, lặp nếu ngắn. Chấm chế độ cả file: toàn bộ.
- Thêm một lần chấm sau khi cắt im lặng bằng Silero VAD ngưỡng 0.5.
- Đặc trưng (LFCC, log-mel…) tính BÊN TRONG model; đầu vào luôn là waveform.

## Huấn luyện
- 3 seed: **0, 1, 2**; báo mean ± std và seed tốt nhất.
- Chọn checkpoint theo **dev EER**. Tối đa 100 epoch, dừng sớm sau 10 epoch dev EER không giảm.
- Weighted CE: 0.9 cho thật (lớp ít), 0.1 cho giả → với nhãn trên là `[0.9, 0.1]`. Lưu ý: câu chữ hiện tại trong tài liệu ghi "[0.9 cho giả, 0.1 cho thật]" — NGƯỢC với code AASIST; suggestion sửa câu chữ CHƯA được chấp nhận. Code dùng `[0.9, 0.1]` (thật, giả).
- Augmentation: A0 = không; A1 = RawBoost (cấu hình 3) + Opus 8–32 kbps + MUSAN SNR 5–20 dB. Báo riêng.
- Ghi: phiên bản PyTorch, GPU, batch size, lr, thời gian train.

## Đánh giá
- File điểm: `scores/<model>_<train>_<test>_<aug>_seed<k>.txt`, mỗi dòng `file_id score`.
- Đề xuất (chờ B đồng ý): ghi chế độ chấm bằng thư mục con `scores/window/` và `scores/full/`, vì tên file không có trường chế độ.
- Chỉ dùng script chấm chung (dựa trên evaluation của clovaai/aasist): EER ở `common/metrics.py`, chấm file điểm + min t-DCF ở `eval/score.py` (theo tài liệu; repo `Phvster/antispoof-benchmark`).
- Bắt buộc: EER toàn tập; min t-DCF (2019 LA, dùng điểm ASV của BTC); EER theo từng kiểu tấn công; FPR giọng thật tại ngưỡng EER của dev.
- Ngưỡng vận hành = điểm EER trên dev, áp nguyên sang test.
- Model gốc dùng nhãn khác (vd AASIST: bonafide = 1) → đổi khi xuất điểm: `score = logit_spoof - logit_bonafide`.

## Android (A đo cho mọi model, cùng một máy, ghi tên máy)
- Số tham số, GFLOPs / cửa sổ 4 s.
- Kích thước .tflite float32 và INT8 dynamic-range.
- Độ trễ median / cửa sổ, CPU 4 luồng, 100 lần sau 10 lần khởi động.
- EER sau TFLite trên cùng tập test.
- Ngưỡng chấp nhận: < 150 ms/cửa sổ; EER tăng ≤ 0,3 điểm % sau INT8.

## Quy trình nộp
1. Train → chọn checkpoint theo dev → xuất file điểm → chạy script chấm chung.
2. Điền một dòng vào bảng kết quả trong tài liệu, đẩy file điểm + config lên thư mục chung (thiếu file điểm = không tính).
3. Người kia chấm lại chéo từ file điểm.
4. Muốn đổi quy ước: comment vào đúng mục trong tài liệu, cả hai đồng ý rồi mới chạy lại.
