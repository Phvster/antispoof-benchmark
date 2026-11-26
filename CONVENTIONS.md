# CONVENTIONS.md — tóm tắt quy ước chung (bản trong repo)

Nguồn chính thức: https://claude.ai/artifact/WLWq2e6Ymw3TL58PBT445X
Nếu file này khác tài liệu đó → tài liệu thắng; cập nhật lại file này.

## Còn chờ duyệt trong tài liệu
- T1 / T2 / T3 (ba cách train: chỉ 2019 LA, chỉ VSASV, fine-tune 2019 LA → VSASV) — chưa code.

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

- Nhãn (chốt 26/11/2026): `1 = thật (bona fide)`, `0 = giả (spoof)`. **Điểm càng cao càng THẬT.** Giống code chính thức ASVspoof/AASIST/RawNet2/SSL_Anti-spoofing nên không đổi dấu. Chỉ khi đưa lên Android mới đổi thành `p_fake`.
- Protocol: cột `file_id, speaker_id, label, attack_id, subset`; ghi md5 từng file protocol vào bảng kết quả.
- Không dùng test để chọn epoch, ngưỡng hay siêu tham số.

## Tiền xử lý
- 16 kHz, mono, PCM 16-bit chia 32768 về [-1, 1]; dùng chung `load_audio()`.
- Độ dài: **64.600 mẫu** (~4,04 s, đúng nb_samp của code AASIST) — đã chốt. Dùng hằng số `NB_SAMP` trong `common/audio.py`, không viết số cứng ở nơi khác.
- Train: cắt ngẫu nhiên, lặp nếu ngắn; rng theo từng mẫu `default_rng([seed, epoch, idx])` (không phụ thuộc num_workers, máy, resume). Chấm chế độ cửa sổ: đoạn đầu, lặp nếu ngắn. Chấm chế độ cả file: toàn bộ.
- Thêm một lần chấm sau khi cắt im lặng bằng Silero VAD ngưỡng 0.5.
- Đặc trưng (LFCC, log-mel…) tính BÊN TRONG model; đầu vào luôn là waveform.

## Huấn luyện
- 3 seed: **0, 1, 2**; báo mean ± std và seed tốt nhất.
- Chọn checkpoint theo **dev EER** (tính ở chế độ cửa sổ). Tối đa 100 epoch, dừng sớm sau 10 epoch dev EER không giảm.
- Weighted CE như code AASIST: 0,1 cho giả (nhãn 0), 0,9 cho thật (nhãn 1, lớp ít) → tensor `[0.1, 0.9]`.
- Augmentation: A0 = không; A1 = RawBoost (cấu hình 3) + Opus 8–32 kbps + MUSAN SNR 5–20 dB. Báo riêng.
- Ghi: phiên bản PyTorch, GPU, batch size, lr, thời gian train.

## Đánh giá
- File điểm: `scores/<mode>/<model>_<train>_<test>_<aug>_seed<k>.txt` với `<mode>` = `window` | `full` (đã chốt), mỗi dòng `file_id score`. Checkpoint pretrained của tác giả: `seed = pretrained`.
- Chỉ dùng script chấm chung (dựa trên evaluation của clovaai/aasist): EER ở `common/metrics.py`, chấm file điểm + min t-DCF ở `eval/score.py` (theo tài liệu; repo `Phvster/antispoof-benchmark`).
- Bắt buộc: EER toàn tập; min t-DCF (2019 LA, dùng điểm ASV của BTC); EER theo từng kiểu tấn công; FPR giọng thật tại ngưỡng EER của dev.
- Ngưỡng vận hành = điểm EER trên dev, áp nguyên sang test.
- Model: `logits(x)` cột 0 = giả, cột 1 = thật; `score = logit[:,1] - logit[:,0]` (cao = thật). AASIST/clovaai đã cùng nhãn → không đảo cột.
- Kiểm chứng pipeline bằng pretrained clovaai ở **chế độ cửa sổ**: AASIST-L ≈ EER 0,99% / min t-DCF 0,0309; AASIST ≈ 0,83% / 0,0275.

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
