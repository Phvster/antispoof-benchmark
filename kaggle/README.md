# kaggle/ — chạy AASIST-L trên Kaggle

| Notebook | Accelerator | Việc |
|---|---|---|
| `prepare_npy.ipynb` | None (CPU) | một lần: protocol chung + giải mã FLAC → `.npy` → tạo Dataset |
| `aasist_l.ipynb` | **GPU T4 ×2** | chấm pretrained, train 2 seed song song, resume, xuất file điểm |

Cả hai cần **Internet: bật**. Chạy bằng **Save Version → Save & Run All (Commit)**. Không chạy tay từng cell,
vì phiên tương tác bị ngắt khi đóng tab.

## 0. Repo private: token chỉ đọc

1. Trên GitHub, tạo **fine-grained token**:
   - Repository access: *Only select repositories* → `Phvster/antispoof-benchmark`;
   - Permissions: **Contents: Read-only** (không cần quyền nào khác).
2. Trên Kaggle, trong notebook: **Add-ons → Secrets → Add secret**, tên `GITHUB_TOKEN`, rồi bật secret đó cho notebook.
3. Notebook đọc token qua `UserSecretsClient` lúc clone. Token không bao giờ được in ra hay ghi vào notebook:
   - repo clone vào `/tmp`, nằm ngoài `/kaggle/working`, nên không lọt vào output version;
   - sau khi clone, remote được đặt lại thành URL không có token.

Mỗi notebook có `REPO_REF` = **hash commit đã push**, không dùng tên nhánh, để mọi seed và mọi version chạy cùng
một code. `env.json` của mỗi run ghi lại `git_commit`.

## 1. Chuẩn bị một lần: dataset `.npy` (`prepare_npy.ipynb`)

Kaggle chỉ có 4 nhân CPU yếu nên đọc FLAC chậm. Vì vậy giải mã sẵn sang `.npy` int16 **một lần**.
`load_audio()` đọc `.npy` cho kết quả **giống hệt** FLAC. Test `test_flac_vs_npy_identical` kiểm tra điều này, và notebook cũng
so lại một file mỗi tập.

1. Input: dataset ASVspoof 2019 LA gốc. Cell 0: sửa `LA` cho đúng đường dẫn của dataset.
2. Save & Run All. Output khoảng 13 GB (giới hạn 20 GB):
   ```
   protocols/asvspoof2019_la_{train,dev,eval}.csv
   train/<file_id>.npy   dev/<file_id>.npy   eval/<file_id>.npy
   asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt     # cho min t-DCF
   ```
3. Ghi md5 protocol mà Cell 2 in ra vào `protocols/README.md`.
4. **New Dataset** từ output của version này. Tên dataset là tham số `NPY_DATASET` trong `aasist_l.ipynb`.

## 2. Chạy thử lần đầu (`TRIAL_RUN = True`, mặc định)

`TRIAL_RUN = True` đặt `--max_minutes 10` và `--stop_after_epochs 2`: mỗi phiên chạy tối đa 2 epoch rồi dừng ở trạng thái `paused`.
Cờ này không đổi `MAX_EPOCHS`, nên lịch learning rate và kết quả giống hệt chạy liền.

**Chép dữ liệu sang ổ local (`COPY_TO_LOCAL = True`, mặc định).** Lần chạy thử đầu đọc thẳng từ `/kaggle/input`
(ổ mạng) chỉ được khoảng 28–30 file/s, nên mất 55 phút/epoch. Cell 1b chép `train/` và `dev/` (khoảng 50 nghìn file) sang
`/tmp/la_npy` bằng 64 luồng, rồi in số file/s và dung lượng ổ. Dữ liệu giống hệt từng bit nên kết quả không đổi. Tập `eval/`
chỉ được chép khi cần (chấm pretrained hoặc xuất điểm).

**Chấm pretrained (`RUN_PRETRAINED_CHECK`) mặc định tắt**, vì đã kiểm chứng: EER = **0,992%** (commit `ca4c91d`).
Bật lại khi đổi `load_audio`, `score_export`, `metrics` hoặc model.

**Version 1** (Cell 0: `NPY_DATASET`, `PREV_RUN = None`):
- Nếu bật `RUN_PRETRAINED_CHECK`: Cell 2 chấm pretrained AASIST-L ở **chế độ cửa sổ**, ghi
  `scores/window/aasist_l_la19_la19eval_A0_seedpretrained.txt`. Kỳ vọng EER ≈ 0,99%; nếu nằm ngoài khoảng [0,5; 1,5]% thì notebook dừng.
- Cell 4 in tiến độ của từng seed mỗi 2 phút: số batch, mẫu/s, **% thời gian chờ dữ liệu**, thời gian còn lại.
- Cell 4 train seed 0 và 1 song song (`CUDA_VISIBLE_DEVICES=0` và `1`) và ghi `last.pt`.
- Cell 6 in số phút / epoch, tách thành train và dev, kèm thời gian chờ dữ liệu của mỗi phần, để tính quota và biết còn nghẽn ở đâu.

**Version 2**: Add Input → Your Work → output của version 1. Cell 0: `PREV_RUN = "/kaggle/input/<output v1>"`.
- Cell 3 chép `runs/` và `scores/` của version 1. `--resume auto` nạp `last.pt` rồi chạy tiếp từ đúng epoch.
- Kiểm tra: `log.csv` phải **liền mạch** (epoch 0, 1 từ v1, rồi 2, 3 từ v2, không trùng, không mất).

Mang về các thứ sau rồi đưa cho `experiment-auditor`:
- tốc độ chép ở Cell 1b (file/s, thời gian chép);
- `log.csv` + `run_status.json` + `env.json` của cả hai version;
- số phút / epoch.

## 3. Train thật (`TRIAL_RUN = False`)

- `--max_minutes 660`. Phiên Save & Run All tối đa 12 giờ, chia như sau:
  - 660 phút train;
  - khoảng 15 phút chấm pretrained (chỉ ở phiên đầu);
  - phần còn lại để xuất điểm.
- `--max_minutes` chỉ chặn việc **bắt đầu** epoch mới. Epoch đang chạy dở sẽ chạy hết, nên đừng tăng quá 660.
- Resume như version 2 ở trên. Lặp lại tới khi `run_status.json` là `finished` hoặc `early_stopped`. Sau đó chạy seed 2
  bằng `SEEDS = [2]`.
- Cell 5 xuất file điểm dev/eval (chế độ cửa sổ, thêm chế độ cả file nếu `EXPORT_FULL`) từ `best.pt` của run đã xong.

Crop train dùng `default_rng([seed, epoch, idx])`, nên kết quả không phụ thuộc `num_workers` hay việc resume.

## 4. t-DCF

min t-DCF (2019 LA) cần file điểm ASV của BTC **`ASVspoof2019.LA.asv.eval.gi.trl.scores.txt`**.
`prepare_npy.ipynb` đã chép file này vào `asv_scores/` của dataset `.npy`. Chấm bằng `eval/score.py` (B phụ trách) từ các file
điểm trong `scores/`.

## 5. Sau khi chạy

- Mang `runs/aasist_l_seed*/` (gồm `log.csv`, `env.json`, `config_used.yaml`, `run_status.json`) và `scores/` về.
  Gọi `experiment-auditor` trước khi điền bảng kết quả.
- Không commit `*.pt`, `*.pth`, `*.npy` (đã có trong `.gitignore`).
