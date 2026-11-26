# kaggle/ — chạy AASIST-L trên Kaggle

Notebook: `aasist_l.ipynb`. Accelerator: **GPU T4 ×2**. Internet: **bật** (clone repo, tải pretrained).

## 1. Chuẩn bị một lần: dataset `.npy`

Đọc FLAC trên Kaggle chậm (CPU yếu, chỉ 4 nhân), nên giải mã sẵn sang `.npy` int16 **một lần**.
`load_audio()` cho kết quả **giống hệt** FLAC (có test `tests/test_audio.py::test_flac_vs_npy_identical`).

Trong một notebook phụ (CPU là đủ), input = dataset ASVspoof 2019 LA gốc:

```bash
git clone https://github.com/Phvster/antispoof-benchmark.git && cd antispoof-benchmark
pip install -q -r requirements.txt
LA=/kaggle/input/<asvspoof2019-la>/LA          # đổi theo dataset thật
OUT=/kaggle/working
python -m protocols.convert_asvspoof2019_la --src $LA/ASVspoof2019_LA_cm_protocols --out_dir $OUT/protocols
for s in train dev eval; do
  python -m tools.decode_to_npy --protocol $OUT/protocols/asvspoof2019_la_$s.csv \
      --audio_dir $LA/ASVspoof2019_LA_$s/flac --out_dir $OUT/$s
done
```

- Ghi md5 mà script in ra vào `protocols/README.md`.
- Tổng dung lượng khoảng 13 GB, vừa giới hạn 20 GB của `/kaggle/working`.
- Save Version, rồi tạo **Dataset từ output** của notebook này.

Layout dataset (tên dataset là tham số `NPY_DATASET` trong notebook):

```
<NPY_DATASET>/
  protocols/asvspoof2019_la_{train,dev,eval}.csv
  train/<file_id>.npy   dev/<file_id>.npy   eval/<file_id>.npy
```

## 2. Chạy: Save Version → Save & Run All

1. Gắn input `NPY_DATASET`. Sửa **Cell 0**: `NPY_DATASET`, `REPO_REF`, `SEEDS = [0, 1]`.
   `REPO_REF` nên là **hash commit đã push**, không phải tên nhánh, để mọi seed và mọi version chạy cùng code
   (`env.json` ghi lại `git_commit`).
2. **Save Version → Save & Run All (Commit)**. Không chạy tay từng cell: phiên tương tác bị ngắt khi đóng tab.
3. Các cell lần lượt làm:
   - Cell 2 chấm pretrained AASIST-L ở **chế độ cửa sổ**, ghi
     `scores/window/aasist_l_la19_la19eval_A0_seedpretrained.txt`. EER lệch khỏi khoảng [0,5; 1,5]% thì notebook dừng.
   - Cell 4 chạy 2 tiến trình `common.train` song song (`CUDA_VISIBLE_DEVICES=0` và `1`). Mỗi tiến trình có `--max_minutes 660`.
   - Cell 5 xuất file điểm dev/eval (chế độ cửa sổ) từ `best.pt`. Chỉ làm khi run đã `finished` hoặc `early_stopped`.

Phiên Save & Run All tối đa 12 giờ. Chia thời gian như sau:
- 660 phút train;
- khoảng 15 phút chấm pretrained (chỉ phiên đầu);
- phần còn lại để xuất điểm.

`--max_minutes` chỉ chặn việc **bắt đầu** epoch mới. Epoch đang chạy dở sẽ chạy hết, nên đừng tăng quá 660.

## 3. Resume ở version sau

`/kaggle/working` mất sau mỗi phiên. Muốn đi tiếp:

1. **Add Input → Your Work → output của version trước**, ví dụ `/kaggle/input/aasist-l-run-v1`.
2. Cell 0: `PREV_RUN = "/kaggle/input/aasist-l-run-v1"`. `RUN_PRETRAINED_CHECK` tự tắt.
3. Save & Run All. Cell 3 chép `runs/` và `scores/` về `/kaggle/working`. `common.train --resume auto` nạp `last.pt` (model, optimizer, scheduler, RNG) và đi tiếp đúng epoch.
4. Lặp lại tới khi `run_status.json` là `finished` hoặc `early_stopped`. Sau đó chạy seed 2 bằng `SEEDS = [2]`.

Crop train dùng `default_rng([seed, epoch, idx])`, nên kết quả không phụ thuộc `num_workers` hay việc resume.

## 4. t-DCF

min t-DCF (2019 LA) cần file điểm ASV của BTC:
**`ASVspoof2019.LA.asv.eval.gi.trl.scores.txt`**. File này nằm trong dataset LA gốc, ở
`LA/ASVspoof2019_LA_asv_scores/`. Dataset `.npy` không chứa nó, nên khi chấm phải gắn thêm dataset LA gốc, hoặc
chép file này vào dataset `.npy`. Chấm bằng `eval/score.py` (B phụ trách) từ các file điểm trong `scores/`.

## 5. Sau khi chạy

- Mang `runs/aasist_l_seed*/` (gồm `log.csv`, `env.json`, `config_used.yaml`, `run_status.json`) và `scores/` về.
  Gọi `experiment-auditor` trước khi điền bảng kết quả.
- Không commit `*.pt`, `*.pth`, `*.npy` (đã có trong `.gitignore`).
- Repo private: thêm GitHub token qua **Kaggle Secrets** và clone bằng `https://<token>@github.com/...`.
  Không ghi token vào notebook.
