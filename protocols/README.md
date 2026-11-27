# protocols/

Protocol dùng chung, định dạng CSV có header: `file_id,speaker_id,label,attack_id,subset`
(label 1 = thật, 0 = giả — chốt 26/11/2026; attack `-` cho file thật).
Script kiểm tra số nhãn 1 chính thức: train 2.580, dev 2.548, eval 7.355.

Sinh bằng `python -m protocols.convert_asvspoof2019_la --src <ASVspoof2019_LA_cm_protocols> --out_dir protocols`.

## md5

md5 file CSV: sinh trên Kaggle bằng `prepare_npy.ipynb` (commit `ca4c91d`), dataset `phvster/asvspoof-la-npy`.
md5 file BTC gốc: xem output Cell 2 của `prepare_npy.ipynb` (dòng `md5 src`) rồi điền vào.

| File | Số dòng | md5 file BTC gốc | md5 file CSV |
|---|---|---|---|
| asvspoof2019_la_train.csv | 25.380 | _chưa có_ | `4eddfbde03e33018b0f74462067ee068` |
| asvspoof2019_la_dev.csv | 24.844 | _chưa có_ | `69a90a703b6cfe506a67cb727af9d10b` |
| asvspoof2019_la_eval.csv | 71.237 | _chưa có_ | `ffd9f515d2f44031672d73cf0e637286` |
