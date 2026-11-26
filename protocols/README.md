# protocols/

Protocol dùng chung, định dạng CSV có header: `file_id,speaker_id,label,attack_id,subset`
(label 1 = thật, 0 = giả — chốt 26/11/2026; attack `-` cho file thật).
Script kiểm tra số nhãn 1 chính thức: train 2.580, dev 2.548, eval 7.355.

Sinh bằng `python -m protocols.convert_asvspoof2019_la --src <ASVspoof2019_LA_cm_protocols> --out_dir protocols`.

## md5 (điền sau khi chạy với file thật của BTC)

| File | Số dòng | md5 file BTC gốc | md5 file CSV |
|---|---|---|---|
| asvspoof2019_la_train.csv | 25.380 | _chưa có_ | _chưa có_ |
| asvspoof2019_la_dev.csv | 24.844 | _chưa có_ | _chưa có_ |
| asvspoof2019_la_eval.csv | 71.237 | _chưa có_ | _chưa có_ |
