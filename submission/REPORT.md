# Báo cáo Day 08 — 03096 - Đinh Hồng Hải

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26s | car=0.45, person=0.3, bicycle=0.3, motorbike=0.3, bus=0.3, truck=0.3 | YOLO26s nhận diện tốt 6 lớp COCO (r01, r02). Từ kết quả threshold_sweep.md, lớp car đạt recall 100% tại ngưỡng 0.45 (precision 73% trên 8 vật gold). Các lớp person (recall max 86%), bicycle (max 57%), motorbike (max 69%) không mức nào <=0.5 đạt recall 90% nên giữ ngưỡng mặc định 0.3 để tránh sót. Lớp bus và truck có 0 mẫu trong gold (< 5) nên theo quy tắc mẫu nhỏ giữ mặc định 0.3. |
| three_wheeler | YOLOE-26s, prompt: "tuk-tuk, cyclo, cargo tricycle" | 0.15 | YOLOE giải quyết được điểm yếu nhận diện xe ba bánh mà YOLO26 bỏ qua (r04). Kết quả quét threshold_sweep.md cho thấy tại ngưỡng 0.15 đạt recall 100% (precision 67% trên 6 vật gold), giúp bắt trọn tuk-tuk và xe ba gác trên đường. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "car=0.45,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1 base (`sam2.1_b.pt`)**. Lý do: chất lượng đường viền mask bám rất sát chi tiết thân xe và người, không bị vỡ vụn biên như mobile_sam; tốc độ xử lý và dung lượng RAM/VRAM cân bằng tốt trên máy local (~4.3s/ảnh trên CPU, dưới 1s trên GPU); không yêu cầu thủ tục xin cấp quyền token phức tạp như SAM 3 (đã kiểm chứng trực quan ở lần chạy r03).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Đó là lỗi **sai phân loại (misclassification)** do giới hạn danh mục đóng: YOLO26 chỉ biết 80 lớp COCO nên gặp xe ba bánh địa phương (tuk-tuk, xích lô, ba gác) sẽ ép về `car`, `truck` hoặc `motorbike` (như ở `img_002.jpg`). Tăng hay hạ `--conf` chỉ quyết định giữ hay lọc bỏ box chứ không đổi được tên lớp. Cách sửa: Phải kết hợp mô hình open-vocabulary (YOLOE có text prompt riêng cho `three_wheeler`), dùng `merge_prelabels.py` để đè nhãn `three_wheeler` lên `car`/`truck` khi trùng vị trí (IoU >= 0.6), và con người phải can thiệp sửa nhãn trong bước review.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - *Điểm cao mà sai:* Ở `img_002.jpg`, YOLO26 phát hiện chiếc xe tuk-tuk nhưng gán nhãn `car` với điểm conf rất cao (~0.85+). Máy hoàn toàn tự tin nhưng sai bản chất ngữ nghĩa.
  - *Điểm thấp mà đúng:* Ở `img_011.jpg` và `img_002.jpg`, YOLOE tìm ra `three_wheeler` ở mức conf chỉ khoảng 0.18 - 0.25 (khá thấp), nhưng đối chiếu ảnh thật thì đúng là xe tuk-tuk/ba gác. Nếu để ngưỡng cao thì vật này sẽ bị bỏ sót hoàn toàn.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Vì việc chọn ngưỡng tối ưu và việc chấm điểm lại được thực hiện trên cùng một tập dữ liệu 4 ảnh gold (in-sample). Hơn nữa cỡ mẫu 4 ảnh là quá nhỏ, nhiều lớp không có hoặc có rất ít cá thể (bus=0, truck=0, bicycle=7), nên bộ ngưỡng này có thể bị overfit với 4 ảnh gold; khi áp dụng vào 24 ảnh batch thực tế với nhiều bối cảnh khác nhau, độ phủ và độ chính xác thực tế sẽ không cao bằng con số đo trên gold.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
