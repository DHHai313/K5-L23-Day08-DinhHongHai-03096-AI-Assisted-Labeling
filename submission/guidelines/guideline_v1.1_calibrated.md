# Label contract — Phố châu Á — BỔ SUNG v1.1 — 03096 - Đinh Hồng Hải

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 và thêm quy tắc mới rút ra từ lỗi của máy / bẫy lọt. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Nếu có bàn đạp cơ và dáng thanh mảnh xếp vào `bicycle`; nếu có sàn để chân rộng, khung đúc to và không có bàn đạp thì xếp vào `motorbike`. | `img_004.jpg` | CAL-01 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_013.jpg` | Xe tuk-tuk chở khách có người lái bên trong và mái che | YOLO26 gọi là `car`, SAM mở rộng mask phủ kín cả xe lẫn người lái | Gán xe là `three_wheeler` (ôm cả xe + mái che), vẽ riêng box `person` cho người lái nếu thấy $\ge 20\%$. Khi dùng SAM phải click điểm âm loại người lái. | Bổ sung mục 2.7 và 4.2.4 |
| `img_005.jpg` | Cụm xe máy đông đúc ở hậu cảnh xa và lỗi box hở nền | Bẫy `loose` nới rộng viền hở nền, máy vẽ nhiều box dính chùm | Cụm xe nhỏ cao $< 20\text{ px}$ ở xa gom thành một vùng `ignore_region`; xe đơn lẻ ở gần phải kéo 4 cạnh khít viền sai số $\le 2\text{ px}$. | Bổ sung mục 4.1.6 |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | — |
| v1.1 | Chốt phân loại xe đạp điện/xe máy điện; làm rõ tách người lái khỏi `three_wheeler`; quy chuẩn vùng `ignore_region` cho cụm xe xa để chống box lỏng. | Cập nhật từ CAL-01, CAL-02, CAL-03 |
