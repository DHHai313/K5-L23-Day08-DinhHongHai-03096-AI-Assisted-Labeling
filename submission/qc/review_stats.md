# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 444; đối tượng sau duyệt: 419; ngưỡng accepted: IoU >= 0.9
- Đã **loại 20 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 405 | 405/444 = 91.2% | 444 box máy |
| edited (sửa) | 7 | 7/444 = 1.6% | 444 box máy |
| deleted (xoá) | 32 | 32/444 = 7.2% | 444 box máy |
| added (vẽ thêm) | 7 | 7/419 = 1.7% | 419 đối tượng sau duyệt |

Trong số deleted + added, **1 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 6.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 227 | 1 | 1 | 1 |
| bicycle | 2 | 2 | 1 | 0 |
| motorbike | 123 | 0 | 0 | 2 |
| car | 37 | 0 | 0 | 0 |
| bus | 2 | 0 | 1 | 1 |
| truck | 1 | 0 | 3 | 0 |
| three_wheeler | 13 | 4 | 26 | 3 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 56 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 43 | 21 | 13 | 67.2% (KTC 95%: 55–77%) | 76.8% (KTC 95%: 64–86%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 1/11 = 9.1% (KTC 95%: 2–38%) (ngưỡng đạt: 80%). Chỉ 11 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - loose: 0/3
  - missing: 0/3
  - extra: 1/3
  - wrong_class: 0/2
  - lọt: img_031.jpg (loose), img_021.jpg (missing), img_005.jpg (loose), img_010.jpg (extra), img_013.jpg (wrong_class), img_034.jpg (extra), img_026.jpg (missing), img_035.jpg (wrong_class), img_017.jpg (missing), img_028.jpg (loose)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
