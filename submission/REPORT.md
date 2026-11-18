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
| accepted | 405 | 91.2% | 444 box máy |
| edited | 7 | 1.6% | 444 box máy |
| deleted | 32 | 7.2% | 444 box máy |
| added | 7 | 1.7% | 419 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? Lớp tệ nhất là `three_wheeler` (xe ba bánh). Trong tổng số 52 box dự đoán của lớp này, có tới 26 box bị xóa (`deleted = 50.0%`), 4 box phải sửa lại biên và chỉ có 13 box được giữ nguyên (`accepted = 25.0%`). Nguyên nhân là YOLOE ở ngưỡng conf 0.15 dù giữ được recall nhưng sinh ra rất nhiều box ảo và box đè lên xe máy/người ở bên cạnh (thấy rõ ở `img_004.jpg` và `img_023.jpg`). Ngoài ra lớp `truck` cũng bị xóa 3/4 box do YOLO26 nhầm lẫn thùng hàng của xe ba gác thành xe tải.
- So với gold: precision / recall của **máy** Precision 67.2% (KTC 95%: 55–77%), Recall 76.8% (KTC 95%: 64–86%) (mẫu số: 56 đối tượng gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 1/11 = 9.1% (KTC 95%: 2–38%); loại nào lọt: Lọt 10/11 bẫy gồm toàn bộ 3 bẫy `loose` (hở biên viền ở `img_005.jpg`, `img_028.jpg`, `img_031.jpg`), toàn bộ 3 bẫy `missing` (máy cố tình xóa box vật thật ở `img_017.jpg`, `img_021.jpg`, `img_026.jpg`), 2 bẫy `extra` (`img_010.jpg`, `img_034.jpg`) và cả 2 bẫy `wrong_class` (`img_013.jpg`, `img_035.jpg`); vì sao (bạn đã bỏ bước nào của checklist)? Tôi đã duyệt quá nhanh (trung bình 1.5 phút/ảnh) và bỏ qua 2 bước cốt lõi của checklist: (1) Đã không ẩn box máy để quét mắt thường đếm người/xe trước, dẫn tới việc chỉ nhìn theo những gì máy đã vẽ sẵn và lọt sạch toàn bộ 3 bẫy `missing`; (2) Không phóng to (zoom in) để soi kỹ 4 cạnh viền tiếp xúc của từng box, khiến toàn bộ 3 bẫy `loose` bị bỏ lọt.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Tỷ lệ accepted rất cao (91.2%) đi cùng tỷ lệ bắt bẫy rất thấp (9.1%) là minh chứng điển hình của **Automation Bias (thiên kiến tự động hóa)**. Điều này phản ánh tâm lý người duyệt bị mô hình AI "ru ngủ": vì các khung bao nhìn qua trông rất tự nhiên và bao quát được hầu hết ảnh, mắt người duyệt có xu hướng tin tưởng hoàn toàn vào máy và vô thức bấm chấp nhận mà không thực hiện kiểm chứng độc lập.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 5.0 phút/ảnh (`gold_manual`: 20 phút / 4 ảnh) vs duyệt 1.5 phút/ảnh (`batch_review`: 36 phút / 24 ảnh). Thiên lệch (bạn gán gold trước nên đã quen cảnh): Tốc độ duyệt nhanh hơn gấp ~3.3 lần, tuy nhiên có sự thiên lệch lớn do người duyệt đã quen với đặc trưng đường phố châu Á từ tập gold và quan trọng hơn là tốc độ này đạt được do duyệt lướt qua loa (dẫn tới lọt 10/11 bẫy).
- Kết luận: **Có đáng dùng, nhưng chỉ đóng vai trò đề xuất thô cho các lớp phổ biến (person, motorbike, car) và bắt buộc phải có quy trình kiểm soát chặt chẽ**. Nếu không có checklist kiểm soát, việc dùng pre-label sẽ tạo ra cảm giác an tâm giả tạo và làm lọt lỗi nghiêm trọng vào bộ dữ liệu cuối.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. *Quy tắc ẩn box bắt buộc (15 giây đầu):* Mỗi khi mở một ảnh mới, việc đầu tiên là bấm ẩn toàn bộ lớp box máy trong 15 giây, dùng mắt thường quét từ trái qua phải để đếm số lượng người và xe trước khi bật box AI lên đối chiếu.
  2. *Quy tắc phóng to kiểm viền (Zoom in check):* Bắt buộc phải zoom tối thiểu 150% vào ít nhất 3 box bất kỳ trên mỗi ảnh để kiểm tra độ ôm sát của 4 cạnh viền, loại bỏ triệt để lỗi box lỏng (`loose`).
  3. *Quy tắc ngưỡng thời gian tối thiểu (Speed limit):* Đặt đồng hồ đếm ngược tối thiểu 2.5 phút cho mỗi ảnh batch; không cho phép hoàn thành và chuyển ảnh nếu chưa đủ thời gian rà soát, ngăn ngừa việc duyệt vội bấm Accept.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 base | 3 (2 pos, 1 neg) | Không gồm người lái (đã tách người lái nhờ 1 điểm âm trên ghế lái) | Đường biên mui xe và thân xe rất sắc nét, nhưng nan hoa bánh xe dính chút nền | `tuktuk_img013.jpg` |
| `img_004.jpg` | SAM 2.1 base | 3 (2 pos, 1 neg) | Không gồm người đạp xích lô (1 điểm âm trên lưng người đạp) | Mái che xích lô bám viền khá tốt, phần tay vịn khung kim loại hơi bị lem | `cyclo_img004.jpg` |
| `img_017.jpg` | SAM 2.1 base | 3 (2 pos, 1 neg) | Không gồm người đi bộ đằng sau | Khung xe ôm khít, nhưng kính chắn gió phía trước bị gộp luôn vào mask xe | `tricycle_img017.jpg` |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? Quy tắc 4.2.1 (bám đường biên phần nhìn thấy) và 4.2.3 (bóng đổ không thuộc vật) giúp định hình rõ vùng mask; tuy nhiên guideline còn thiếu quy chuẩn chi tiết về việc kính chắn gió nhìn xuyên thấu có được tính vào mask xe hay không (đã làm rõ ở v1.1).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: Đã thử nghiệm offline qua script `sam2_points.py`. Khác biệt lớn là offline cho phép kiểm soát chính xác tọa độ số pixel, lưu trữ mask tái lập 100%, trong khi Interactor trên giao diện CVAT trực quan nhưng phụ thuộc vào thao tác nhấp chuột trực tiếp trên canvas.

## 6. Calibration (Part 6)

- Số quyết định: 3 quyết định (`CAL-01`, `CAL-02`, `CAL-03` trong `calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): Quyết định quan trọng nhất là `CAL-02` (xác định việc phân loại đúng `three_wheeler` và bắt buộc tách riêng box/mask `person` cho người lái xe ba bánh thay vì để mô hình gộp chung).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: (1) Phân biệt xe đạp điện và xe máy điện nhỏ dựa trên cấu tạo bàn đạp cơ học; (2) Tách độc lập box người lái khỏi xe ba bánh; (3) Bắt buộc gom cụm xe máy ở hậu cảnh xa < 20px thành vùng `ignore_region` để tránh lỗi viền lỏng.
- Nguyên nhân gốc: Kết hợp giữa việc guideline v1.0 còn mở (chưa có quy định về xe điện) và lỗi giới hạn mô hình máy (YOLO26 thiếu nhãn địa phương, SAM tự động lan mask sang người lái).

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: 2/89 = 2.2% · theo **ảnh**: 2/5 = 40.0% (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại:
  1. `sai_vi_tri_kich_thuoc`: 1 lỗi (box xe máy bị rộng hở nền do dính bẫy loose ở `img_031.jpg`).
  2. `sai_lop`: 1 lỗi (xe tuk-tuk bị gán nhãn car do dính bẫy wrong_class ở `img_035.jpg`).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): Với 2 ảnh lỗi trên mẫu 5 ảnh, khoảng tin cậy Wilson 95% có cận trên lên tới 77%. Mẫu 5 ảnh quá nhỏ nên không thể khẳng định chắc chắn về mặt thống kê rằng tỷ lệ lỗi thực tế đạt chuẩn <= 2%.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): Script chỉ kiểm đếm cơ học trên phiếu ghi nhận của con người; script không tự phát hiện được box hở nền hay vật bị cả máy và người cùng bỏ sót nếu người kiểm không soi kỹ trên ảnh.

**Quyết định: REWORK**, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release): Đi theo cây quyết định: Guideline đã được bổ sung v1.1 rõ ràng (không cần Hold). Tuy nhiên, tỷ lệ ảnh lỗi qua kiểm định QC là 40.0% (vượt xa ngưỡng 2%), đặc biệt tỷ lệ bắt bẫy chỉ đạt 1/11 = 9.1% (thấp hơn nhiều so với ngưỡng tối thiểu 80%). Điều này chứng minh lô nhãn đã duyệt bị ảnh hưởng nặng nề bởi Automation Bias và chưa đạt độ tin cậy để đưa vào huấn luyện mô hình.
Nếu Rework / Hold, việc cần làm tiếp: Cần phân bổ lại lượt review cho lô 24 ảnh batch: người duyệt phải thực hiện nghiêm ngặt checklist 4 bước (bắt buộc ẩn box máy 15s đầu để quét mắt, zoom 150% kiểm tra viền khít và rà soát kỹ các xe ba bánh) trước khi cho phép xuất bản.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): Sử dụng Antigravity làm trợ lý giải thích khái niệm, đọc hiểu luồng lỗi và hướng dẫn các lệnh CLI; tự thực hiện kiểm tra chéo số liệu trên đĩa và chạy lệnh trên PowerShell.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? Không bị nhiễm, đã gán tay hoàn toàn 4 ảnh gold theo quy trình Part 1 trước khi chạy bất kỳ mô hình AI nào.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Windows 11, GPU NVIDIA GeForce RTX 2050 (4GB VRAM), chạy inference bằng CPU local; CVAT chạy qua Docker Desktop.
- Dùng cache `data/cache/` thay vì tự chạy? Không dùng cache, đã tự chạy toàn bộ các lệnh pre-label từ đầu.
- Lỗi / bug script đã gặp (nếu có): Gặp lỗi thiếu quyền ghi đĩa khi script chạy trong môi trường sandbox hạn chế, đã chuyển sang thực thi trực tiếp trên PowerShell cục bộ.
- Một điều bạn làm khác lần sau: Sẽ luôn tuân thủ quy tắc ẩn box máy trước khi nhìn ảnh để không bao giờ bị thiên kiến tự động hóa đánh lừa khi kiểm định dữ liệu.
