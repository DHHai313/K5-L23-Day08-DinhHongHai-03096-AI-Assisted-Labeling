# Thông tin học viên

- Họ và tên: Đinh Hồng Hải
- MSHV: 03096
- Lớp (L2 / L3 / khác): L23
- Tài khoản GitHub: DHHai313
- Repo bài nộp (URL): https://github.com/DHHai313/K5-L23-Day08-DinhHongHai-03096-AI-Assisted-Labeling
- Commit chốt bài (hash): b769452 (và commit hoàn thiện cuối cùng)
- Máy: Windows 11, GPU NVIDIA GeForce RTX 2050 (4GB) (chạy CPU cho prelabel)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 405/444 = 91.2% / 7/444 = 1.6% / 32/444 = 7.2% (mẫu số: 444 box máy); added: 7/419 = 1.7% (mẫu số: 419 đối tượng sau duyệt)
- So với gold: precision / recall của **máy**: 67.2% / 76.8% (mẫu số: 56 đối tượng gold)
- Bắt bẫy: 1/11 = 9.1%
- Thời gian: gán tay 5.0 phút/ảnh; duyệt có pre-label 1.5 phút/ảnh
- Lỗi QC: 40.0% theo ảnh (2/5), 2.2% theo đối tượng (2/89)
- **Quyết định:** Rework — lý do một dòng: Tỷ lệ bắt bẫy thấp (9.1% < 80%) và ảnh lỗi QC cao (40% > 2%), dính Automation Bias cần duyệt lại.
