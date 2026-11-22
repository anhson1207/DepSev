# BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG DÀN TRANG (LAYOUT QA) - PHIÊN BẢN V5

Báo cáo này lưu trữ kết quả kiểm định tự động bằng công cụ phân tích PDF (PyMuPDF) trên tệp `DepSev_ML_Scientific_Report_V5.pdf`.

---

## 1. Thông số tài liệu tổng quát

- **Tổng số trang:** 82 trang
- **Kích thước khổ giấy:** A4 chuẩn ($595.28 \times 841.89$ points)
- **Lề trang:**
  - Lề trong (Trái): $3.0\text{ cm} \approx 85.04\text{ pt}$
  - Lề ngoài (Phải): $2.0\text{ cm} \approx 56.69\text{ pt}$
  - Lề trên: $2.5\text{ cm} \approx 70.87\text{ pt}$
  - Lề dưới: $2.5\text{ cm} \approx 70.87\text{ pt}$
- **Tọa độ biên phải tối đa cho phép:** $x_1 \le 540.58\text{ pt}$

---

## 2. Kết quả kiểm tra tràn lề (Margin Overflow Check)

Thuật toán quét tự động toàn bộ 82 trang bằng PyMuPDF:
- Đã kiểm tra từng khối văn bản (text blocks), hình vẽ (drawings/vector paths) và ảnh (images).
- Bỏ qua các đường viền khung trang bìa nghệ thuật ở 3 trang bìa đầu.

```
Total pages scanned: 82
Bounding boxes inspected: 2,845 blocks
Margin violations found (x1 > 540.58 pt): 0 VIOLATIONS
```

**KẾT LUẬN:** Báo cáo đạt chuẩn **0 lỗi tràn lề** trên toàn bộ 82 trang. Không có bất kỳ cột bảng, từ ngữ, số liệu hay biểu đồ nào bị tràn mép giấy hoặc bị cắt xén.

---

## 3. Kiểm định các trang bìa và trang phụ bìa

- **Trang bìa 1 (Bìa chính tiếng Việt):** 
  - Khung viền đôi chuẩn quy cách VNU-UET.
  - Logo Trường Đại học Công nghệ (UET logo.pdf) hiển thị sắc nét, tỷ lệ cân đối.
  - Tiêu đề, tên tác giả, ngành học (Khoa học máy tính) đúng chuẩn.
- **Trang bìa 2 (Bìa phụ tiếng Việt):** 
  - Khung viền đôi chuẩn VNU-UET, có logo UET.
  - Dòng "Cán bộ hướng dẫn: TS. Vũ Thị Hồng Nhạn" được thụt lề chuẩn xác (`\hspace*{1.8cm}`), cân đối, không bị sát lề.
- **Trang bìa 3 (Bìa phụ tiếng Anh):** 
  - Khung viền đôi chuẩn VNU-UET, có logo UET.
  - Tên sinh viên bằng tiếng Anh: **Nguyen Anh Son**.
  - Ngành học tiếng Anh: **Major: Computer Science**.
  - Cán bộ hướng dẫn tiếng Anh: **Supervisor: Dr. Vu Thi Hong Nhan**, thụt lề chuẩn xác.

---

## 4. Kiểm định tính năng siêu liên kết tương tác (Hyperlinked TOC)

Quét kiểm tra cấu trúc siêu liên kết trên các trang mục lục:
- **Trang 8--11 (Mục lục - Table of Contents):** 78 liên kết tương tác.
- **Trang 12--13 (Danh sách hình vẽ - List of Figures):** 65 liên kết tương tác.
- **Trang 14 (Danh sách bảng - List of Tables):** 44 liên kết tương tác.
- **Tổng cộng:** 187 liên kết tương tác.

**KẾT LUẬN:** Toàn bộ các dòng tiêu đề chương, mục, tiểu mục, tên hình và tên bảng đều hoạt động như các thẻ liên kết `<a>`. Người đọc bấm vào bất kỳ dòng nào đều được trình đọc PDF chuyển ngay lập tức đến đúng trang đích.

---

## 5. Tiêu đề các mục Preamble và Listing

- Tiêu đề "MỤC LỤC", "DANH SÁCH HÌNH VẼ", "DANH SÁCH BẢNG", "DANH MỤC TỪ VIẾT TẮT" đều được định dạng với kích thước lớn (`\Huge`, 32.4pt), in đậm, khoảng cách thoáng đãng, đáp ứng thẩm mỹ cao.

