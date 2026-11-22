# BÁO CÁO KIỂM ĐỊNH DÀN TRANG VÀ BỐ CỤC PDF (V7)
*(PDF Layout and Typographic QA Report)*

Tài liệu này ghi nhận kết quả kiểm tra tự động và trực quan về cấu trúc dàn trang của tài liệu `DepSev_ML_Scientific_Report_V7.pdf`.

---

## 1. THÔNG SỐ TRANG VÀ CẤU TRÚC TỔNG THỂ

- **Tổng số trang PDF**: **89 trang**.
  - Trang 1--3: Trang bìa (Tiếng Việt, Tiếng Anh, Bìa phụ).
  - Trang 4--7: Lời cam đoan, Lời cảm ơn, Tóm tắt đề tài (Tiếng Việt và Tiếng Anh).
  - Trang 8--17: Mục lục, Danh sách hình vẽ, Danh sách bảng, Danh mục từ viết tắt.
  - Trang 18--23: Chương 1 – Mở đầu (Bối cảnh, Mục tiêu, Phát biểu bài toán, Đóng góp, Bố cục).
  - Trang 24--36: Chương 2 – Cơ sở lý thuyết và Thuật toán (Tập trung Random Forest, Decision Tree, Bagging).
  - Trang 37--51: Chương 3 – Phương pháp nghiên cứu và Thiết kế thực nghiệm (Dữ liệu NHANES, Tiền xử lý, Module hóa).
  - Trang 52--77: Chương 4 – Kết quả thực nghiệm và Thảo luận ($T_5$, $T_3$, Đánh đổi chi phí Bayes, Kiểm định ngoại suy).
  - Trang 78--80: Chương 5 – Kết luận và Hướng phát triển (Ngắn gọn, súc tích trong 3 trang).
  - Trang 81--82: Tài liệu tham khảo (Đã bổ sung 4 trích dẫn thực tế).
  - Trang 83--89: Phụ lục A, B, C, D (Đã sửa dứt điểm lỗi float của Bảng C.1).

---

## 2. KẾT QUẢ KIỂM TRA LỀ VÀ TRÀN CHỮ (OVERFLOW CHECK)

Quy cách định dạng: Khổ giấy A4 (rộng 595.28 pt, cao 841.89 pt).  
Lề trái: 30 mm (85.04 pt). Lề phải: 20 mm (56.69 pt).  
Vùng nội dung có thể in: chiều rộng tối đa $595.28 - 56.69 = 538.59$ pt.

- **Kiểm tra lề bằng tập lệnh PyMuPDF (`fitz`)**:
  - Tiêu chuẩn nghiêm ngặt: $x_1 \le 539.0$ pt.
  - Số lượng khối văn bản vượt lề phải ($x_1 > 539.0$ pt) trên tất cả các trang nội dung: **0 khối**.
  - **Kết luận**: Hoàn toàn không có hiện tượng tràn chữ lề phải (Zero Margin Overflow).

---

## 3. KIỂM ĐỊNH LIÊN KẾT VÀ THAM CHIẾU GÃY (BROKEN REFS)

- Tập lệnh tự động quét chuỗi ký tự `"??"` trên toàn bộ 89 trang PDF.
- Kết quả: **0 vị trí phát hiện tham chiếu gãy** (Broken references = `[]`).
- Toàn bộ liên kết chéo (`\ref`, `\cite`, `\label`) và liên kết mục lục (`hyperref`) hoạt động chính xác.

---

## 4. KIỂM ĐỊNH HIỆN TƯỢNG FLOAT NỔI TRÊN TIÊU ĐỀ PHỤ LỤC C

- **Hiện tượng ở V6**: Bảng C.1 bị trôi lên đỉnh trang 85, nằm phía trên tiêu đề chương `Phụ lục C. Kết quả chi tiết theo Fold của Nested Cross-Validation`.
- **Nguyên nhân cốt lõi**: Lớp tài liệu `uetgraduation.cls` khi khởi tạo lệnh `\chapter` thiếu cơ chế vô hiệu hóa float đỉnh trang (`\suppressfloats[t]`). Khi gặp môi trường bảng floating có chỉ định `t`, bộ dàn trang TeX đã tự động đẩy bảng lên vị trí `t` của trang mới trước tiêu đề chương.
- **Biện pháp khắc phục trong V7**:
  1. Thêm `\suppressfloats[t]` trực tiếp vào định nghĩa lệnh `\chapter` trong `uetgraduation.cls`.
  2. Bổ sung `\FloatBarrier` và `\suppressfloats[t]` ngay sau `\chapter{...}` trong `chapters/appendix.tex`.
- **Kết quả trực quan (Trang 85)**:
  - Vị trí trên cùng của trang 85 là tiêu đề: **"Phụ lục C. Kết quả chi tiết theo Fold của Nested Cross-Validation"**.
  - Tiếp theo là câu dẫn dắt văn bản.
  - Sau đó mới đến **Bảng C.1**.
  - Lỗi float đã được giải quyết triệt để và hoàn mỹ.

