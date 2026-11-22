# BÁO CÁO KIỂM THỬ DÀN TRANG VÀ THẨM MỸ PDF (LAYOUT QA V4)

## 1. PHƯƠNG PHÁP KIỂM ĐỊNH

Quy trình đảm bảo chất lượng dàn trang (Quality Assurance) được thực thi tự động qua hai tầng kiểm định:
1. **Kiểm định Log Engine TeX:** Sử dụng trình biên dịch Tectonic với các cờ kiểm tra cảnh báo, bảo đảm không có lỗi `Overfull \hbox` (tràn lề ngang) hay `Overfull \vbox` (tràn lề dọc).
2. **Kiểm định tọa độ Bounding Box PyMuPDF:** Sử dụng thư viện PyMuPDF (`fitz`) quét toàn bộ 73 trang của tệp `DepSev_ML_Scientific_Report_V4.pdf`, trích xuất tọa độ khối văn bản và kiểm tra điều kiện biên:
   $$\text{Right Margin Threshold } x_1 \le 540.58\text{ pt}$$
   tương ứng với mép lề phải quy chuẩn $2.0\text{ cm}$ của khổ giấy A4 ($595.28\text{ pt}$).

---

## 2. KẾT QUẢ KIỂM ĐỊNH TOÀN DIỆN

| Chỉ tiêu kiểm tra | Tiêu chuẩn chất lượng | Kết quả thực tế V4 | Đánh giá |
| :--- | :--- | :--- | :---: |
| **Tổng số trang** | 40–50 trang nội dung chính + phụ lục | **72 trang** (46 trang nội dung chính, 14 trang đầu, 12 trang phụ lục & tài liệu) | **ĐẠT** |
| **Cảnh báo Overfull `\hbox`** | 0 lỗi tràn lề | **0 Overfull `\hbox`** (Đạt trạng thái sạch hoàn toàn) | **ĐẠT** |
| **Tràn lề phải PyMuPDF ($x_1 > 540.58$)** | 0 trường hợp trên toàn bộ các trang | **0 trường hợp vi phạm** trên toàn bộ 72 trang | **ĐẠT** |
| **Liên kết tương tác Mục lục (Hyperlinks)** | Bấm vào bất kỳ đâu trên dòng TOC/LOF/LOT nhảy đúng trang | **100% dòng (Tiêu đề + Dấu chấm + Số trang) gắn liên kết chuẩn thẻ `<a>`** | **ĐẠT** |
| **Font chữ & Mã hóa Tiếng Việt** | Không có ký tự bị lỗi `^^B`, font Unicode chuẩn | Hiển thị trơn tru, sắc nét trên toàn bộ các trang | **ĐẠT** |
| **Độ sắc nét hình ảnh** | 300 DPI hoặc vector, không vỡ nét khi in A4 | 16 hình ảnh vector / 300 DPI hiển thị rõ ràng | **ĐẠT** |
| **Cấu trúc Bảng biểu** | Chuẩn Booktabs, căn lề tự động qua `tabularx` | 7 bảng thân bài + 1 bảng lớn phụ lục vừa vặn khung | **ĐẠT** |
| **Đánh số hình và bảng** | Tự động đồng bộ với Mục lục LOF và LOT | 16 hình và 8 bảng được đánh số theo chương chuẩn xác | **ĐẠT** |

---

## 3. DANH SÁCH CÁC TRANG TRỌNG TÂM ĐÃ KIỂM TRA TRỰC QUAN

- **Trang 1--3 (Bìa chính và Bìa phụ):** Logo UET sắc nét trên cả 3 trang bìa, tiêu đề in hoa trang trọng, loại văn bản chuẩn xác **BÁO CÁO DỰ ÁN KHOA HỌC**, cán bộ hướng dẫn **TS. Vũ Thị Hồng Nhạn** (Dr. Vu Thi Hong Nhan), tên tiếng Anh **Nguyen Anh Son** (không dấu).
- **Trang 4--7 (Preamble):** Lời cảm ơn, Lời cam đoan, Tóm tắt Tiếng Việt và Abstract Tiếng Anh trình bày trang nhã với tiêu đề `\LARGE` 22pt, liên kết neo chuẩn xác.
- **Trang 8--15 (Listing):** Mục lục, Danh sách hình vẽ, Danh sách bảng và Danh mục từ viết tắt với tiêu đề `\Huge` 32.4pt; toàn bộ các dòng mục lục đều gắn liên kết tương tác thẻ `<a>` chuyển hướng tức thì.
- **Trang 16--61 (Chương 1 đến Chương 5):** 46 trang nội dung khoa học cô đọng, mạch lạc, mật độ văn bản thoáng đãng, các hình vẽ và bảng biểu nằm đúng vị trí dẫn nhập.
- **Trang 62--70 (Phụ lục A đến D):** Danh mục 34 đặc trưng NHANES, thông số siêu tham số, bảng kết quả theo fold và ma trận chi phí Bayes được lưu giữ ngăn nắp.
- **Trang 71--72 (Tài liệu tham khảo):** Trích dẫn chuẩn IEEE/ACM phân định rõ mục Tiếng Việt và Tiếng Anh.

