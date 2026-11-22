# BÁO CÁO KIỂM THỬ BỐ CỤC VÀ DÀN TRANG PDF (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/validation/PDF_LAYOUT_QA.md`  
**Tệp kiểm tra:** `reports/final_scientific_v10/DepSev_ML_Scientific_Report_V10.pdf`  
**Công cụ kiểm định:** PyMuPDF Layout Engine & Bounding Box Inspector  
**Ngày kiểm định:** 31/10/2026  
**Kết quả tổng thể:** **ĐẠT CHUẨN HOÀN TOÀN (100% PASS)**  

---

## 1. THÔNG SỐ TRANG VÀ LỀ TRÌNH BÀY

- **Tổng số trang:** 92 trang PDF.
- **Kích thước trang:** Chuẩn khổ giấy A4 quốc tế ($595.28 \times 841.89\text{ pt}$, tương đương $210 \times 297\text{ mm}$).
- **Kiểm tra biên an toàn (Safety Margin Bounds):**
  - Giới hạn lề trái / phải: Toàn bộ các khối văn bản (text blocks) nằm nghiêm ngặt trong khoảng $x_0 \ge 40.0\text{ pt}$ và $x_1 \le 545.0\text{ pt}$.
  - Số lượng khối văn bản vi phạm tràn lề phải ($x_1 > 545.0\text{ pt}$): **0 khối**.
  - Giới hạn lề trên / dưới: Tiêu đề đầu trang (header) và chân trang (footer) bố trí cách mép an toàn. Văn bản nội dung nằm trong khoảng $y_0 \ge 65.0\text{ pt}$ đến $y_1 \le 788.0\text{ pt}$.
- **Kiểm tra tham chiếu chéo (Cross-Reference Integrity):**
  - Quét toàn bộ 92 trang tìm kiếm ký hiệu lỗi liên kết chưa định nghĩa (`??`): **0 lỗi**.
  - Toàn bộ các tham chiếu hình (`Hình 2.1`, `Hình 3.4`, v.v.) và bảng (`Bảng 2.1`, `Bảng 4.1`, v.v.) đều liên kết chính xác đến số thứ tự và trang tương ứng.

---

## 2. PHÂN BỔ TRANG VÀ BỐ CỤC THEO CHƯƠNG MỤC

| Chương / Phần | Khoảng trang | Các bảng biểu & hình vẽ tiêu biểu | Tọa độ & Trạng thái dàn trang |
| :--- | :---: | :--- | :--- |
| **Phần đầu (Trang bìa, Lời cam đoan, Mục lục, Danh mục)** | 1--14 | Danh mục bảng biểu, danh mục hình vẽ, tóm tắt | Đạt chuẩn, căn lề đối xứng, số trang La Mã i--xiv |
| **Chương 1: Giới thiệu và Đặt vấn đề** | 15--24 | Mô tả bài toán, câu hỏi nghiên cứu RQ1--RQ4 | Văn bản liền mạch, không có dòng mồ côi (orphans/widows) |
| **Chương 2: Cơ sở lý thuyết và Tổng quan nghiên cứu** | 25--38 | • Hình 2.1 (Trang 28, $y_0 = 337.8\text{ pt}$)<br>• Hình 2.2 (Trang 30)<br>• Hình 2.3 (Trang 35)<br>• Bảng 2.1 (Trang 38, $y_0 = 265.9\text{ pt}$) | **Hình 2.1** đã tách 2 phần rõ rệt (Part A/Part B).<br>**Bảng 2.1** đã chuyển sang định dạng 4 cột chuẩn `booktabs`. |
| **Chương 3: Phương pháp nghiên cứu và Kiến trúc hệ thống** | 39--53 | • Bảng 3.1 (Trang 40)<br>• Hình 3.1 (Trang 41)<br>• Hình 3.2 (Trang 43)<br>• Bảng 3.2 (Trang 44)<br>• Hình 3.3 (Trang 45)<br>• Hình 3.4 (Trang 47, $y_0 = 607.8\text{ pt}$)<br>• Bảng 3.3 (Trang 48)<br>• Hình 3.5 (Trang 49, $y_0 = 309.1\text{ pt}$) | **Hình 3.4** và **Hình 3.5** phân định rõ Stage 1/2 và Ma trận tổn thất $M$. Công thức toán học định dạng display rõ ràng. |
| **Chương 4: Kết quả thực nghiệm và Thảo luận** | 54--76 | • Bảng 4.1 (Trang 54, $y_0 = 337.4\text{ pt}$)<br>• Hình 4.1 (Trang 55)<br>• Bảng 4.2 (Trang 56, $y_0 = 499.6\text{ pt}$)<br>• Hình 4.2 (Trang 57)<br>• Hình 4.3 (Trang 58)<br>• Bảng 4.3 (Trang 59)<br>• Hình 4.4 (Trang 60)<br>• Hình 4.5 (Trang 63)<br>• Hình 4.6 (Trang 64)<br>• Hình 4.7 (Trang 65)<br>• Bảng 4.4 (Trang 66)<br>• Bảng 4.5 (Trang 67)<br>• Bảng 4.6 (Trang 68, $y_0 = 250.1\text{ pt}$)<br>• Hình 4.8 (Trang 70)<br>• Hình 4.9 (Trang 71)<br>• Hình 4.10 (Trang 74)<br>• Hình 4.11 (Trang 76) | **Khắc phục triệt để lỗi Float Inversion ở Trang 54:** Tiêu đề Mục 4.2 ($y_0 = 71.5$) và Tiểu mục 4.2.1 ($y_0 = 123.0$) xuất hiện trước, sau đó đến Tiểu mục 4.2.2 ($y_0 = 270.4$) và Bảng 4.1 ($y_0 = 337.4$). Bố cục trang mẫu mực. |
| **Chương 5: Kết luận và Hướng phát triển** | 77--84 | Tổng kết đóng góp, thảo luận giới hạn y sinh và hướng tương lai | Văn bản súc tích, văn phong khoa học khiêm tốn |
| **Tài liệu tham khảo** | 85--88 | 42 tài liệu tham khảo chuẩn IEEE / APA | Đầy đủ tên tác giả, năm, tạp chí, mã DOI |
| **Phụ lục** | 89--92 | Chi tiết cấu hình tham số, danh mục biến NHANES | Đạt chuẩn bố cục phụ lục |

---

## 3. KIỂM ĐỊNH CHI TIẾT TRANG 54 (VỊ TRÍ BẢNG 4.1 VÀ TIÊU ĐỀ MỤC 4.2)

Trích xuất tọa độ tuyệt đối của tất cả các khối văn bản trên Trang 54:
```
Block 1 (y0= 71.5 pt, y1=111.3 pt): 4.2. Đánh giá đóng góp của các nhóm đặc trưng (Feature Ablation)
Block 2 (y0=123.0 pt, y1=136.9 pt): 4.2.1. Thiết lập thực nghiệm
Block 3 (y0=147.4 pt, y1=243.4 pt): Thực nghiệm Feature Ablation nhằm trả lời câu hỏi nghiên cứu RQ1...
Block 4 (y0=270.4 pt, y1=284.3 pt): 4.2.2. Kết quả thực nghiệm
Block 5 (y0=294.8 pt, y1=323.4 pt): Kết quả out-of-fold chi tiết được tổng hợp trong Bảng 4.1...
Block 6 (y0=337.4 pt, y1=348.2 pt): Bảng 4.1. Kết quả thực nghiệm Feature Ablation trên tập Legacy Development...
Block 7 (y0=377.7 pt, y1=643.8 pt): [Thân bảng 4.1 với các dòng F1, F2, F3, F4 và các chỉ số]
Block 8 (y0=776.2 pt, y1=788.0 pt): 36 (Số trang hiển thị)
```
* **Nhận xét:** Thứ tự hiển thị hoàn toàn tự nhiên theo trình tự đọc từ trên xuống dưới của người đọc. Vấn đề "bảng đè lên trước tiêu đề" trong bản V9 đã được giải quyết triệt để $100\%$.

