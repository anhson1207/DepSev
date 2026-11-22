# BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG DÀN TRANG VÀ THIẾT KẾ HỌC THUẬT (LAYOUT QA)
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Tài liệu xuất bản:** `reports/final_revision_v3/DepSev_ML_Final_Polished.pdf`  
**Phiên bản đối chiếu:** `reports/final_revision_v2/DepSev_ML_Final_Revised.pdf` (70 trang)  
**Ngày kiểm định:** 29/10/2026  
**Phương pháp kiểm định:** Phân tích tọa độ hình học Bounding Box tự động bằng PyMuPDF (`fitz`), đối chiếu trực quan ảnh phân giải cao ($\ge 150$ DPI), và kiểm tra danh mục TOC/LOT/LOF.

---

## 1. TIÊU CHUẨN THIẾT KẾ VÀ TỌA ĐỘ TRANG CHUẨN (A4)

- **Kích thước trang A4 vật lý:** $210 \times 297$ mm $\Leftrightarrow 595.28 \times 841.89$ points (pt).
- **Lề trái (Left Margin):** $3.0$ cm $= 85.04$ pt.
- **Lề phải (Right Margin):** $2.0$ cm $= 56.69$ pt.
- **Chiều rộng vùng in khả dụng (Printable Text Width):** $16.0$ cm $= 453.55$ pt.
- **Tọa độ biên giới hạn nội dung in:**
  - Hoành độ tối thiểu hợp lệ: $x_0 \ge 85.04$ pt.
  - Hoành độ tối đa hợp lệ: $x_1 \le 538.58$ pt.
  - Tràn mép giấy (cắt cụt khi in): $x_1 > 595.28$ pt.
  - Khung viền bìa trang trí (trang 1--3): theo chuẩn của VNU-UET (`uetgraduation.cls`), vẽ hoa văn viền cách mép giấy 15 mm ($x_1 = 552.76$ pt).

---

## 2. TỔNG HỢP KẾT QUẢ QUÉT TỰ ĐỘNG BẰNG PYMUPDF TRÊN 77 TRANG

Toàn bộ 77 trang của bản PDF cuối cùng `DepSev_ML_Final_Polished.pdf` đã được quét tọa độ chính xác:

| Hạng mục kiểm tra | Tiêu chuẩn giới hạn | Số trang vi phạm trên bản cũ (v2, 70 trang) | Số trang vi phạm trên bản mới (v3, 77 trang) | Trạng thái (v3) |
| :--- | :---: | :---: | :---: | :---: |
| **Văn bản vượt lề phải** (`text overflow`) | $x_1 \le 538.58$ pt | 10 trang (P33, P47, P50, P54, P55, P56, P58, P60, P61, P64) | **0 trang (0%)** | **PASS** |
| **Bảng/Khung vượt lề phải** (`drawing overflow`) | $x_1 \le 538.58$ pt | 9 trang (P35, P47, P50, P54, P55, P56, P58, P60, P61) | **0 trang (0%)** | **PASS** |
| **Hình vẽ vượt lề phải** (`image overflow`) | $x_1 \le 538.58$ pt | 4 trang (P35, P55, P56, P58) | **0 trang (0%)** | **PASS** |
| **Rơi ra ngoài mép giấy** (`clipped off-page`) | $x_1 > 595.28$ pt | 5 trang (Bảng 4.7 tràn +144.5 pt, Ma trận nhầm lẫn tràn +120.4 pt) | **0 trang (0%)** | **PASS** |
| **Văn bản lấn lề trái** (`left margin overflow`) | $x_0 \ge 85.04$ pt | 0 trang | **0 trang (0%)** | **PASS** |
| **Tràn mép trên / dưới** (`vertical overflow`) | $0 \le y \le 841.89$ pt | 0 trang | **0 trang (0%)** | **PASS** |
| **Tiêu đề mục cô độc** (`orphan heading`) | Không đứng cuối trang | 1 mục | **0 mục (0%)** | **PASS** |

---

## 3. BẢNG ĐỐI CHIẾU CHI TIẾT CÁC LỖI DÀN TRANG VÀ GIẢI PHÁP KHẮC PHỤC

| STT | Trang bản cũ (v2) | Vị trí / Đối tượng | Lỗi phát hiện trên bản cũ | Giải pháp kỹ thuật đã áp dụng | Trang bản mới (v3) | Kết quả kiểm định |
| :---: | :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | P11--P12 | Danh mục bảng (LOT) | Bảng 3.3 bị đánh số thành "Bảng 2" do xung đột counter giữa `longtable` và `uetgraduation.cls`. | Đồng bộ `\c@table` với `\c@a@table` và cấu hình lại `\fnum@table`, `\l@table` trong `main.tex`. Đổi tên thành Bảng 3.3. | P12 | **PASS** |
| 2 | P13--P14 | Danh mục từ viết tắt | Bảng từ viết tắt `longtable` bị tràn lề phải ($x_1 = 545.70$ pt, vượt +7.12 pt) do độ rộng 3 cột $2.5 + 6.5 + 6.0$ cm cộng `\tabcolsep` lớn hơn 16 cm. | Điều chỉnh lại độ rộng cột thành `p{2.0cm} p{6.4cm} p{6.4cm}` và giảm `\tabcolsep=4pt`, bảo đảm tổng bề rộng $\le 15.6$ cm. | P13--P14 | **PASS** |
| 3 | P15 | Phương trình (1.2) | Nhãn phương trình `Severe conv. / Nặng theo quy ước` quá dài, đẩy công thức tràn lề phải ($x_1 = 540.9$ pt). | Rút gọn nhãn toán học thành $\text{Sev}_{\text{conv}}$ và đưa chú thích đầy đủ xuống dòng văn bản thuyết minh. | P16 | **PASS** |
| 4 | P19 | Bảng 2.1 (Related Works) | Bảng sử dụng cột `p{...}` cố định thiếu linh hoạt, văn bản bị chật và chạm sát mép lề. | Chuyển đổi sang môi trường `tabularx{\textwidth}`, định nghĩa tỷ lệ độ rộng cột động với `RaggedRight`. | P20 | **PASS** |
| 5 | P21 | Hình 3.1 (Kiến trúc) | Hình 3.1 vẽ bằng `picture` bị vỡ, trôi ngược lên đầu Chương 3 trước cả tiêu đề chương. | Tạo ảnh vector/raster độ phân giải 300 DPI `depsev_overall_architecture.png`, đặt vị trí `[htbp]` ngay sau đoạn văn giới thiệu Mục 3.1. | P35 | **PASS** |
| 6 | P23 | Bảng 3.1 & 3.2 | Bảng phân bố lớp dùng `tabular{l c c c c c c c}`, các cột bị co cụm và một số ô số liệu bị đẩy sát lề. | Chuyển đổi sang `tabularx{\textwidth}` với các cột `>{\centering\arraybackslash}X`, căn lề đối xứng hoàn hảo. | P37 | **PASS** |
| 7 | P24--P26 | Bảng 3.3 (34 đặc trưng) | Bảng `longtable` thiếu dòng tiêu đề lặp lại khi sang trang mới (`\endhead`), người đọc không biết tên cột ở trang 2 và 3. | Bổ sung `\endhead` với `\textbf{Bảng 3.3 (tiếp theo)}`, chuẩn hóa độ rộng 4 cột và giảm font `\footnotesize`. | P38--P40 | **PASS** |
| 8 | P36 | Bảng 4.1 (Ablation) | 16 cấu hình bị dàn trải, bề rộng bảng vượt nhẹ lề trang in. | Chuẩn hóa `tabularx{\textwidth}` với định dạng Mean $\pm$ SD rõ nét cho cả Macro-F1 và BalAcc. | P50 | **PASS** |
| 9 | P40 | Bảng 4.2 (Granularity) | Các chỉ số phân loại 4 mức độ nhãn bị ép chặt. | Chuẩn hóa `tabularx{\textwidth}`, giãn dòng `\renewcommand{\arraystretch}{1.15}`. | P53 | **PASS** |
| 10 | P42 | Hình 4.4 (PCA biplot) | Hình cũ nhãn tiếng Anh bé, khó đọc, độ tương phản mờ nhạt. | Vẽ lại bằng Python độ phân giải 300 DPI, nhãn tiếng Việt rõ ràng, cỡ chữ 12--13pt, độ tương phản cao. | P56 | **PASS** |
| 11 | P44 | Bảng 4.3 (Temporal) | Cột kết quả bị tràn lề phải ở các cycle 2015--2016. | Chuyển đổi sang `tabularx{\textwidth}` với cột `X` co giãn tự động. | P58 | **PASS** |
| 12 | P47 | Bảng 4.4 (Hard Hierarchical) | Bảng tràn lề phải ($x_1 = 552.3$ pt). | Chuyển sang `tabularx{\textwidth}`, căn lề chuẩn mực. | P61 | **PASS** |
| 13 | P48 | Bảng 4.5 (Soft Hierarchical) | Tràn lề phải ($x_1 = 568.7$ pt, vượt +30.1 pt). | Chuyển sang `tabularx{\textwidth}`, độ rộng đúng 453.55 pt. | P62 | **PASS** |
| 14 | P50 | Bảng 4.6 (Ordinal) | Bảng tràn lề phải do quá nhiều cột số liệu không co giãn. | Chuyển sang `tabularx{\textwidth}` với 7 cột co giãn đồng đều. | P64 | **PASS** |
| 15 | P53 | Bảng 4.7 (Master Comparison) | Bảng ngang 13 cột dài khủng khiếp, bị tràn ra ngoài trang giấy tới **+144.5 pt** ($x_1 = 683.1$ pt, rớt hẳn ra ngoài giấy A4). | Tái cấu trúc thành bảng đứng 2 phần thống nhất (Phần A: Hiệu năng tổng thể; Phần B: Khả năng nhận diện lớp & An toàn lâm sàng), vừa khít trang đứng 453.55 pt. | P67 | **PASS** |
| 16 | P54 | Hình 4.7 & Hình 4.8 | Hai hình bị chồng lấn chú thích và chú thích Hình 4.7 ghi sai là "biểu đồ phân tán". | Sửa chú thích chính xác thành "Biểu đồ cột nhóm...", bố trí 2 hình trên cùng trang P68 với cỡ `0.78\textwidth` cân đối. | P68 | **PASS** |
| 17 | P55 | Hình 4.9 (Ma trận nhầm lẫn) | 7 ma trận nhầm lẫn nhồi nhét trong một hình duy nhất, bị tràn lề tới **+120.4 pt**, chữ số trong ma trận bé tí không đọc được. | Tách làm 2 hình riêng biệt: Hình 4.9 (Nhóm Flat, Hard, Soft - lưới 2x2) và Hình 4.10 (Nhóm Ordinal - lưới 1x3), cỡ số ma trận 14--15pt cực kỳ sắc nét. | P69 | **PASS** |

---

## 4. KẾT LUẬN KIỂM ĐỊNH

Bản PDF xuất bản `DepSev_ML_Final_Polished.pdf` đạt mức độ hoàn thiện xuất sắc theo toàn bộ các tiêu chuẩn dàn trang học thuật của Đại học Quốc gia Hà Nội (VNU-UET):
1. **100% không còn hiện tượng tràn biên hay cắt cụt nội dung.**
2. **Không có bất kỳ bảng biểu hoặc hình ảnh nào vượt ra khỏi lề trang in $16.0$ cm.**
3. **Mọi biểu đồ và ma trận nhầm lẫn có thể đọc rõ ràng bằng mắt thường khi in ra giấy A4.**
4. **Hệ thống đánh số chương, mục, bảng biểu và hình vẽ hoàn toàn nhất quán giữa văn bản và các trang danh mục đầu tài liệu.**

