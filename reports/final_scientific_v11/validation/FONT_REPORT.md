# BÁO CÁO KIỂM TRA FONT CHỮ VÀ TYPOGRAPHY PDF (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/validation/FONT_REPORT.md`  
**Tệp kiểm tra:** `reports/final_scientific_v10/DepSev_ML_Scientific_Report_V10.pdf`  
**Công cụ trích xuất:** PyMuPDF (`fitz`) v1.25+ trên nền Python 3.13  
**Ngày kiểm định:** 31/10/2026  
**Kết quả tổng thể:** **ĐẠT TIÊU CHUẨN XUẤT BẢN KHOA HỌC (100% PASS)**  

---

## 1. TỔNG HỢP DANH MỤC FONT NHÚNG TRONG FILE PDF HOÀN THIỆN

Phân tích trích xuất trên toàn bộ 92 trang của tài liệu `DepSev_ML_Scientific_Report_V10.pdf` ghi nhận **20 tập con font (font subsets)** được nhúng hợp lệ:

| STT | Mã định danh trong PDF | Tên họ font chuẩn | Kiểu dáng | Kích cỡ điển hình | Mục đích sử dụng chính |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 1 | `BZIVMZ+TimesNewRomanPSMT` | Times New Roman | Regular | 12 pt | Thân bài toàn bộ các chương, tiểu mục, bảng biểu, phụ lục |
| 2 | `IHAIZV+TimesNewRomanPS-BoldMT` | Times New Roman | Bold | 12--16 pt | Tiêu đề chương, đề mục lớn, tiêu đề bảng/hình, điểm nhấn |
| 3 | `IXFAWU+TimesNewRomanPS-ItalicMT` | Times New Roman | Italic | 11--12 pt | Thuật ngữ gốc tiếng nước ngoài, tên sách báo, ghi chú |
| 4 | `TQTHRL+TimesNewRomanPS-BoldItalicMT` | Times New Roman | Bold-Italic | 12 pt | Nhấn mạnh chuyên sâu trong các định nghĩa quan trọng |
| 5 | `YEUUOC+CMMI12` | Computer Modern Math Italic | Italic | 12 pt | Ký hiệu biến toán học trong thân bài ($x, y, w, \theta, p$) |
| 6 | `FKPJSN+CMR9` | Computer Modern Roman | Regular | 9 pt | Chỉ số dưới, chỉ số trên trong công thức toán học |
| 7 | `VDVGTR+CMSY10` | Computer Modern Math Symbols | Regular | 10 pt | Các toán tử tập hợp và quan hệ ($\in, \times, \pm, \cap, \le, \ge, \to$) |
| 8 | `CAFSZL+CMR12` | Computer Modern Roman | Regular | 12 pt | Hằng số và dấu ngoặc trong các khối phương trình toán lớn |
| 9 | `PVTTLK+CMMI9` | Computer Modern Math Italic | Italic | 9 pt | Biến toán học kích cỡ 9pt trong bảng biểu và chú thích |
| 10 | `AKHCZK+CMR8` | Computer Modern Roman | Regular | 8 pt | Số đánh dấu cước chú (footnote markers) |
| 11 | `FZDNHS+CMR7` | Computer Modern Roman | Regular | 7 pt | Chỉ số toán học cấp 2 (sub-subscripts) |
| 12 | `DZGKAF+CMMI10` | Computer Modern Math Italic | Italic | 10 pt | Biến toán học trong bảng ma trận và tiêu đề cột |
| 13 | `KPXLOM+CMSY9` | Computer Modern Math Symbols | Regular | 9 pt | Dấu cộng trừ ($\pm$) và quan hệ trong các ô bảng |
| 14 | `GJMHKB+CMR10` | Computer Modern Roman | Regular | 10 pt | Số liệu thống kê trong các bảng ma trận |
| 15 | `NOMARO+CMEX10` | Computer Modern Math Extension | Regular | 10 pt | Dấu tổng lớn ($\sum$), dấu tích lớn ($\prod$), ngoặc ma trận |
| 16 | `UTACNV+CMMI7` | Computer Modern Math Italic | Italic | 7 pt | Ký hiệu biến nhỏ trong số mũ và chỉ số phức |
| 17 | `DDIEWW+CMMI8` | Computer Modern Math Italic | Italic | 8 pt | Biến toán học trong các ghi chú chân trang |
| 18 | `JAGKKY+MSBM10` | AMS Blackboard Bold | Regular | 10 pt | Tập số thực $\mathbb{R}$, không gian mẫu, kỳ vọng $\mathbb{E}$ |
| 19 | `NITCSQ+CMR6` | Computer Modern Roman | Regular | 6 pt | Chỉ số cấp ba trong khai triển chuỗi xác suất |
| 20 | `UBBQTN+CMMI6` | Computer Modern Math Italic | Italic | 6 pt | Ký hiệu ma trận phụ nhỏ |

---

## 2. KẾT QUẢ KIỂM TRA TRIỆT TIÊU FONT MONOSPACE (MÁY CHỮ)

- **Mục tiêu:** Đảm bảo không còn bất kỳ ký tự nào mang font máy chữ thô cứng (`LMMono`, `Courier`, `cmtt`) trong tài liệu.
- **Phương pháp quét:** Tìm kiếm tất cả các font chứa từ khóa `mono`, `courier`, `ttfamily`, `typewriter` trong siêu dữ liệu font của 92 trang.
- **Kết quả:** **0 font monospace được phát hiện.**
- **Đánh giá thẩm mỹ:** Các thông số kỹ thuật (`n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `class_weight`) hiển thị đồng nhất với kiểu chữ Times New Roman của tài liệu, tạo cảm giác một bài báo nghiên cứu hoàn thiện và chuyên nghiệp.

---

## 3. KIỂM ĐỊNH TYPOGRAPHY TRONG TẤT CẢ CÁC HÌNH ẢNH

- Toàn bộ 19 hình ảnh trong `reports/final_scientific_v10/source/figures/` đều được kết xuất từ mã nguồn Python với bộ cấu hình đồ họa Matplotlib STIX + Times New Roman.
- Các biểu thức toán học trong hình (như $\cup$, $\ge$, $\mu$, $\sigma$) sử dụng bộ phông STIX tương thích hoàn toàn về độ dày nét và khoảng cách với font Times New Roman trong văn bản xung quanh.
- Không có hiện tượng chữ bị vỡ hạt, răng cưa hoặc mất dấu khi phóng to $400\%$ trên màn hình độ phân giải cao.

