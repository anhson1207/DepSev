# BÁO CÁO KIỂM TOÁN FONT CHỮ BÁO CÁO KHOA HỌC DEPSEV-ML V11
**Tập tin kiểm định:** `DepSev_ML_Scientific_Report_V11.pdf`  
**Ngày kiểm định:** 31/10/2026  
**Công cụ trích xuất:** PyMuPDF (`fitz` v1.25.x / Python 3.13)  
**Phạm vi:** Toàn bộ 91 trang PDF

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM TOÁN FONT

- **Tổng số trang:** 91 trang
- **Tổng số ký tự trích xuất:** 137.287 ký tự
- **Số ký tự font Monospace (LMMono, Courier, Typewriter, Consolas):** **0 ký tự (0,00%)**
- **Độ phủ font Times New Roman (Học thuật chuẩn):** **131.319 ký tự (95,65%)**
- **Ký tự ký hiệu toán học LaTeX (Computer Modern Math):** **5.968 ký tự (4,35%)**

---

## 2. BẢNG CHI TIẾT PHÂN BỐ FONT CHỮ

| Tên Font (PostScript Name) | Số ký tự trích xuất | Tỷ lệ (%) | Số trang xuất hiện | Nhóm chức năng |
| :--- | :---: | :---: | :---: | :--- |
| `TimesNewRomanPSMT` | 111.165 | 80,97% | 88 trang | Văn bản chính (Regular) |
| `TimesNewRomanPS-BoldMT` | 17.638 | 12,85% | 84 trang | Tiêu đề mục, nhãn bảng biểu (Bold) |
| `TimesNewRomanPS-ItalicMT` | 2.500 | 1,82% | 19 trang | Thuật ngữ, trích dẫn, chú thích (Italic) |
| `CMMI12` | 1.934 | 1,41% | 72 trang | Ký hiệu toán học cỡ 12pt (LaTeX Math Italic) |
| `CMR12` | 1.667 | 1,21% | 58 trang | Ký tự số trong công thức toán 12pt |
| `CMR9` | 864 | 0,63% | 65 trang | Chỉ số phụ dưới bảng biểu và đồ thị (9pt) |
| `CMSY10` | 761 | 0,55% | 61 trang | Ký hiệu toán học đặc biệt (LaTeX Math Symbol) |
| `CMMI9` | 192 | 0,14% | 30 trang | Ký hiệu toán học chỉ số trên/dưới cỡ 9pt |
| `CMMI10` | 167 | 0,12% | 11 trang | Ký hiệu toán học cỡ 10pt |
| `CMR7` | 146 | 0,11% | 13 trang | Chỉ số phụ nhỏ cỡ 7pt |
| `CMR8` | 93 | 0,07% | 25 trang | Chỉ số phụ cỡ 8pt |
| `CMR10` | 35 | 0,03% | 9 trang | Ký tự toán học cỡ 10pt |
| `CMMI7` | 32 | 0,02% | 7 trang | Ký hiệu toán học chỉ số cỡ 7pt |
| `CMSY9` | 26 | 0,02% | 10 trang | Ký hiệu toán học cỡ 9pt |
| `CMEX10` | 17 | 0,01% | 9 trang | Dấu ngoặc, dấu tích phân lớn (LaTeX Math Extension) |
| `TimesNewRomanPS-BoldItal` | 16 | 0,01% | 2 trang | Nhấn mạnh đặc biệt trong tiêu đề |
| `CMR6` | 15 | 0,01% | 3 trang | Chỉ số phụ siêu nhỏ cỡ 6pt |
| `CMMI8` | 11 | 0,01% | 5 trang | Ký hiệu toán học chỉ số cỡ 8pt |
| `MSBM10` | 7 | 0,01% | 6 trang | Ký hiệu tập hợp số thực Blackboard Bold ($\mathbb{R}$) |
| `CMMI6` | 1 | 0,00% | 1 trang | Ký hiệu toán học chỉ số cỡ 6pt |
| **Font Monospace (`LMMono*`)** | **0** | **0,00%** | **0 trang** | **ĐÃ TRIỆT TIÊU HOÀN TOÀN** |

---

## 3. CÁC BIỆN PHÁP ĐÃ TRIỂN KHAI ĐỂ ĐẠT 0% MONOSPACE

1. **Loại bỏ toàn bộ các lệnh `\texttt{...}` trong văn bản học thuật:**
   - Thay thế toàn bộ tên biến (như `RIDAGEYR`, `LBXTR`, `SLD010H`, `PAD680`, `PAQ605`) bằng font văn bản thông thường `RIDAGEYR` hoặc in nghiêng `\textit{...}`.
   - Thay thế đường dẫn tập tin (như `src/common31_features.py`) bằng `\textnormal{src/common31\_features.py}`.
2. **Loại bỏ môi trường `verbatim` và `listings` có font typewriter:**
   - Các bảng liệt kê tham số và thuật toán được định dạng bằng môi trường bảng học thuật `booktabs` (`tabularx`) tiêu chuẩn, sử dụng font Times New Roman.
3. **Đồng bộ hóa nhãn trục và chú giải trong toàn bộ đồ thị (Python matplotlib):**
   - File sinh hình ảnh `generate_perfect_figures.py` được cấu hình cố định `font.family: serif` và `font.serif: ['Times New Roman', 'Times', 'DejaVu Serif']`, đảm bảo tất cả 20 hình ảnh minh họa đều đồng nhất typography với văn bản chính.

---

## 4. KẾT LUẬN

Báo cáo DepSev-ML V11 hoàn toàn thỏa mãn tiêu chuẩn khắt khe về mặt typography học thuật:
- **100% tài liệu** sử dụng hệ font chuẩn mực (Times New Roman và hệ font toán học LaTeX Computer Modern tương thích).
- **Không còn bất kỳ ký tự nào mang font monospace**, loại bỏ hoàn toàn cảm giác "báo cáo kỹ thuật lập trình" hay "mã nguồn chưa được định dạng".

