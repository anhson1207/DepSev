# BÁO CÁO KIỂM TOÁN TÍNH ĐỒNG NHẤT FONT CHỮ VÀ TYPOGRAPHY (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/audit/FONT_CONSISTENCY_AUDIT.md`  
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Ngày thực hiện:** 31/10/2026  
**Người thực hiện:** Antigravity Research Team  

---

## 1. TIÊU CHUẨN TYPOGRAPHY ĐẶT RA CHO BẢN V10

Nhằm đạt chuẩn mực xuất bản học thuật cao nhất, báo cáo DepSev-ML V10 đặt ra các yêu cầu khắt khe về mặt kiểu chữ (typography):
1. **Font chữ chính thống nhất:** Toàn bộ thân bài, tiêu đề các cấp, nhãn bảng biểu, chú thích hình vẽ và phần tài liệu tham khảo phải sử dụng họ font **Times New Roman** (Serif).
2. **Ký hiệu toán học chuẩn mực:** Sử dụng bộ font toán học TeX tiêu chuẩn (Computer Modern Math / STIX / AMS Symbols), bảo đảm sự hài hòa về độ dày nét (stroke weight) và tỷ lệ chiều cao (x-height) với font Times New Roman.
3. **Loại bỏ hoàn toàn font Monospace (Font máy chữ):** Trong các bản thảo trước (V8, V9), nhiều trang văn bản xuất hiện việc chèn xen kẽ font `\texttt` (Latin Modern Monospace) khi gọi tên siêu tham số hoặc biến số. Điều này tạo ra các mảng chữ bị thưa, không cân đối và mang phong cách mã nguồn lập trình hơn là một báo cáo khoa học.
4. **Đồng nhất font chữ trong hình ảnh thực nghiệm:** Toàn bộ 19 hình vẽ đồ thị và sơ đồ phải được dựng bằng font Times New Roman với kích cỡ tối thiểu $\ge 9$ pt, hoàn toàn tương thích với font chữ của văn bản xung quanh.

---

## 2. KẾT QUẢ KIỂM TRA MÃ NGUỒN LATEX (`source/`)

Thực hiện lệnh quét toàn bộ các file `.tex` trong thư mục `reports/final_scientific_v10/source/`:
```bash
grep -rn '\\texttt\|\\ttfamily\|\\verb' reports/final_scientific_v10/source/
```
* **Kết quả:** **0 kết quả** (Không còn bất kỳ lệnh tạo font monospace nào tồn tại trong toàn bộ mã nguồn).

### Biện pháp thay thế đã thực hiện:
- Các tên siêu tham số như `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `class_weight` được đưa về font La-tinh thông thường với ký tự gạch dưới được thoát (`\_`):
  - Trước: `\texttt{n\_estimators} = 150, \texttt{max\_depth} = 8`
  - Sau: `n\_estimators = 150, max\_depth = 8` (hoặc in đậm `\textbf{n\_estimators} = 150` trong các bảng danh mục).
- Các định danh tệp dữ liệu hoặc nhãn biến được định dạng bằng văn bản chuẩn có giải thích ngữ cảnh rõ ràng.

---

## 3. KẾT QUẢ TRÍCH XUẤT VÀ PHÂN TÍCH FONT TỪ FILE PDF HOÀN THIỆN (`DepSev_ML_Scientific_Report_V10.pdf`)

Sử dụng thư viện phân tích PDF chuyên sâu (PyMuPDF) để duyệt toàn bộ 92 trang của tệp `DepSev_ML_Scientific_Report_V10.pdf`. Bảng dưới đây liệt kê đầy đủ tất cả các họ font nhúng (embedded font subsets) có mặt trong tài liệu:

| Tên font nhúng trong PDF | Họ font gốc | Vai trò và vị trí xuất hiện | Số trang xuất hiện / 92 trang |
| :--- | :--- | :--- | :---: |
| `BZIVMZ+TimesNewRomanPSMT` | Times New Roman Regular | Toàn bộ thân bài văn bản, danh mục, phụ lục | 89 |
| `IHAIZV+TimesNewRomanPS-BoldMT` | Times New Roman Bold | Tiêu đề chương, mục, tiểu mục, tiêu đề bảng, tên hình | 85 |
| `IXFAWU+TimesNewRomanPS-ItalicMT` | Times New Roman Italic | Cụm từ thuật ngữ tiếng nước ngoài, tên sách/báo, ghi chú | 19 |
| `TQTHRL+TimesNewRomanPS-BoldItalicMT`| Times New Roman Bold-Italic | Tiêu đề tiểu mục cấp sâu, từ khóa nhấn mạnh đặc biệt | 2 |
| `YEUUOC+CMMI12` | Computer Modern Math Italic 12pt | Biến số toán học ($x, y, w, \theta, \mathcal{C}, \mathcal{S}$) trong thân bài | 73 |
| `FKPJSN+CMR9` | Computer Modern Roman 9pt | Chỉ số dưới, số mũ, hằng số toán học 9pt | 67 |
| `VDVGTR+CMSY10` | Computer Modern Math Symbols 10pt | Dấu toán học ($\in, \times, \pm, \cap, \le, \ge, \to$) | 61 |
| `CAFSZL+CMR12` | Computer Modern Roman 12pt | Số nguyên và hằng số trong các khối công thức lớn | 54 |
| `PVTTLK+CMMI9` | Computer Modern Math Italic 9pt | Biến số toán học trong chú thích bảng và biểu thức phụ | 29 |
| `AKHCZK+CMR8` | Computer Modern Roman 8pt | Chỉ số chân trang (superscript footnote, sub-subscripts) | 25 |
| `FZDNHS+CMR7` | Computer Modern Roman 7pt | Chỉ số cấp hai trong các công thức phức tạp | 13 |
| `DZGKAF+CMMI10` | Computer Modern Math Italic 10pt | Biến số toán học trong bảng biểu | 11 |
| `KPXLOM+CMSY9` | Computer Modern Math Symbols 9pt | Ký hiệu toán học cỡ 9pt trong bảng | 10 |
| `GJMHKB+CMR10` | Computer Modern Roman 10pt | Số liệu toán học trong bảng biểu | 9 |
| `NOMARO+CMEX10` | Computer Modern Math Extension 10pt | Dấu tổng tích lớn ($\sum, \prod$), dấu ngoặc ma trận lớn | 9 |
| `UTACNV+CMMI7` | Computer Modern Math Italic 7pt | Biến số toán học cấp hai trong chỉ số dưới | 7 |
| `DDIEWW+CMMI8` | Computer Modern Math Italic 8pt | Biến số toán học trong chú thích chân trang | 6 |
| `JAGKKY+MSBM10` | AMS Blackboard Bold 10pt | Tập số thực $\mathbb{R}$, kỳ vọng toán $\mathbb{E}$ | 6 |
| `NITCSQ+CMR6` | Computer Modern Roman 6pt | Ký hiệu số mũ cực nhỏ trong phân phối xác suất | 3 |
| `UBBQTN+CMMI6` | Computer Modern Math Italic 6pt | Ký hiệu biến cực nhỏ trong chỉ số ma trận lồng | 1 |

### Kết luận kiểm toán font PDF:
- **Tỷ lệ hiện diện của font Monospace:** **Đúng 0%** (Không có bất kỳ sự xuất hiện nào của `LMMono`, `Courier`, `cmtt` hay bất kỳ font monospace nào).
- **Tính đồng nhất của văn bản:** $100\%$ các đoạn văn bản tiếng Việt và tiếng Anh đều sử dụng 4 biến thể chính thống của Times New Roman (Regular, Bold, Italic, Bold-Italic).
- **Tính tương thích toán học:** $100\%$ các biểu thức toán học sử dụng bộ font chuẩn TeX / AMS, bảo đảm hiển thị sắc nét ở mọi độ phân giải khi in ấn.

---

## 4. ĐỒNG NHẤT HÓA TYPOGRAPHY TRONG TẤT CẢ 19 HÌNH ẢNH THỰC NGHIỆM

File sinh hình ảnh tự động [generate_perfect_figures.py](file:///Users/nguyenanhson/DepSev-ML/reports/final_scientific_v10/source/generate_perfect_figures.py) đã được cấu hình toàn cục:
```python
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13
```
- Sử dụng bộ giải mã `stix` cho các biểu thức toán học nhúng trong đồ thị (ví dụ $\cup, \ge, \le, \mu, \sigma$), bảo đảm nét vẽ của ký hiệu toán học trong hình hoàn toàn ăn khớp với nét chữ Times New Roman của tài liệu.
- Đã khắc phục triệt để cảnh báo thiếu ký tự Unicode `Glyph 8746 (\N{UNION})` bằng cách sử dụng biểu thức toán học LaTeX chuẩn `r"$\cup$"`.
- Độ phân giải của toàn bộ 19 tệp hình ảnh đầu ra đạt chuẩn xuất bản $300\text{ DPI}$.

