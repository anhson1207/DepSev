# BÁO CÁO KHOA HỌC DỰ ÁN DEPSEV-ML (PHIÊN BẢN HOÀN THIỆN V10)
**DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data**  
*Tác giả nghiên cứu:* Nhóm nghiên cứu DepSev-ML  
*Đơn vị:* Khoa Công nghệ Thông tin - Trường Đại học Công nghệ (VNU-UET)  
*Thời gian hoàn thành:* Tháng 10/2026  

---

## 1. GIỚI THIỆU TỔNG QUAN

Thư mục `reports/final_scientific_v10/` chứa toàn bộ tài liệu nghiên cứu, mã nguồn LaTeX, các tệp đồ họa khoa học và hồ sơ kiểm định chất lượng học thuật của **Bản báo cáo khoa học DepSev-ML Phiên bản 10 (V10)**.

Phiên bản V10 là kết quả của quá trình rà soát, tái cấu trúc và hoàn thiện toàn diện từ bản V9 (91 trang) sang V10 (92 trang), giải quyết triệt để các hạn chế về:
1. **Tính đúng đắn khoa học và nhất quán số liệu:** Hòa giải toàn diện sự khác biệt giữa mô hình Random Forest trong Nested Cross-Validation ($0.2755$) và cấu hình Frozen Spec ($0.2741$); phân tích mức độ đóng góp của các nhóm đặc trưng theo từng họ mô hình riêng biệt; đối chiếu chính xác số ca nặng bị bỏ sót ($62$ ca trong $E_{6b}$, giảm $48.33\%$ so với $E_1$).
2. **Đồng nhất Typography:** Chuyển đổi $100\%$ sang font **Times New Roman** chuẩn mực học thuật, loại bỏ hoàn toàn font Monospace (`\texttt`) trong thân bài, chuẩn hóa ký hiệu toán học TeX/STIX trên cả văn bản và hình ảnh.
3. **Chất lượng hình ảnh và bảng biểu:** Tái thiết kế toàn bộ 19 hình ảnh khoa học ở độ phân giải $300\text{ DPI}$; thiết kế lại Hình 2.1 (loại bỏ chỉ dẫn y tế, tách Part A/Part B rõ ràng); chuẩn hóa Bảng 2.1 sang 4 cột `booktabs`; giải quyết triệt để lỗi trôi bảng trước tiêu đề (Float Inversion Bug) tại Bảng 4.1 trang 54.
4. **Văn phong khoa học:** Loại bỏ toàn bộ các cụm từ đao to búa lớn/AI puffery, định hình phong cách khiêm tốn, trung thực của sinh viên ngành Khoa học Máy tính; chuyển đổi mạch dẫn dắt sang hướng so sánh đa chiều các kiến trúc học máy trên dữ liệu y tế dạng bảng.

---

## 2. CẤU TRÚC THƯ MỤC BÀN GIAO

```
reports/final_scientific_v10/
├── DepSev_ML_Scientific_Report_V10.pdf   # Bản báo cáo PDF hoàn thiện (92 trang, không lỗi font, không liên kết hỏng)
├── README.md                            # Tài liệu tổng quan này
│
├── source/                              # Toàn bộ mã nguồn LaTeX và tài nguyên biên dịch
│   ├── main.tex                         # Tệp master LaTeX
│   ├── uetgraduation.cls               # Định nghĩa định dạng luận văn chuẩn UET
│   ├── chapters/                        # Các chương nội dung (chapter1 -> chapter5, appendix)
│   ├── tables/                          # Các bảng số liệu tách rời (table 2.1, 4.1 -> 4.6, v.v.)
│   ├── figures/                         # 19 tệp hình ảnh PNG 300 DPI đã hoàn thiện
│   ├── references.bib                   # Cơ sở dữ liệu tài liệu tham khảo BibTeX
│   └── generate_perfect_figures.py      # Kịch bản Python tự động sinh toàn bộ 19 hình ảnh
│
├── audit/                               # Hồ sơ kiểm định học thuật và số liệu chuyên sâu
│   ├── V9_COMPREHENSIVE_REVIEW.md       # Đánh giá toàn diện 15 lỗi của bản V9
│   ├── SCIENTIFIC_NUMERICAL_AUDIT.md    # Kiểm toán số liệu, độ lệch chuẩn, hòa giải mâu thuẫn
│   ├── ACADEMIC_WRITING_AUDIT.md        # Kiểm định văn phong học thuật và chuẩn hóa thuật ngữ
│   ├── CONTENT_REMOVED_OR_SIMPLIFIED.md # Danh mục công thức trùng lặp và nội dung rườm rà đã lược bỏ
│   ├── FONT_CONSISTENCY_AUDIT.md        # Báo cáo kiểm định font chữ trên toàn bộ 92 trang
│   ├── FIGURE_AND_TABLE_AUDIT.md        # Kiểm định chi tiết 19 hình vẽ và các bảng biểu quan trọng
│   └── UNRESOLVED_ISSUES.md             # Minh bạch về giới hạn dữ liệu NHANES và các câu hỏi mở
│
└── validation/                          # Minh chứng kiểm thử kỹ thuật và đối sánh
    ├── FONT_REPORT.md                   # Trích xuất 20 font nhúng từ PyMuPDF, xác nhận 0% monospace
    ├── PDF_LAYOUT_QA.md                 # Kiểm tra dàn trang, tọa độ biên an toàn, 0 lỗi ??
    ├── BEFORE_AFTER_COMPARISON.md       # So sánh trực quan đối đầu V9 vs V10
    ├── BUILD_LOG.txt                    # Nhật ký biên dịch sạch của Tectonic
    └── previews/                        # Ảnh chụp các trang quan trọng (Trang 28, 38, 47, 54, 55, 56)
```

---

## 3. HƯỚNG DẪN TÁI LẬP (REPRODUCTION GUIDE)

Mọi thành phần trong báo cáo V10 đều có thể tái lập hoàn toàn từ mã nguồn với các lệnh độc lập sau:

### 3.1. Tái sinh toàn bộ 19 hình ảnh đồ họa:
Yêu cầu môi trường Python 3.10+ đã cài đặt `matplotlib`, `seaborn`, `numpy`, `pandas`:
```bash
cd /Users/nguyenanhson/DepSev-ML
.venv/bin/python reports/final_scientific_v10/source/generate_perfect_figures.py
```
*Kết quả:* Toàn bộ 19 tệp `.png` độ phân giải cao sẽ được lưu tự động vào `reports/final_scientific_v10/source/figures/`.

### 3.2. Biên dịch tài liệu PDF bằng Tectonic:
Sử dụng trình biên dịch hiện đại Tectonic để biên dịch độc lập không phụ thuộc môi trường:
```bash
cd /Users/nguyenanhson/DepSev-ML/reports/final_scientific_v10/source
tectonic main.tex
cp main.pdf ../DepSev_ML_Scientific_Report_V10.pdf
```
*Kết quả:* Sinh tệp `DepSev_ML_Scientific_Report_V10.pdf` (92 trang, dung lượng khoảng 9.12 MiB).

### 3.3. Chạy kịch bản kiểm tra chất lượng PDF:
Kiểm tra số trang, tham chiếu chéo, cảnh báo tràn lề và font chữ:
```bash
.venv/bin/python -c '
import fitz
doc = fitz.open("reports/final_scientific_v10/DepSev_ML_Scientific_Report_V10.pdf")
print("Total pages:", len(doc))
broken = [i+1 for i, page in enumerate(doc) if "??" in page.get_text()]
print("Broken refs (??):", len(broken))
'
```

---

## 4. TÓM TẮT ĐÓNG GÓP KHOA HỌC CHÍNH

1. **Khảo sát phân loại 5 lớp ($T_5$):** Thử nghiệm 14 thuật toán và biến thể qua Nested Cross-Validation (10x5 folds) trên $N = 4{,}880$ mẫu NHANES. Xác định Random Forest (Tuned) đạt hiệu năng tổng thể tốt nhất (Macro-F1 = $0.2755 \pm 0.0270$, Weighted-F1 = $0.6437 \pm 0.0087$), đồng thời chỉ rõ rào cản phân biệt các lớp cận kề do phân phối đặc trưng sinh học chồng lấn.
2. **Khảo sát phân loại phân cấp 2 bước ($T_3$):** Kiến trúc phân loại phân cấp cứng $E_2$ (Hard RF--RF Tuned) tối ưu điểm số thống kê (Macro-F1 = $0.4586 \pm 0.0124$).
3. **Tối ưu chi phí rủi ro Bayes ($E_{6b}$):** Ứng dụng ma trận tổn thất bất đối xứng lâm sàng $M_1$, giúp giảm thiểu $48.33\%$ số ca trầm cảm nặng ($C_2^+$) bị bỏ sót (từ 120 ca xuống 62 ca), nâng cao độ an toàn cho bài toán sàng lọc cộng đồng.
4. **Hồi quy Logistic thứ bậc cân bằng ($E_{7a}$):** Đạt Balanced Accuracy cao nhất ($0.5151 \pm 0.0152$) và duy trì độ ổn định cao nhất trước sự trôi phân phối thời gian ($0.5073$).

