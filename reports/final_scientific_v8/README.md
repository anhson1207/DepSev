# DEPSEV-ML — BÁO CÁO NGHIÊN CỨU KHOA HỌC PHIÊN BẢN V8 (FINAL DELIVERABLES)

**Tên đề tài:** Nghiên cứu khung học máy cho dự đoán mức độ triệu chứng trầm cảm từ dữ liệu sinh lý  
**Tên tiếng Anh:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Tác giả:** Nguyễn Anh Sơn  
**Giáo viên hướng dẫn:** TS. Vũ Thị Hồng Nhạn  
**Đơn vị:** Trường Đại học Công nghệ – Đại học Quốc gia Hà Nội  
**Repository:** https://github.com/anhson1207/DepSev  
**Phiên bản hoàn thiện:** **V8 (Phiên bản xuất bản chính thức, 87 trang PDF)**  

---

## 1. Cấu trúc thư mục bàn giao V8

```text
reports/final_scientific_v8/
├── DepSev_ML_Scientific_Report_V8.pdf   # Báo cáo PDF hoàn chỉnh (87 trang, không lỗi biên dịch)
├── README.md                            # Tài liệu tổng quan bàn giao phiên bản V8
├── audit/                               # Hồ sơ kiểm toán khoa học và bảo đảm chất lượng
│   ├── NUMERICAL_AUDIT.md               # Kiểm toán 100% số liệu thực nghiệm đối soát với repo
│   ├── ACADEMIC_EDITING.md              # Nhật ký chuẩn hóa văn phong học thuật Computer Science
│   ├── REFERENCES_AUDIT.md              # Báo cáo kiểm tra và chuẩn hóa tài liệu tham khảo
│   ├── FIGURE_LAYOUT_QA.md              # Báo cáo thẩm định hình ảnh, bảng biểu và dàn trang
│   ├── UNRESOLVED_ISSUES.md             # Tự phản biện khoa học về các giới hạn nghiên cứu
│   └── previews/                        # Ảnh chụp kết xuất độ phân giải cao của các trang quan trọng
│       ├── page_35.png                  # Bảng 2.1 (Tổng quan nghiên cứu liên quan 4 cột chuẩn)
│       ├── page_57.png                  # Hình 4.6 (Tương quan Spearman & Cramer's V với PHQ-9)
│       ├── page_58.png                  # Hình 4.7 (Biểu đồ hộp 6 biến tiêu biểu 2x3)
│       ├── page_59.png                  # Hình 4.8 (PCA 2D 2 panel: 5 lớp vs 3 lớp)
│       ├── page_65.png                  # Bảng 4.6/4.7 (Đối sánh tổng thể các mô hình T3)
│       ├── page_83.png                  # Phụ lục C (Bảng Macro-F1 chi tiết 5 Outer Folds)
│       └── page_86.png                  # Danh mục tài liệu tham khảo (Hợp nhất Pratt)
└── source/                              # Toàn bộ mã nguồn LaTeX V8
    ├── main.tex                         # Tệp master LaTeX
    ├── references.bib                   # Thư mục BibTeX chuẩn hóa
    ├── generate_perfect_figures.py      # Mã nguồn Python tái tạo 100% hình ảnh 300 DPI
    ├── figures/                         # 19 tệp hình ảnh PNG phân giải cao (300 DPI)
    ├── tables/                          # 10 tệp bảng biểu TeX được đối soát số liệu
    └── chapters/                        # 5 chương chính và phụ lục
        ├── chapter1.tex                 # Giới thiệu (Bối cảnh, Bài toán, RQ1-RQ5, Tiến trình T5->T3)
        ├── chapter2.tex                 # Cơ sở lý thuyết (PHQ-9, Thách thức dữ liệu, RF sâu, 3 hướng mô hình)
        ├── chapter3.tex                 # Dữ liệu & Phương pháp (Quy trình 7 bước, In-fold CV, Không gian F1-F4, E0-E7)
        ├── chapter4.tex                 # Thực nghiệm & Thảo luận (Ablation, Nested CV, Granularity, PCA, E0-E7, Temporal)
        ├── chapter5.tex                 # Kết luận & Hướng phát triển (4 mô hình tiêu biểu, Giới hạn, Tương lai)
        └── appendix.tex                 # Phụ lục A (34 biến), B (Siêu tham số), C (Fold CV), D (Ma trận chi phí)
```

---

## 2. Các điểm cải tiến cốt lõi trong phiên bản V8

1. **Chuẩn hóa văn phong học thuật sinh viên Computer Science (xưng "em"):**
   - Loại bỏ hoàn toàn các cụm từ sáo rỗng, máy móc kiểu AI ("trường phái mô hình hóa", "kiến trúc toàn diện", "cơ chế cách ly bảo toàn tuyệt đối", "nguyên tắc Zero Data Leakage", "triệt tiêu phương sai", "suy thoái nghiêm trọng", "đối chuẩn chuẩn tắc", "chỉ số an toàn lâm sàng").
   - Giữ nguyên danh pháp thuật toán tiếng Anh chuẩn mực (`Random Forest`, `Decision Tree`, `Logistic Regression`, `XGBoost`, `Nested Cross-Validation`, `Soft Voting`, `Stacking`, `Ordinal Logistic Regression`) kết hợp từ vựng dữ liệu tiếng Việt tự nhiên, chính xác.
2. **Sửa dứt điểm các lỗi số liệu và mâu thuẫn thực nghiệm:**
   - Cập nhật chỉ số Recall $C_0$ trong Bảng 4.6 (Phần B) khớp 100% với ma trận nhầm lẫn gộp (pooled out-of-fold): $E_0 = 76.39\%$, $E_1 = 82.16\%$, $E_2 = 77.02\%$, $E_4 = 93.90\%$, $E_5 = 92.61\%$, $E_{6a} = 61.94\%$, $E_{6b} = 61.87\%$, $E_{7a\text{ unw}} = 98.04\%$, $E_{7a\text{ bal}} = 61.24\%$, $E_{7b\text{ LR}} = 75.60\%$, $E_{7b\text{ RF}} = 89.56\%$.
   - Đồng nhất số ca nặng bị bỏ sót ($C_2^+ \to C_0$) ở mô hình $E_2$ là **98 ca** trên toàn bộ báo cáo.
   - Phân định rõ ràng giữa RF mặc định trong Feature Ablation (Macro-F1 = 0.2232) và RF tinh chỉnh trong Nested CV (Macro-F1 = 0.2755 ± 0.0270).
3. **Tái thiết kế toàn diện các hình ảnh và bảng biểu:**
   - **Bảng 2.1:** Cấu trúc lại thành 4 cột thoáng đãng, loại bỏ Kaggle Benchmark phi học thuật, tập trung vào 5 công trình học thuật chính thức.
   - **Hình 4.6:** Tạo mới biểu đồ 2 panel thể hiện tương quan Spearman $\rho$ với PHQ-9 (biến liên tục) và Cramer's V (biến phân loại) từ dữ liệu CSV gốc.
   - **Hình 4.7:** Tinh gọn 15 boxplot dính chữ thành lưới 2x3 gồm 6 biến liên tục tiêu biểu với font chữ lớn và rõ đơn vị đo.
   - **Hình 4.8:** Thiết kế 2 panel PCA 2D (5 lớp $T_5$ và 3 nhóm $T_3$), làm rõ đường bao mật độ và tỷ lệ phương sai giải thích 21.1%.
   - **Các sơ đồ nguyên lý (1.1, 1.2, 2.4, 2.5, 3.1, 3.2, 3.4, 3.5):** Tinh gọn câu chữ, bo góc mượt mà, tăng khoảng cách padding.
4. **Chuẩn hóa danh mục tài liệu tham khảo:**
   - Hợp nhất hoàn toàn mục trích dẫn trùng lặp của Pratt (2014) thành một mục duy nhất `[16]`.
   - Phân tách mạch lạc giữa tài liệu tiếng Việt và tiếng Anh.
5. **Dàn trang và kiểm định chất lượng PDF hoàn hảo:**
   - Tổng độ dài báo cáo: **87 trang**.
   - Số lỗi tham chiếu gãy (`??`): **0 lỗi**.
   - Số lỗi tràn lề (`x1 > 542 pt`): **0 lỗi**.

---

## 3. Hướng dẫn biên dịch lại từ mã nguồn

Để biên dịch báo cáo từ mã nguồn, sử dụng trình biên dịch `tectonic`:

```bash
cd reports/final_scientific_v8/source
/Users/nguyenanhson/.local/bin/tectonic main.tex
cp main.pdf ../DepSev_ML_Scientific_Report_V8.pdf
```

