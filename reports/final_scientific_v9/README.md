# DEPSEV-ML — BÁO CÁO NGHIÊN CỨU KHOA HỌC PHIÊN BẢN V9 (FINAL DELIVERABLES)

**Tên đề tài:** Nghiên cứu khung học máy cho dự đoán mức độ triệu chứng trầm cảm từ dữ liệu sinh lý  
**Tên tiếng Anh:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Tác giả:** Nguyễn Anh Sơn  
**Giáo viên hướng dẫn:** TS. Vũ Thị Hồng Nhạn  
**Đơn vị:** Trường Đại học Công nghệ – Đại học Quốc gia Hà Nội  
**Repository:** https://github.com/anhson1207/DepSev  
**Phiên bản hoàn thiện:** **V9 (Phiên bản biên tập học thuật hoàn chỉnh, 91 trang PDF)**  

---

## 1. Cấu trúc thư mục bàn giao V9

```text
reports/final_scientific_v9/
├── DepSev_ML_Scientific_Report_V9.pdf   # Báo cáo PDF hoàn chỉnh (91 trang, không lỗi biên dịch)
├── README.md                            # Tài liệu tổng quan bàn giao phiên bản V9
├── RESEARCH_NARRATIVE_AUDIT.md          # Nhật ký chuyển đổi mạch nghiên cứu sang so sánh nhiều hướng
├── audit/                               # Hồ sơ kiểm toán khoa học và bảo đảm chất lượng
│   ├── NUMERICAL_AUDIT.md               # Kiểm toán 100% số liệu thực nghiệm đối soát với repo
│   ├── ACADEMIC_EDITING.md              # Nhật ký chuẩn hóa văn phong học thuật Computer Science
│   ├── REFERENCES_AUDIT.md              # Báo cáo kiểm tra và chuẩn hóa tài liệu tham khảo
│   ├── FIGURE_LAYOUT_QA.md              # Báo cáo thẩm định hình ảnh, bảng biểu và dàn trang
│   ├── UNRESOLVED_ISSUES.md             # Tự phản biện khoa học về các giới hạn nghiên cứu
│   └── previews/                        # Ảnh chụp kết xuất độ phân giải cao của các trang quan trọng
│       ├── page_39.png                  # Bảng 2.1 (Tổng quan nghiên cứu liên quan 4 cột chuẩn)
│       ├── page_60.png                  # Hình 4.6 (Tương quan Spearman & Cramer's V với PHQ-9)
│       ├── page_62.png                  # Hình 4.7 (Biểu đồ hộp 6 biến tiêu biểu 2x3)
│       ├── page_63.png                  # Hình 4.8 (PCA 2D 2 panel: 5 lớp vs 3 lớp)
│       ├── page_68.png                  # Bảng 4.6/4.7 (Đối sánh tổng thể các mô hình T3)
│       └── page_79.png                  # Chương 5 (Kết luận và hướng phát triển)
└── source/                              # Toàn bộ mã nguồn LaTeX V9
    ├── main.tex                         # Tệp master LaTeX
    ├── references.bib                   # Thư mục BibTeX chuẩn hóa
    ├── generate_perfect_figures.py      # Mã nguồn Python tái tạo 100% hình ảnh 300 DPI
    ├── figures/                         # 19 tệp hình ảnh PNG phân giải cao (300 DPI)
    ├── tables/                          # 10 tệp bảng biểu TeX được đối soát số liệu
    └── chapters/                        # 5 chương chính và phụ lục
        ├── chapter1.tex                 # Giới thiệu (Bối cảnh, Bài toán, RQ1-RQ5, Hai hướng nghiên cứu)
        ├── chapter2.tex                 # Cơ sở lý thuyết (PHQ-9, Thách thức dữ liệu, RF sâu, 3 hướng mô hình)
        ├── chapter3.tex                 # Dữ liệu & Phương pháp (Quy trình 7 bước, In-fold CV, F1-F4, E0-E7)
        ├── chapter4.tex                 # Thực nghiệm & Thảo luận (Ablation, Nested CV, Granularity, PCA, E0-E7, Temporal)
        ├── chapter5.tex                 # Kết luận & Hướng phát triển (Tổng kết 2 hướng, 4 mô hình tiêu biểu, Giới hạn)
        └── appendix.tex                 # Phụ lục A (34 biến), B (Siêu tham số), C (Fold CV), D (Ma trận chi phí)
```

---

## 2. Các điểm cải tiến cốt lõi trong phiên bản V9

1. **Chuyển đổi toàn diện mạch nghiên cứu từ "khắc phục thất bại" sang "nghiên cứu so sánh nhiều hướng":**
   - Loại bỏ hoàn toàn các cấu trúc câu tiêu cực (*"không thành công nên chuyển hướng"*, *"buộc phải thay đổi"*, *"thất bại của mô hình"*, *"điểm nghẽn nghiêm trọng"*, *"bất lực trong nhận diện"*).
   - Tinh thần chủ đạo: Nghiên cứu thiết kế, triển khai và đánh giá nhiều cách tiếp cận khác nhau nhằm tìm hiểu hiệu quả, đặc điểm và sự đánh đổi của các mô hình dự đoán mức độ triệu chứng trầm cảm từ dữ liệu NHANES.
   - Hai hướng nghiên cứu được kết nối logic: Hướng 1 khảo sát phân loại đa mức trên $T_5$ (làm rõ đóng góp đặc trưng và hiện tượng chồng lấn sinh lý ngoại vi); Hướng 2 khảo sát phân loại phân cấp và thứ bậc trên $T_3$ (đối sánh Flat, Hard Hierarchical, Soft Hierarchical + Chi phí Bayes, và Ordinal).
2. **Khẳng định tính độc lập và khách quan giữa $T_3$ và $T_5$:**
   - Tuyệt đối không khẳng định $T_3$ "tốt hơn" $T_5$ chỉ bằng cách so sánh Macro-F1 giữa hai không gian nhãn khác nhau.
   - Không xuyên tạc lịch sử thực nghiệm; kết quả từ $T_5$ là cơ sở thực nghiệm mở đường cho $T_3$.
3. **Bảo toàn 100% số liệu gốc và tính trung thực khoa học:**
   - Giữ nguyên các số liệu thực nghiệm: Random Forest được chọn trong phạm vi so sánh phù hợp (Nested CV Macro-F1 = 0.2755), không phải tốt nhất tuyệt đối; $E_2$ có Macro-F1 cao nhất trên $T_3$ (0.4586) nhưng bỏ sót 98 ca nặng tại Stage 1; $E_{7a}$ Balanced OLR đạt Balanced Accuracy cao nhất (0.5151); $E_{6b}$ giảm lỗi bỏ sót ca nặng nhưng đánh đổi với số ca báo động nhầm.
   - Nêu rõ: Đề tài chưa chứng minh khả năng ứng dụng lâm sàng thực tế, đóng vai trò là nghiên cứu khảo sát phương pháp trên dữ liệu cộng đồng.
4. **Viết lại toàn diện Chương 5:**
   - Nhấn mạnh đầy đủ các phát hiện tích cực đạt được trên từng hướng nghiên cứu.
   - Trình bày 4 mô hình tiêu biểu đại diện cho 4 tiêu chí thực tiễn khác nhau.
   - Thảo luận khách quan các giới hạn và định hướng tương lai.
5. **Chất lượng dàn trang và biên dịch:**
   - Biên dịch thành công 91 trang PDF đạt chuẩn học thuật bằng `tectonic`.
   - 0 lỗi tham chiếu gãy (`??`), 0 lỗi tràn lề.

---

## 3. Hướng dẫn biên dịch lại từ mã nguồn

Để biên dịch báo cáo từ mã nguồn, sử dụng trình biên dịch `tectonic`:

```bash
cd reports/final_scientific_v9/source
/Users/nguyenanhson/.local/bin/tectonic main.tex
cp main.pdf ../DepSev_ML_Scientific_Report_V9.pdf
```
