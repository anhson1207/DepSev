# DEPSEV-ML: FINAL SCIENTIFIC REPORT VERSION 6 (V6)
**Đề tài:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Tác giả:** Nguyễn Anh Sơn (Khoa học Máy tính - Trường Đại học Công nghệ, ĐHQGHN)  
**Cán bộ hướng dẫn:** TS. Vũ Thị Hồng Nhạn  
**Phiên bản hoàn thiện:** V6 (30/10/2026)  

---

## 1. Cấu trúc thư mục Báo cáo V6

```text
reports/final_scientific_v6/
├── DepSev_ML_Scientific_Report_V6.pdf   # Báo cáo PDF hoàn chỉnh 89 trang, 0 lỗi biên lề, 543 hyperlinks
├── README.md                            # Hướng dẫn và tổng quan phiên bản V6 (tệp này)
├── source/                              # Toàn bộ mã nguồn LaTeX hoàn chỉnh có thể biên dịch độc lập
│   ├── main.tex                         # Tệp gốc LaTeX (cấu hình class, gói thư viện, front matter)
│   ├── uetgraduation.cls                # Lớp văn bản khóa luận chuẩn UET (ĐHQGHN)
│   ├── sizes.clo                        # Cỡ chữ chuẩn 13pt
│   ├── references.bib                   # Thư viện tài liệu tham khảo BibTeX (19 trích dẫn chuẩn IEEE)
│   ├── UET logo.pdf                     # Logo trường Đại học Công nghệ vector độ nét cao
│   ├── cover_page_*.png                 # Ảnh bìa trang nhã
│   ├── generate_perfect_figures.py      # Mã nguồn sinh 17 hình ảnh khoa học chất lượng cao (300 DPI)
│   ├── chapters/                        # Nội dung chi tiết các chương
│   │   ├── chapter1.tex                 # Chương 1: Giới thiệu (Dẫn dắt bài toán T5 -> T3)
│   │   ├── chapter2.tex                 # Chương 2: Cơ sở lý thuyết (Tập trung sâu Random Forest)
│   │   ├── chapter3.tex                 # Chương 3: Dữ liệu và Phương pháp nghiên cứu
│   │   ├── chapter4.tex                 # Chương 4: Thực nghiệm và Thảo luận (Đối soát 100% số liệu)
│   │   ├── chapter5.tex                 # Chương 5: Kết luận và Hướng phát triển
│   │   └── appendix.tex                 # Phụ lục A, B, C, D (Khớp chính xác số liệu fold)
│   ├── figures/                         # Toàn bộ 56 hình vẽ minh họa đồ họa
│   └── tables/                          # 10 tệp bảng biểu LaTeX dạng bảng chuẩn
├── editorial/                           # Báo cáo biên tập học thuật độc lập
│   ├── V5_SCIENTIFIC_AUDIT.md           # Kiểm định khoa học chi tiết 8 lỗi P0 của V5 và cách khắc phục
│   ├── CONTENT_CHANGELOG.md             # Nhật ký thay đổi chi tiết từng chương, mục, bảng
│   ├── ACADEMIC_WRITING_REVIEW.md       # Rà soát phong cách viết học thuật, loại bỏ khoa trương
│   ├── FIGURE_AND_EQUATION_REVIEW.md    # Đánh giá 17 hình ảnh xuất bản và các công thức toán học
│   └── BEFORE_AFTER.md                  # Bảng đối sánh trực diện V5 vs V6
└── validation/                          # Bằng chứng kiểm định số liệu và kiểm thử kỹ thuật
    ├── NUMERICAL_VERIFICATION.csv       # Ánh xạ toàn bộ số liệu báo cáo về file CSV gốc trong repository
    ├── EVIDENCE_MAPPING.csv             # Bản đồ bằng chứng cho 5 câu hỏi khoa học (RQ1 - RQ5)
    ├── TEMPORAL_PROTOCOL_AUDIT.md       # Kiểm toán chuyên sâu giao thức dòng thời gian Branch D
    ├── LAYOUT_QA.md                     # Báo cáo QA dàn trang: 0 lỗi lề, 0 tham chiếu gãy, 543 liên kết
    ├── UNRESOLVED_ISSUES.md             # Danh mục rào cản dữ liệu và điều kiện biên thực nghiệm
    └── BUILD_LOG.txt                    # Nhật ký biên dịch hoàn hảo của Tectonic Engine
```

---

## 2. Hướng dẫn biên dịch lại PDF từ mã nguồn

Báo cáo được thiết kế để biên dịch hoàn hảo và tái lập độc lập bằng công cụ `tectonic` hoặc `xelatex`:

```bash
# Cách 1: Sử dụng Tectonic (Khuyến nghị, tự động tải gói và quản lý cache)
cd reports/final_scientific_v6/source
tectonic main.tex

# Cách 2: Tái sinh toàn bộ 17 hình ảnh xuất bản trước khi biên dịch
python generate_perfect_figures.py
tectonic main.tex
```

---

## 3. Các đặc điểm vượt trội của Báo cáo Khoa học V6
1. **Tính trung thực dữ liệu tuyệt đối:** Khắc phục triệt để mâu thuẫn số ca bỏ sót ca nặng của $E_{6b}$ (xác nhận chính xác **62 ca**, giảm **48.33%** với ma trận $M_1$; giảm tiếp còn 19 ca với $M_2$).
2. **Khôi phục tính đúng đắn toán học:** Điền chính xác số liệu từng Outer Fold trong Phụ lục C từ `nested_cv_baseline_comparison.csv` (độ lệch chuẩn khớp phương sai mẫu ddof=1). Sửa định nghĩa ma trận chi phí Bayes trong Phụ lục D.
3. **Mạch nghiên cứu chuẩn mực $T_5 \to T_3$:** Phân tích ma trận nhầm lẫn $5 \times 5$ của $T_5$, chỉ rõ lý do toán học và y sinh học khiến mô hình phẳng thất bại (Recall $C_4$ chỉ đạt $2.27\%$), tạo tiền đề tất yếu cho kiến trúc phân cấp $T_3$.
4. **Tập trung sâu sắc vào Random Forest:** Giải thích cặn kẽ Cây quyết định, chỉ số Gini, hạn chế phương sai cao, kỹ thuật Bagging, mẫu OOB ($36.8\%$), cơ chế giảm phương sai và vai trò thực tiễn của từng siêu tham số trong *Frozen Random Forest Spec*.
5. **Mỹ thuật trình bày đỉnh cao:** Dàn trang 89 trang chuẩn khổ A4, không có bất kỳ dòng tràn lề nào ($x_1 \le 540.58$ pt), loại bỏ hoàn toàn các liên kết gãy (`??`), kích hoạt 543 siêu liên kết tương tác thông minh.

