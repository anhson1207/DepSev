# DEPSEV-ML: PHIÊN BẢN BÁO CÁO KHOA HỌC V7 (FINAL EDITION)

Thư mục này chứa toàn bộ sản phẩm hoàn thiện của phiên bản báo cáo nghiên cứu **DepSev-ML V7** (`DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data`), được biên tập toàn diện về văn phong học thuật, tinh gọn nội dung và kiểm định số liệu khoa học.

---

## 1. CẤU TRÚC THƯ MỤC V7

```
reports/final_scientific_v7/
├── DepSev_ML_Scientific_Report_V7.pdf  # Báo cáo PDF hoàn chỉnh (89 trang, 9.21 MB)
├── README.md                           # Giới thiệu tổng quan cấu trúc V7
├── source/                             # Toàn bộ mã nguồn LaTeX V7
│   ├── main.tex                        # Tệp cấu hình gốc của báo cáo
│   ├── uetgraduation.cls               # Lớp tài liệu (đã sửa lệnh \chapter)
│   ├── references.bib                  # Danh mục tài liệu tham khảo (đã cập nhật)
│   ├── generate_perfect_figures.py     # Script tạo 19 hình vẽ chuẩn khoa học
│   ├── chapters/                       # Nội dung các chương 1 đến 5 và Phụ lục
│   │   ├── chapter1.tex                # Chương 1: Mở đầu
│   │   ├── chapter2.tex                # Chương 2: Cơ sở lý thuyết & Random Forest
│   │   ├── chapter3.tex                # Chương 3: Phương pháp & Thiết kế thực nghiệm
│   │   ├── chapter4.tex                # Chương 4: Kết quả thực nghiệm & Thảo luận
│   │   ├── chapter5.tex                # Chương 5: Kết luận & Hướng phát triển
│   │   └── appendix.tex                # Phụ lục A, B, C, D
│   ├── tables/                         # Các bảng biểu LaTeX độc lập
│   └── figures/                        # Thư mục chứa toàn bộ 19 hình ảnh phân tích
├── editorial/                          # Tài liệu biên tập văn phong và thuật ngữ
│   ├── ACADEMIC_WRITING_AUDIT.md       # Báo cáo kiểm định văn phong học thuật
│   ├── REDUNDANT_CONTENT_REMOVED.md    # Danh sách nội dung/công thức đã cắt bỏ
│   ├── TERMINOLOGY_STANDARDIZATION.md  # Quy chuẩn thuật ngữ ML & Y tế
│   ├── BEFORE_AFTER_EXAMPLES.md        # 16 ví dụ đối chiếu trực tiếp V6 vs V7
│   └── CONTENT_CHANGELOG.md            # Nhật ký thay đổi nội dung chi tiết
└── validation/                         # Tài liệu kiểm định số liệu và bố cục PDF
    ├── NUMERICAL_AUDIT.md              # Báo cáo kiểm toán số liệu đối soát CSV gốc
    ├── EVIDENCE_MAPPING.csv            # Ánh xạ chi tiết tuyên bố -> tệp tin CSV
    ├── FIGURE_QA.md                    # Báo cáo kiểm tra 19 hình vẽ
    ├── LAYOUT_QA.md                    # Báo cáo kiểm định dàn trang, lề & float
    ├── REMAINING_ISSUES.md             # Tự phê bình các hạn chế còn tồn tại
    ├── BUILD_LOG.txt                   # Nhật ký biên dịch Tectonic
    └── preview_page_*.png              # Hình ảnh trích xuất kiểm tra trực quan
```

---

## 2. NHỮNG ĐIỂM CẢI TIẾN CỐT LÕI CỦA V7

1. **Chuẩn hóa giọng văn học thuật sinh viên CS**:
   - Loại bỏ 100% từ ngữ đao to búa lớn, quảng cáo, sáo rỗng hoặc bi kịch hóa sai số y tế.
   - Giữ nguyên tiếng Anh cho tên thuật toán: `Random Forest (RF)`, `Decision Tree (DT)`, `Logistic Regression (LR)`, `XGBoost`.
   - Sử dụng tiếng Việt tự nhiên cho khái niệm dữ liệu và dịch tễ học cộng đồng.
2. **Đi sâu vào bản chất Random Forest**:
   - Trình bày mạch lạc cấu trúc Decision Tree đơn lẻ, kỹ thuật Bootstrap Sampling, cơ chế ngẫu nhiên hóa không gian đặc trưng và Soft Voting để giải thích việc giảm phương sai mà không làm tăng độ chệch.
   - Cắt bỏ các công thức giải tích OOB 36.8% và phương sai tổng quát mang tính giáo trình hình thức.
3. **Thống nhất tuyệt đối số liệu thực nghiệm**:
   - Số ca nặng $C_2^+ \to C_0$ của mô hình $E_2$ được thống nhất là **98 ca** (đúng theo CSV gốc).
   - Số liệu cột $C_0 \to C_2^+$ trong Bảng 4.6 Phần B được khớp chính xác với ma trận nhầm lẫn out-of-fold gộp.
   - Thống nhất quy ước ma trận chi phí $\mathbf{C}(i, j)$ với $i$ là nhãn thực tế, $j$ là nhãn dự đoán.
4. **Bổ sung và sắp xếp lại hệ thống hình ảnh**:
   - Bổ sung biểu đồ nhiệt Confusion Matrix 5x5 của $T_5$ (Hình 4.5) thể hiện cả số lượng tuyệt đối và phần trăm.
   - Thay thế Hình 4.8 tại Mục 4.5 bằng biểu đồ đối sánh các nhánh dữ liệu $T_2 \to T_5$ (`multicycle_cohort_comparison.png`).
   - Chuyển `multicycle_soft_comparison.png` về đúng vị trí kiểm định dòng thời gian (Hình 4.19, Mục 4.8).
5. **Sửa triệt để lỗi dàn trang TeX**:
   - Bổ sung `\suppressfloats[t]` trong `uetgraduation.cls` và `appendix.tex`, chấm dứt hoàn toàn hiện tượng Bảng C.1 bị trôi lên phía trên tiêu đề Phụ lục C.
   - 0 khối văn bản vượt lề phải ($x_1 \le 539.0$ pt trên toàn bộ 89 trang).
   - 0 liên kết tham chiếu gãy (`??`).

