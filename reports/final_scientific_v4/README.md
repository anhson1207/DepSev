# BÁO CÁO DỰ ÁN KHOA HỌC DEPSEV-ML (PHIÊN BẢN V4)

**Đề tài:** Nghiên cứu khung học máy cho dự đoán mức độ triệu chứng trầm cảm từ dữ liệu sinh lý (DepSev-ML)  
**Tiếng Anh:** A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Sinh viên thực hiện:** Nguyễn Anh Sơn  
**Cán bộ hướng dẫn:** TS. Vũ Thị Hồng Nhạn  
**Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Công nghệ – ĐHQGHN  
**Thời gian:** 2026  

---

## 1. CẤU TRÚC BÀN GIAO THƯ MỤC V4

```text
reports/final_scientific_v4/
├── DepSev_ML_Scientific_Report_V4.pdf    # Tệp PDF báo cáo khoa học hoàn chỉnh (72 trang, 0 lỗi tràn lề, full hyperlink TOC)
├── README.md                             # Hướng dẫn tổng quan gói bàn giao V4
├── source/                               # Toàn bộ mã nguồn LaTeX hoàn chỉnh
│   ├── main.tex                          # Tệp điều khiển chính
│   ├── uetgraduation.cls                 # Class định dạng chuẩn VNU-UET
│   ├── sizes.clo                         # Tệp cấu hình kích thước font
│   ├── UET logo.pdf                      # Biểu trưng vector Trường ĐH Công nghệ
│   ├── references.bib                    # Thư viện tài liệu tham khảo BibTeX
│   ├── chapters/                         # 5 chương nội dung + Phụ lục kỹ thuật
│   │   ├── chapter1.tex                  # Chương 1: Giới thiệu (Đặt vấn đề)
│   │   ├── chapter2.tex                  # Chương 2: Cơ sở lý thuyết & Nghiên cứu liên quan
│   │   ├── chapter3.tex                  # Chương 3: Dữ liệu & Phương pháp nghiên cứu
│   │   ├── chapter4.tex                  # Chương 4: Thực nghiệm & Thảo luận
│   │   ├── chapter5.tex                  # Chương 5: Kết luận & Hướng phát triển
│   │   └── appendix.tex                  # Phụ lục A, B, C, D
│   ├── figures/                          # 16 hình vẽ khoa học vector / 300 DPI
│   └── tables/                           # 8 tệp bảng dữ liệu chuẩn Booktabs
├── editorial/                            # Hồ sơ biên tập học thuật chuyên nghiệp
│   ├── REFERENCE_STYLE_ANALYSIS.md       # Phân tích 8 khóa luận mẫu UET và định vị V4
│   ├── CURRENT_REPORT_CRITIQUE.md        # Đánh giá phản biện bản V3 và định hướng V4
│   ├── V4_OUTLINE.md                     # Đề cương chi tiết từng chương mục
│   ├── VISUAL_AND_EQUATION_PLAN.md       # Kế hoạch tinh gọn công thức và hệ thống hình ảnh
│   ├── CONTENT_CHANGELOG.md              # Nhật ký chi tiết các thay đổi từ V3 sang V4
│   └── ACADEMIC_WRITING_REVIEW.md        # Đánh giá biên tập văn phong và tự chấm điểm
└── validation/                           # Hồ sơ kiểm định khoa học và dàn trang
    ├── SCIENTIFIC_VERIFICATION.md        # Báo cáo kiểm định tính đúng đắn khoa học
    ├── EVIDENCE_MAPPING.csv              # Ánh xạ từng khẳng định khoa học với file nguồn có thật
    ├── LAYOUT_QA.md                      # Báo cáo kiểm thử dàn trang và quét tọa độ PyMuPDF
    ├── REMAINING_ISSUES.md               # Các lưu ý khi rà soát thủ công trước khi nộp
    └── BUILD_LOG.txt                     # Nhật ký biên dịch hoàn chỉnh từ TeX Engine
```

---

## 2. HƯỚNG DẪN BIÊN DỊCH LẠI (REPRODUCIBILITY)

Để biên dịch lại tệp PDF từ mã nguồn LaTeX, sử dụng trình biên dịch Tectonic:

```bash
cd reports/final_scientific_v4/source
tectonic main.tex -o ..
cp ../main.pdf ../DepSev_ML_Scientific_Report_V4.pdf
```

