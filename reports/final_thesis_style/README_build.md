# HƯỚNG DẪN BIÊN DỊCH VÀ XÂY DỰNG KHÓA LUẬN DEPSEV-ML
**(DEPSEV-ML THESIS BUILD INSTRUCTIONS -- UET GRADUATION CLASS)**

Tài liệu này hướng dẫn cách biên dịch toàn bộ mã nguồn $\LaTeX$ của Báo cáo nghiên cứu khoa học / Khóa luận tốt nghiệp **DepSev-ML** dựa trên lớp tài liệu chuẩn chính thức `uetgraduation.cls` (Copyright © 2024 Lê Duy Quang, Trường Đại học Công nghệ -- Đại học Quốc gia Hà Nội).

---

## 1. CẤU TRÚC THƯ MỤC NGUỒN

```text
reports/final_thesis_style/
├── DepSev_ML_Final_Report.pdf      # [SẢN PHẨM CHÍNH] Bản PDF hoàn chỉnh chuẩn UET (66 trang)
├── README_build.md                 # Tài liệu hướng dẫn biên dịch này
├── source/                         # Mã nguồn LaTeX đầy đủ
│   ├── uetgraduation.cls           # Lớp tài liệu chính thức VNU-UET
│   ├── sizes.clo                   # Cấu hình kích cỡ phông và giãn dòng chuẩn
│   ├── UET logo.pdf                # Biểu trưng chính thức Trường ĐH Công nghệ
│   ├── main.tex                    # File LaTeX chủ trì (Master LaTeX)
│   ├── references.bib              # File dữ liệu trích dẫn tham khảo BibTeX
│   ├── frontmatter/                # Các trang bìa và phần mở đầu
│   ├── chapters/                   # Nội dung 5 chương nghiên cứu cốt lõi
│   │   ├── chapter1.tex            # Chương 1: Giới thiệu, Đặt vấn đề, RQ1-RQ5
│   │   ├── chapter2.tex            # Chương 2: Cơ sở lý thuyết, PHQ-9, Tabular ML
│   │   ├── chapter3.tex            # Chương 3: Phương pháp luận, Targets T2-T5, Features F1-F4, 3 Thuật toán
│   │   ├── chapter4.tex            # Chương 4: Thực nghiệm chuyên sâu, E0-E7, Bóc tách ngụy biện Accuracy
│   │   └── chapter5.tex            # Chương 5: Kết luận, Ranh giới y học và Lộ trình tương lai
│   └── figures/                    # 37 biểu đồ và hình ảnh khoa học chất lượng cao
└── validation/                     # Hồ sơ kiểm định và đối soát độc lập
    ├── evidence_mapping.csv        # Ma trận ánh xạ 1-1 từng con số trong báo cáo về file dữ liệu thô
    ├── report_audit.md             # Báo cáo phản biện độc lập toàn diện
    └── preview_pages/              # Ảnh chụp xem trước các trang tiêu biểu của PDF
```

---

## 2. QUY TRÌNH BIÊN DỊCH BẰNG TECTONIC (KHUYẾN NGHỊ CAO NHẤT)

Trình biên dịch **Tectonic** tự động tải các gói TeX cần thiết và đồng bộ tham chiếu chéo nhiều vòng một cách hoàn toàn tự động.

Từ thư mục gốc của repository `DepSev-ML`:
```bash
cd reports/final_thesis_style/source
~/.local/bin/tectonic main.tex -o ../
cd ..
cp main.pdf DepSev_ML_Final_Report.pdf
```
Tệp `DepSev_ML_Final_Report.pdf` sẽ được cập nhật trực tiếp tại thư mục `reports/final_thesis_style/`.

---

## 3. QUY TRÌNH BIÊN DỊCH BẰNG LUATEX / XELATEX (BỘ TE X LIVE TRUYỀN THỐNG)

Nếu hệ thống của bạn đã cài đặt bộ công cụ TeX Live hoặc MacTeX đầy đủ:

### Sử dụng LuaLaTeX (Khuyên dùng theo thiết kế gốc của tác giả Lê Duy Quang):
```bash
cd reports/final_thesis_style/source
lualatex -interaction=nonstopmode main.tex
lualatex -interaction=nonstopmode main.tex
cp main.pdf ../DepSev_ML_Final_Report.pdf
```

### Sử dụng XeLaTeX:
```bash
cd reports/final_thesis_style/source
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
cp main.pdf ../DepSev_ML_Final_Report.pdf
```

---

## 4. CÁC QUY ƯỚC ĐỊNH DẠNG CỦA LỚP TÀI LIỆU UETGRADUATION

1. **Khai báo thông tin bìa:**
   - `\studentname{...}`: Họ và tên sinh viên.
   - `\title{...}`: Tên đề tài khóa luận.
   - `\documenttype{...}`: Loại tài liệu (Khóa luận tốt nghiệp đại học hệ chính quy).
   - `\major{...}`: Ngành học (Công nghệ thông tin).
   - `\year{...}`: Năm bảo vệ (2026).
   - `\supervisor{...}`: Cán bộ hướng dẫn.
   - `\englishtitle{...}`, `\englishmajor{...}`, `\englishsupervisor{...}`: Thông tin tiếng Anh cho trang phụ bìa sinh viên CLC.
   - `\makecovers`: Lệnh tự động tạo 3 trang bìa.
2. **Môi trường Preamble:** Sử dụng `\begin{preamble}{<Tên trang>}` cho các trang Lời cảm ơn, Lời cam đoan, Tóm tắt.
3. **Môi trường Contentlisting:** Đóng gói `\tableofcontents`, `\listoffigures`, `\listoftables` và `\begin{contentlistingsection}{...}` cho danh mục từ viết tắt.
4. **Môi trường Bảng và Hình:**
   - `\begin{table}{<Tiêu đề bảng>} ... \label{...} \end{table}`: Tự động đặt tiêu đề in đậm phía trên bảng và đăng ký vào Danh sách bảng.
   - `\begin{figure}{<Tiêu đề hình>} ... \label{...} \end{figure}`: Tự động đặt tiêu đề căn giữa phía dưới hình và đăng ký vào Danh sách hình vẽ.
5. **Môi trường Tài liệu tham khảo:** Sử dụng `\begin{thebibliography}{99}` kết hợp `\begin{bibsection}{...}` để phân nhóm danh mục tham khảo tiếng Việt và tiếng Anh.
