# BÁO CÁO DỰ ÁN KHOA HỌC DEPSEV-ML (PHIÊN BẢN TINH GỌN V12)
**Đề tài:** Nghiên cứu khung học máy cho dự đoán mức độ triệu chứng trầm cảm từ dữ liệu sinh lý (DepSev-ML)  
**Tập tin PDF hoàn chỉnh:** `DepSev_ML_Scientific_Report_V12.pdf` (56 trang)  
**Ngày hoàn thiện:** 22/11/2026

---

## 1. TỔNG QUAN VỀ PHIÊN BẢN V12

Phiên bản **V12** là bản báo cáo khoa học tinh gọn và hoàn thiện nhất của dự án DepSev-ML, được tái cấu trúc toàn diện theo các góp ý then chốt từ giảng viên:
- **Độ dài tinh gọn thực chất:** Rút gọn từ 91 trang (V11) xuống **56 trang** ("tầm 50 gì đó"), loại bỏ các phần độn lý thuyết giáo khoa chung chung (Decision Tree, Gini, cơ chế sinh học xa rời), tập trung 100% vào phương pháp và phân tích thực nghiệm.
- **Khử triệt để định dạng AI (De-AI Typography & Style):**
  - **Giảm 63% lượng chữ in đậm** (`\textbf`), xóa bỏ việc bôi đậm vụn vặt ở đầu mỗi câu và gạch đầu dòng.
  - Thay thế khuôn mẫu **RQ1, RQ2...** bằng diễn đạt câu hỏi khoa học và mục tiêu nghiên cứu tự nhiên của con người.
- **Cá nhân hóa đồ họa (Custom Visual Taste):**
  - Thay thế toàn bộ bảng màu mặc định kiểu Claude/Tailwind bằng tone màu kỹ thuật hàn lâm (Slate Navy `#2C3E50`, Muted Sage `#4A6572`, Muted Ochre `#B7950B`, Terracotta `#A93226`).
  - Viền mỏng thanh mảnh (`lw=1.1`), góc bo vi mô (`rounding_size=0.8`), thiết kế tối giản như bản vẽ kỹ thuật.
- **Bảo tồn tính đúng đắn khoa học 100%:**
  - Khẳng định chính xác Logistic Regression F4 = $0{,}2399 \pm 0{,}0148$.
  - Toàn bộ độ lệch chuẩn SD ($E_2, E_{3b}, E_{6b}, E_{7a}$) khớp từng con số với CSV gốc.
  - Định nghĩa chuẩn xác 3 biến bị loại của $F_{4\_\text{common31}}$ (`SLD010H`, `PAQ605`, `PAD680`).
  - Phân định rạch ròi mức giảm 70,62% (nhân quả cô lập Bayes trên cùng nền tảng mềm $E_5 \to E_{6b}$) và 48,33% (hệ thống so với baseline phân tầng cứng $E_1 \to E_{6b}$).
  - Gán nhãn `[UNVERIFIED]` cho Patel (2016) và Zhang (2020); cập nhật Dipnall (2016) trên *PLOS ONE*.

---

## 2. CẤU TRÚC THƯ MỤC BÀN GIAO (`reports/final_scientific_v12/`)

```text
reports/final_scientific_v12/
├── DepSev_ML_Scientific_Report_V12.pdf   # Báo cáo PDF hoàn chỉnh (56 trang, biên dịch sạch)
├── README.md                            # Bản hướng dẫn tổng quan này
│
├── source/                              # Mã nguồn LaTeX và tài nguyên
│   ├── main.tex                         # File chính điều phối báo cáo
│   ├── references.bib                   # Danh mục trích dẫn chuẩn IEEE
│   ├── generate_perfect_figures.py      # Mã nguồn sinh toàn bộ đồ thị theo palette mới
│   ├── chapters/                        # 5 chương và phụ lục đã biên tập tinh gọn
│   ├── tables/                          # Các bảng số liệu LaTeX học thuật (booktabs)
│   └── figures/                         # 20 đồ họa phân giải cao 300 DPI theo palette mới
│
├── audit/                               # Tài liệu kiểm toán và đối soát học thuật
│   └── CHANGELOG_V11_TO_V12.md          # Nhật ký thay đổi chi tiết từ V11 sang V12
│
└── validation/                          # Báo cáo kiểm định kỹ thuật tự động
    ├── FONT_REPORT_V12.md               # Kiểm toán font (0% monospace, giảm 63% bold)
    ├── PDF_LAYOUT_QA_V12.md             # Đánh giá bố cục (56 trang, 0 lỗi liên kết)
    └── previews/                        # Ảnh chụp các trang quan trọng
```

---

## 3. HƯỚNG DẪN TÁI LẬP (REPRODUCTION)

Biên dịch độc lập tệp PDF bằng trình biên dịch `tectonic`:
```bash
cd reports/final_scientific_v12/source
tectonic main.tex
cp main.pdf ../DepSev_ML_Scientific_Report_V12.pdf
```
Tái sinh toàn bộ đồ họa 300 DPI:
```bash
python reports/final_scientific_v12/source/generate_perfect_figures.py
```
