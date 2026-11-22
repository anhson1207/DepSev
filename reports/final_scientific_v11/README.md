# BÁO CÁO DỰ ÁN KHOA HỌC DEPSEV-ML (PHIÊN BẢN V11)
**Đề tài:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Repository:** https://github.com/anhson1207/DepSev  
**Ngày hoàn thiện:** 31/10/2026  
**Tập tin PDF hoàn chỉnh:** `DepSev_ML_Scientific_Report_V11.pdf` (91 trang)

---

## 1. GIỚI THIỆU TỔNG QUAN

Phiên bản **V11** là bản báo cáo khoa học cuối cùng, hoàn thiện và toàn diện nhất của dự án **DepSev-ML**. Phiên bản này được biên tập và kiểm định khắt khe dưới góc nhìn của một sinh viên ngành Khoa học Máy tính khiêm tốn, cầu thị và trung thực, giải quyết triệt để toàn bộ các phản biện khoa học từ phiên bản V10:
- **Chuẩn hóa số liệu 100%:** Đối soát khớp từng con số với các tệp dữ liệu CSV gốc trong kho mã nguồn (LR F4 = 0,2399; các độ lệch chuẩn SD của $E_2, E_{3b}, E_{6b}, E_{7a}$; phân định rạch ròi quy ước giảm sai số ca nặng 70,62% vs 48,33%).
- **Làm rõ bản chất tập đặc trưng $F_{4\_\text{common31}}$:** Loại bỏ quan niệm sai lầm về Triglycerides (`LBXTR`), chứng minh chính xác 3 biến bị loại là `SLD010H`, `PAQ605`, và `PAD680`.
- **Tái tạo đồ họa xuất bản chuyên nghiệp:** 20 đồ thị chuẩn mực độ phân giải 300 DPI, sử dụng 100% font Times New Roman serif, làm gọn nội dung các sơ đồ khối kiến trúc.
- **Tối ưu hóa dàn trang:** Gỡ bỏ các rào cản đọng phao `\FloatBarrier`, tạo luồng đọc mượt mà từ trang 68 đến trang 76, không còn trang trắng bất thường.
- **Triệt tiêu hoàn toàn font Monospace:** Đạt tỷ lệ **0,00% monospace** trên toàn bộ 91 trang tài liệu.
- **Liêm chính học thuật:** Gán nhãn tường minh `[UNVERIFIED]` cho các tài liệu tham khảo chưa thể xác minh độc lập (Patel 2016, Zhang 2020) và hiệu chỉnh chính xác bài báo của Dipnall (2016) trên *PLOS ONE*.

---

## 2. CẤU TRÚC THƯ MỤC BÀN GIAO (`reports/final_scientific_v11/`)

```text
reports/final_scientific_v11/
├── DepSev_ML_Scientific_Report_V11.pdf   # Báo cáo PDF hoàn chỉnh (91 trang, biên dịch sạch)
├── README.md                            # Bản hướng dẫn tổng quan và tái lập này
│
├── source/                              # Toàn bộ mã nguồn LaTeX và tài nguyên
│   ├── main.tex                         # Tệp chính điều phối tài liệu
│   ├── references.bib                   # Cơ sở dữ liệu trích dẫn chuẩn hóa
│   ├── generate_perfect_figures.py      # Mã nguồn Python tái sinh toàn bộ đồ họa 300 DPI
│   ├── chapters/                        # Các chương nội dung chi tiết
│   │   ├── chapter1.tex                 # Chương 1: Giới thiệu & Đặt vấn đề
│   │   ├── chapter2.tex                 # Chương 2: Cơ sở lý thuyết & Nghiên cứu liên quan
│   │   ├── chapter3.tex                 # Chương 3: Phương pháp & Thiết kế thực nghiệm
│   │   ├── chapter4.tex                 # Chương 4: Kết quả thực nghiệm & Bàn luận
│   │   ├── chapter5.tex                 # Chương 5: Bàn luận chung, Đóng góp & Kết luận
│   │   └── appendix.tex                 # Phụ lục tra cứu chi tiết
│   ├── tables/                          # Các bảng số liệu LaTeX học thuật (booktabs)
│   └── figures/                         # 20 tệp hình ảnh PNG phân giải cao (300 DPI)
│
├── audit/                               # Tài liệu kiểm toán và đối soát học thuật
│   ├── SCIENTIFIC_NUMERICAL_AUDIT_V11.md # Báo cáo kiểm định chi tiết toàn bộ số liệu
│   ├── CHANGELOG_V10_TO_V11.md          # Nhật ký thay đổi chi tiết từ V10 sang V11
│   └── UNVERIFIED_REFERENCES_AND_LIMITATIONS.md # Báo cáo trích dẫn chưa xác minh & giới hạn
│
└── validation/                          # Tài liệu kiểm định kỹ thuật tự động
    ├── BUILD_LOG.txt                    # Nhật ký biên dịch hoàn tất của Tectonic
    ├── FONT_REPORT_V11.md               # Báo cáo trích xuất font qua PyMuPDF (0% monospace)
    ├── PDF_LAYOUT_QA_V11.md             # Báo cáo kiểm định dàn trang và phân bổ float
    └── previews/                        # Ảnh chụp các trang quan trọng để kiểm tra trực quan
```

---

## 3. HƯỚNG DẪN TÁI LẬP (REPRODUCTION GUIDE)

### 3.1. Biên dịch báo cáo PDF từ mã nguồn LaTeX
Để tái lập tệp PDF một cách độc lập từ mã nguồn bằng trình biên dịch `tectonic`:
```bash
cd reports/final_scientific_v11/source
tectonic main.tex
# Sao chép tệp PDF thành phẩm ra thư mục gốc
cp main.pdf ../DepSev_ML_Scientific_Report_V11.pdf
```

### 3.2. Tái sinh toàn bộ đồ họa 300 DPI
Để chạy lại script tạo toàn bộ 20 đồ thị từ các tệp kết quả CSV:
```bash
python reports/final_scientific_v11/source/generate_perfect_figures.py
```

### 3.3. Kiểm định font chữ và liên kết gãy tự động
Để kiểm tra tính toàn vẹn của tệp PDF (xác nhận 0% monospace và 0 lỗi liên kết gãy):
```python
import fitz

doc = fitz.open("reports/final_scientific_v11/DepSev_ML_Scientific_Report_V11.pdf")
print("Tổng số trang:", len(doc))

broken = [i + 1 for i, page in enumerate(doc) if "??" in page.get_text()]
print("Liên kết gãy (??):", broken)

mono_count = 0
for page in doc:
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                fname = span.get("font", "").lower()
                if any(x in fname for x in ["mono", "courier", "consolas", "typewriter"]):
                    mono_count += len(span.get("text", ""))
print("Số ký tự font Monospace:", mono_count)
```

---

## 4. CÁC ĐÓNG GÓP KHOA HỌC CHÍNH CỦA BÁO CÁO V11

1. **Khảo sát đa tầng có hệ thống:** So sánh 12 cấu hình mô hình thuộc 4 hướng tiếp cận (Phân loại phẳng $T_5$, Phân tầng cứng $T_3$, Phân tầng mềm kết hợp Lý thuyết quyết định Bayes, và Phân loại thứ bậc Ordinal).
2. **Minh định nút thắt sai số tích lũy:** Chỉ ra điểm nghẽn thực nghiệm của cấu trúc phân tầng cứng ($E_1$--$E_4$) khi sai số Giai đoạn 1 ($C_0$ vs $C_1^+$) triệt tiêu hoàn toàn khả năng nhận diện ca nặng ở Giai đoạn 2.
3. **Phát hiện giá trị y tế của Lý thuyết quyết định Bayes:** Chứng minh rằng trên cùng một không gian phân phối xác suất mềm ($E_5$), việc áp dụng ma trận phạt tổn thất bất đối xứng ($E_{6b}$) giúp giảm **70,62%** số ca trầm cảm nặng bị bỏ sót ($C_2^+ \to C_0$), tăng độ an toàn sàng lọc y tế dự phòng cộng đồng.
4. **Kiểm định ngoại suy thời gian khách quan:** Đánh giá trên tập dữ liệu độc lập sau 2 năm (NHANES 2015--2016) nhằm kiểm chứng khả năng chống chịu hiện tượng trôi phân phối dữ liệu (data drift).
