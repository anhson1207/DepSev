# BÁO CÁO KIỂM TRA CHẤT LƯỢNG TRÌNH BÀY PDF (PDF LAYOUT QA) — DEPSEV-ML V11
**Tập tin kiểm định:** `DepSev_ML_Scientific_Report_V11.pdf`  
**Ngày kiểm định:** 31/10/2026  
**Tổng số trang:** 91 trang (đạt mục tiêu không tăng số trang từ V10/V9)  
**Tình trạng biên dịch:** Hoàn tất không có lỗi (`tectonic` code 0)  
**Kiểm tra liên kết gãy (`??`):** **0 lỗi phát hiện trên toàn bộ 91 trang**

---

## 1. TỔNG QUAN BỐ CỤC THEO CHƯƠNG VÀ CÁC TRANG

| Phân đoạn | Khoảng trang | Tình trạng dàn trang | Ghi chú kiểm định |
| :--- | :---: | :---: | :--- |
| **Trang bìa & Thông tin dự án** | 1 -- 3 | Chuẩn mực | Khối tiêu đề, tác giả, giảng viên hướng dẫn cân đối, trang nhã. |
| **Tóm tắt (Abstract Tiếng Việt & Anh)** | 4 -- 5 | Cân đối | Mỗi phần tóm tắt chiếm trọn 1 trang, lề đều hai bên. |
| **Mục lục & Danh mục Bảng / Hình** | 6 -- 11 | Không tràn trang | Mục lục, danh mục 20 hình và danh mục 15 bảng hiển thị đúng số trang. |
| **Chương 1: Giới thiệu & Đặt vấn đề** | 12 -- 22 | Liền mạch | Dẫn dắt bài toán, mục tiêu nghiên cứu và cấu trúc luận văn. |
| **Chương 2: Cơ sở lý thuyết & Nghiên cứu liên quan** | 23 -- 39 | Liền mạch, đẹp | Hình 2.1 (trang 27) sắc nét; Bảng 2.1 (trang 38) 4 cột cân xứng hoàn hảo. |
| **Chương 3: Phương pháp & Quy trình thực nghiệm** | 40 -- 54 | Bố cục tối ưu | Hình 3.4 (trang 47) và Hình 3.5 (trang 48) thu gọn chữ, sơ đồ mạch lạc. |
| **Chương 4: Kết quả thực nghiệm & Bàn luận** | 55 -- 79 | Luồng hình mượt mà | Khu vực nhạy cảm trang 68--76 được gỡ bỏ rào cản đọng phao, dàn đều đẹp mắt. |
| **Chương 5: Bàn luận chung, Đóng góp & Kết luận** | 80 -- 84 | Súc tích, học thuật | Tóm lược đóng góp và giới hạn nghiên cứu trung thực. |
| **Tài liệu tham khảo & Phụ lục** | 85 -- 91 | Chuẩn IEEE | Toàn bộ 41 mục tài liệu tham khảo đồng nhất định dạng, gán nhãn `[UNVERIFIED]`. |

---

## 2. KIỂM ĐỊNH CHI TIẾT CÁC ĐIỂM CẢI TIẾN TRỌNG TÂM

### 2.1. Bảng 2.1 (Trang 38): Tổng hợp nghiên cứu liên quan
- **Vấn đề ở bản cũ:** Bảng có 4 cột nhưng độ rộng phân bổ không hợp lý (`p{2.6cm} p{2.2cm} p{3.2cm} X`), khiến dòng tên tác giả Zhang et al. bị xuống dòng cụt ("(2020)") và dòng DepSev-ML bị ép cụm từ dài.
- **Khắc phục ở V11:** Đã tinh chỉnh lại cấu trúc cột thành `p{3.0cm} p{2.2cm} p{3.4cm} X`.
- **Kết quả trực quan:** Tên tác giả "Zhang et al. (2020) [UNVERIFIED]" nằm trọn trên 2 dòng đẹp mắt; thông số của mô hình đề xuất DepSev-ML không bị xô lệch, đường kẻ `\toprule`, `\midrule`, `\bottomrule` chuẩn phong cách tạp chí y sinh.

### 2.2. Hình 2.1 (Trang 27): Cấu trúc phân tầng PHQ-9
- **Vấn đề ở bản cũ:** Bên trong các ô phân tầng có nhiều câu mô tả chữ nhỏ 7pt, chữ dính sát đường biên hộp gây cảm giác chật chội.
- **Khắc phục ở V11:** Đã viết lại hàm `make_phq9_tiers()` trong `generate_perfect_figures.py`:
  - Loại bỏ hoàn toàn các đoạn văn dài thừa thãi bên trong ô hộp.
  - Phần A (Bài toán gốc 5 lớp $T_5$): Chỉ hiển thị rõ ràng Tên lớp, Điểm số, Số mẫu ($N$) và Tỷ lệ phần trăm.
  - Phần B (Bài toán lâm sàng 3 tầng $T_3$): Chỉ hiển thị Nhóm tầng lâm sàng ($C_0, C_1, C_2^+$) và Ý nghĩa hành động y tế thực tiễn (Sàng lọc ban đầu, Can thiệp lối sống, Chuyển gửi chuyên khoa khẩn cấp).

### 2.3. Hình 3.4 (Trang 47) & Hình 3.5 (Trang 48): Sơ đồ kiến trúc mô hình
- **Hình 3.4 (Phân tầng cứng $E_1$--$E_4$):**
  - Text trong các khối Stage 1 (Tách $C_0$ vs $C_1^+$) và Stage 2 (Tách $C_1$ vs $C_2^+$) được cô đọng, cỡ chữ tăng lên $\ge 9$pt Times New Roman.
  - Hộp thắt cổ chai sai số tích lũy (Error Bottleneck) được đóng khung nét đứt màu đỏ thanh lịch, làm nổi bật bản chất hạn chế của cấu trúc cứng.
- **Hình 3.5 (Phân tầng mềm & Lý thuyết quyết định Bayes $E_5$--$E_6$):**
  - Tập trung trực quan hóa chính xác cơ chế toán học: Vector xác suất hậu nghiệm $P(C_k|x)$ đi qua Luật quyết định Bayes Risk $M_1$ với trọng số phạt $c(C_2^+ \to C_0) = 10$.
  - Không còn hiện tượng chữ đè lên các mũi tên luồng dữ liệu.

### 2.4. Khu vực kết quả nhạy cảm: Trang 68 đến Trang 76
- **Vấn đề ở bản cũ:** Do đặt các lệnh `\FloatBarrier` dồn dập sau mỗi đồ thị, LaTeX bị khóa phao khiến đồ thị bị đẩy dồn về cuối chương, để lại các khoảng trống trắng lớn (trang 69, 71, 72 chắp vá).
- **Khắc phục ở V11:**
  - Gỡ bỏ các rào cản cứng `\FloatBarrier` thừa thãi.
  - Thiết lập chiều cao và vị trí float tối ưu:
    - **Trang 68:** Trình bày bảng Bảng 4.6 (Kết quả Ordinal) và đồ thị Hình 4.11.
    - **Trang 69:** Đồ thị Hình 4.12 (Vi phạm đơn điệu E7b) và văn bản dẫn nhập Mục 4.8.
    - **Trang 70:** Bảng 4.7 (Bảng tổng hợp lớn toàn diện từ $E_0$ đến $E_7$) đứng trọn trang, cân đối đẹp mắt.
    - **Trang 71:** Ghép đôi Hình 4.13 (Biểu đồ cột so sánh hiệu năng) và Hình 4.14 (Quan hệ đánh đổi Macro-F1 vs Class MAE).
    - **Trang 72:** Hình 4.15 (Sai số nhảy 2 bước: Bỏ sót ca nặng $C_2^+ \to C_0$ và Báo động giả $C_0 \to C_2^+$) đứng độc lập, rõ nét.
    - **Trang 73:** Ghép đôi Hình 4.16 (Ma trận nhầm lẫn của 4 mô hình $E_0, E_1, E_5, E_{6b}$) và Hình 4.17 (Ma trận nhầm lẫn của 3 mô hình Ordinal $E_{7a}, E_{7b}\text{-LR}, E_{7b}\text{-RF}$).
    - **Trang 74:** Văn bản phân tích ma trận chi phí Mục 4.9 và Hình 4.18 (Đường cong đánh đổi $C_2^+$ Recall vs $C_0$ False Alarm Rate).
    - **Trang 75:** Bảng 4.8 (Ngoại suy thời gian Temporal Evaluation) và phân tích Mục 4.10.
- **Kết quả trực quan:** Mạch đọc tự nhiên, đồ thị và bảng biểu xuất hiện ngay sau khi được đề cập trong văn bản, hoàn toàn không có trang trắng bất thường.

---

## 3. KIỂM ĐỊNH KỸ THUẬT TỰ ĐỘNG

| Tiêu chí kiểm định | Phương pháp kiểm tra | Kết quả đạt được | Đánh giá |
| :--- | :--- | :---: | :---: |
| Ký hiệu tham chiếu gãy | Regex scan chuỗi `??` trên text trích xuất PyMuPDF | 0 lần | ĐẠT CHUẨN |
| Tỷ lệ font Monospace | Duyệt cây font toàn bộ 91 trang PDF | 0,00% (0 ký tự) | ĐẠT CHUẨN |
| Lỗi tràn lề ngang (`Overfull \hbox`) | Phân tích file log biên dịch của Tectonic | 0 lỗi nghiêm trọng | ĐẠT CHUẨN |
| Tỷ lệ tràn lề float | Kiểm tra vị trí bottom/top của float so với lề trang | Nằm trong lề 2,5cm | ĐẠT CHUẨN |
| Độ phân giải hình ảnh | Kiểm tra metadata file PNG nhúng vào source | 300 DPI (sắc nét) | ĐẠT CHUẨN |

---

## 4. KẾT LUẬN

Tập tin PDF `DepSev_ML_Scientific_Report_V11.pdf` đã đạt độ hoàn thiện cao nhất về mặt hình thức chế bản:
- Dàn trang chuẩn mực của một báo cáo khoa học hàn lâm chuyên ngành Machine Learning.
- Không có lỗi kỹ thuật LaTeX, không gãy liên kết, không lỗi font.
- Tối ưu hóa trải nghiệm đọc cho giảng viên phản biện.

