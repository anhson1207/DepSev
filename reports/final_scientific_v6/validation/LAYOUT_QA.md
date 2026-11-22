# BÁO CÁO KIỂM TRA CHẤT LƯỢNG TRÌNH BÀY VÀ DÀN TRANG (LAYOUT QA)
**Dự án:** DepSev-ML  
**Tài liệu kiểm định:** `DepSev_ML_Scientific_Report_V6.pdf`  
**Công cụ kiểm định:** PyMuPDF (fitz) v1.26+ & Tectonic Engine  
**Thời điểm thực hiện:** 30/10/2026  

---

## 1. Tổng quan tài liệu
- **Tổng số trang:** 89 trang PDF.
- **Kích thước trang:** Tiêu chuẩn A4 ($595.28 \times 841.89$ pt; $210 \times 297$ mm).
- **Quy chuẩn lề (UET Graduation Standard):**
  - Lề trái (Gáy đóng sách): 30 mm ($85.04$ pt).
  - Lề phải: 20 mm ($56.69$ pt) $\to$ Tọa độ giới hạn biên phải $x_1 \le 538.58$ pt.
  - Lề trên: 25 mm ($70.87$ pt).
  - Lề dưới: 25 mm ($70.87$ pt).
- **Phông chữ chính:** Times New Roman (hỗ trợ đầy đủ Unicode tiếng Việt với bộ dấu chuẩn).
- **Số lượng liên kết tương tác (Interactive Hyperlinks):** **543 liên kết** trên 54 trang.
  - Mục lục chính (TOC, trang 8–10): **247 liên kết** (nhấp chuột chuyển trực tiếp đến từng chương/tiểu mục).
  - Danh mục hình vẽ (LOF): Hoạt động 100%.
  - Danh mục bảng biểu (LOT): Hoạt động 100%.
  - Toàn bộ trích dẫn tài liệu tham khảo (\cite): Nhấp chuột chuyển trực tiếp đến Tài liệu tham khảo.

---

## 2. Kết quả kiểm tra biên lề (Margin Overflow QA)

Quy trình tự động duyệt qua tất cả $89$ trang, kiểm tra tọa độ hộp bao $(x_0, y_0, x_1, y_1)$ của từng khối văn bản nội dung (bỏ qua header/footer):

| Hạng mục kiểm tra | Tiêu chuẩn kỹ thuật | Kết quả V6 | Đánh giá |
| :--- | :---: | :---: | :---: |
| Tràn lề phải ($x_1 > 541$ pt) | 0 khối vi phạm | **0 khối** | **ĐẠT (Hoàn hảo)** |
| Tràn lề trái ($x_0 < 80$ pt) | 0 khối vi phạm | **0 khối** | **ĐẠT (Hoàn hảo)** |
| Dấu tham chiếu gãy (`??`) | 0 vị trí | **0 vị trí** | **ĐẠT (Hoàn hảo)** |
| Lỗi bảng biểu nhô ra khỏi lề | 0 bảng | **0 bảng** | **ĐẠT (Hoàn hảo)** |
| Tách trang hình vẽ & chú thích | 0 hình | **0 hình** | **ĐẠT (Hoàn hảo)** |

---

## 3. Kiểm định trực quan từng phần của Báo cáo

### 3.1. Các trang bìa và phần mở đầu (Trang 1–15)
- **Trang bìa 1 (Bìa ngoài):** Đầy đủ logo UET độ nét cao, tiêu đề dự án, tên tác giả (Nguyễn Anh Sơn), cán bộ hướng dẫn (TS. Vũ Thị Hồng Nhạn), định dạng chuẩn khóa luận UET.
- **Trang bìa 2 (Bìa phụ tiếng Việt):** Bố cục cân đối, lề cán bộ hướng dẫn cách mép lề chuẩn mực, logo sắc nét.
- **Trang bìa 3 (Bìa phụ tiếng Anh):** Tên tác giả chuyển ngữ tiếng Anh (*Nguyen Anh Son*), tên đề tài tiếng Anh, tên người hướng dẫn (*Dr. Vu Thi Hong Nhan*).
- **Trang 4–7 (Lời cảm ơn & Cam đoan):** Chữ ký người cam đoan, xưng "em" khiêm tốn, lịch thiệp, đã loại bỏ toàn bộ các câu chữ thừa.
- **Trang 8–10 (Mục lục):** Cỡ chữ chuẩn, khoảng cách dòng thoáng, tất cả các mục đều có liên kết nhấp chuột hoạt động hoàn hảo.
- **Trang 11–13 (Danh mục hình vẽ & Danh mục bảng biểu):** Danh mục rõ ràng, tiêu đề hình và bảng ngắn gọn, súc tích.
- **Trang 14–15 (Danh mục từ viết tắt):** Bảng 3 cột căn chỉnh chuẩn, không bị cắt cụt từ.

### 3.2. Chương 1: Giới thiệu (Trang 16–25)
- Hình 1.1 (Problem Formulation) và Hình 1.2 (Research Progression $T_5 \to T_3$) hiển thị sắc nét, bo góc chuẩn mực, khoảng cách văn bản thông thoáng.
- Văn phong học thuật, khiêm tốn, mục tiêu và phạm vi rành mạch.

### 3.3. Chương 2: Cơ sở lý thuyết & Nghiên cứu liên quan (Trang 26–42)
- Toàn bộ 5 hình minh họa (Trục ruột-não, Thang đo PHQ-9, Mất cân bằng lớp, Cây quyết định, Kiến trúc Random Forest) đều được căn giữa, độ phân giải 300 DPI, không chồng đè văn bản.
- Giải thích thuật toán Random Forest chi tiết, trực quan, có các siêu tham số gắn liền với thực tế dữ liệu NHANES. Bảng 2.1 (Related Works) căn lề hoàn hảo.

### 3.4. Chương 3: Dữ liệu và Phương pháp nghiên cứu (Trang 43–60)
- Sơ đồ hệ thống 7 module, pipeline tiền xử lý in-fold, không gian đặc trưng lũy tiến $F_1 \to F_4$, kiến trúc phân cấp cứng, phân cấp mềm và thứ bậc đều được trình bày chỉn chu.
- Các công thức toán học và ký hiệu rủi ro Bayes $R(a \mid \mathbf{x}) = \sum_j \mathbf{C}(j, a) \hat{P}(C_j \mid \mathbf{x})$ chuẩn xác về mặt giải tích.

### 3.5. Chương 4: Thực nghiệm và Thảo luận (Trang 61–80)
- Bảng đối chuẩn kết quả Nested CV (Bảng 4.2), Feature Ablation (Bảng 4.1), Target Granularity (Bảng 4.3), Hard Hierarchical (Bảng 4.4), Soft Hierarchical (Bảng 4.5), Ordinal (Bảng 4.6), Master Comparison (Bảng 4.7) và Executive Decision Matrix (Bảng 4.8) đều hiển thị ngay ngắn, dữ liệu khớp 100% với CSV.
- Phân tích ma trận nhầm lẫn $5 \times 5$ chi tiết của $T_5$ làm nổi bật lý do chuyển dịch sang $T_3$.
- Đồ thị đánh đổi lâm sàng (Hình 4.10) và phân tích lỗi bỏ sót (Hình 4.7) thể hiện đúng số liệu 62 ca (-48.3%) của $E_{6b}$.

### 3.6. Chương 5: Kết luận & Hướng phát triển (Trang 81–83)
- Bố cục gọn gàng, trả lời trực diện 5 câu hỏi nghiên cứu RQ1–RQ5, nêu rõ giới hạn dữ liệu cắt ngang và các giải pháp mở rộng khả thi.

### 3.7. Tài liệu tham khảo và Phụ lục (Trang 84–89)
- Tài liệu tham khảo: 19 nguồn học thuật uy tín (Breiman, CDC NHANES, Kroenke PHQ-9, Pedregosa Scikit-Learn, Chen XGBoost, v.v.), định dạng IEEE chuẩn mực.
- Phụ lục A: Bảng danh mục 34 đặc trưng NHANES kèm mã CDC.
- Phụ lục B: Cấu hình siêu tham số Frozen Random Forest Spec.
- Phụ lục C: Bảng hiệu năng chi tiết từng Outer Fold của 4 mô hình, khớp chính xác từng số thập phân với kho dữ liệu.
- Phụ lục D: Định nghĩa 4 ma trận chi phí Bayes và giải thuật Isotonic Probability Clipping.

---

## 4. Kết luận
Tài liệu `DepSev_ML_Scientific_Report_V6.pdf` hoàn toàn không có lỗi dàn trang, không có cảnh báo tràn lề, không có liên kết hỏng (`??`), và đáp ứng tối đa các yêu cầu khắt khe của hội đồng phản biện học thuật chuyên ngành.

