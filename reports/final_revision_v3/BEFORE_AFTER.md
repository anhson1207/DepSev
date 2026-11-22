# ĐỐI CHIẾU TRƯỚC VÀ SAU CẢI TIẾN TRÌNH BÀY (BEFORE & AFTER COMPARISONS)
**Dự án:** DepSev-ML – Khung học máy dự đoán mức độ triệu chứng trầm cảm từ dữ liệu sinh lý  
**Bản gốc khảo sát:** `DepSev_ML_Final_Revised.pdf` (70 trang)  
**Bản chuẩn hóa hoàn thiện:** `DepSev_ML_Final_Polished.pdf` (77 trang)  

Tài liệu này cung cấp các đối chiếu cụ thể, tường minh giữa cách dàn trang cũ và giải pháp thiết kế học thuật mới được áp dụng trong đợt chuẩn hóa toàn diện.

---

## 1. HÌNH 3.1: SƠ ĐỒ KIẾN TRÚC TỔNG THỂ DEPSEV-ML

### Trước cải tiến (Bản v2 - Trang 21)
- **Mã nguồn:** Môi trường LaTeX thuần `\begin{picture}(420, 260)`.
- **Hiện tượng lỗi:**
  - Các khối hình chữ nhật vẽ bằng lệnh thô sơ, đường nối (`\put(x,y){\vector...}`) bị lệch góc và gãy khúc.
  - Phông chữ bên trong các khối không đồng nhất, một số nhãn dài bị tràn ra ngoài viền khung chữ nhật.
  - Hình bị trôi tự do (float placement) nhảy lên đầu trang 21, **đứng trước cả dòng tiêu đề "Chương 3"**, phá vỡ hoàn toàn cấu trúc đọc tự nhiên.
  - Bounding box bị tính sai làm lấn qua lề phải 52.1 pt.

### Sau cải tiến (Bản v3 - Trang 35)
- **Mã nguồn:** Xuất đồ họa vector/raster 300 DPI chuyên nghiệp (`depsev_overall_architecture.png`) và chèn qua môi trường chuẩn `figure[htbp]`.
- **Cải tiến đạt được:**
  - Toàn bộ 7 module chức năng (NHANES Raw Data $\to$ Preprocessing $\to$ Feature Spaces $\to$ Model Architectures $\to$ Optimization $\to$ Multi-metric Evaluation $\to$ Clinical Safety) được vẽ phân tầng chuẩn xác, có viền bo tròn, đổ bóng nhẹ, phân màu pastel chuyên nghiệp.
  - Hình được đặt ngay sau đoạn văn dẫn nhập của Mục 3.1 ở trang 35. Không bao giờ trôi ngược lên trước tiêu đề chương.
  - Tọa độ Bounding box: $x_0 = 85.04$ pt, $x_1 = 538.58$ pt (chính xác 100% bằng chiều rộng văn bản, 0 pt tràn viền).

---

## 2. BẢNG 3.3: DANH MỤC HỆ THỐNG 34 ĐẶC TRƯNG TRONG KHÔNG GIAN $F_4$

### Trước cải tiến (Bản v2 - Trang 24--26)
- **Cấu trúc:** Sử dụng `longtable` kéo dài 3 trang nhưng chỉ có dòng tiêu đề cột (`\firsthead`) ở trang đầu tiên.
- **Hiện tượng lỗi:**
  - Khi sang trang 25 và trang 26, bảng chỉ tiếp tục danh sách biến mà không có tiêu đề cột (Mã biến, Nhóm đặc trưng, Kiểu dữ liệu, Mô tả lâm sàng), khiến người đọc mất ngữ cảnh.
  - Độ rộng các cột cố định gây chật chội ở cột mô tả lâm sàng.
  - Trong Danh mục bảng (LOT), bảng bị đánh số sai thành **"Bảng 2"** thay vì "Bảng 3.3" do xung đột biến đếm giữa `longtable` và lớp `uetgraduation.cls`.

### Sau cải tiến (Bản v3 - Trang 38--40)
- **Cấu trúc:** Chuẩn hóa với `longtable` tích hợp đầy đủ `\firsthead`, `\head`, `\foot` và `\lastfoot`.
- **Cải tiến đạt được:**
  - Ở đầu mỗi trang tiếp theo (trang 39 và 40), tự động in dòng tiêu đề in nghiêng trang trọng:  
    `Bảng 3.3 (tiếp theo). Danh mục hệ thống 34 đặc trưng trong không gian F4`  
    kèm theo toàn bộ 4 tiêu đề cột được kẻ viền `\toprule` và `\midrule` chuẩn phong cách Booktabs.
  - Bề rộng các cột được phân bổ lại tối ưu: `p{2.2cm} p{2.8cm} p{2.2cm} p{7.6cm}`, tổng độ rộng nằm trọn trong lề in.
  - Sửa dứt điểm xung đột biến đếm LaTeX: Bảng hiện đúng là **"Bảng 3.3"** trong cả nội dung và Danh mục bảng (LOT trang 12).

---

## 3. BẢNG 4.7: BẢNG TỔNG HỢP SO SÁNH MASTER COMPARISON (E0 $\to$ E7)

### Trước cải tiến (Bản v2 - Trang 53)
- **Cấu trúc:** Bảng ngang dài 13 cột chứa toàn bộ các chỉ số của 10 mô hình trên một hàng ngang.
- **Hiện tượng lỗi thảm họa:**
  - Chiều rộng bảng lên tới $683.1$ pt, trong khi trang giấy A4 chỉ rộng $595.28$ pt và lề in là $538.58$ pt.
  - Bảng bị **vượt lề tới +144.5 pt**, rớt hẳn ra ngoài mép phải tờ giấy A4. Các cột quan trọng nhất như Recall $C_2^+$, Bỏ sót $C_2^+ \to C_0$, Báo động $C_0 \to C_2^+$ bị cắt cụt hoàn toàn, không thể đọc được trên bản in.

### Sau cải tiến (Bản v3 - Trang 67)
- **Cấu trúc:** Tái cấu trúc thành bảng đứng 2 phần thống nhất (`tabularx{\textwidth}`):
  - **Phần A (Overall Performance):** Trường phái, Mô hình, Macro-F1, BalAcc, Accuracy, Class MAE, QWK.
  - **Phần B (Class Recalls & Clinical Safety):** Trường phái, Mô hình, Recall $C_0$, Recall $C_1$, Recall $C_2^+$, Bỏ sót $C_2^+ \to C_0$, Báo động $C_0 \to C_2^+$.
- **Cải tiến đạt được:**
  - Giữ lại trọn vẹn 100% số liệu khoa học của cả 10 mô hình ($E_0, E_1, E_2, E_{3a}, E_{3b}, E_4, E_5, E_{6b}, E_{7a}, E_{7b}$).
  - Vừa khít hoàn hảo trên một trang đứng (Trang 67), lề phải dừng chính xác tại $538.58$ pt (0 pt tràn lề).
  - Các con số in đậm rõ ràng cho mô hình tốt nhất ở từng chỉ số (ví dụ: BalAcc cao nhất là $E_{7a}$ đạt 51,51%; Recall $C_2^+$ cao nhất là $E_{7b}$ CumBin LR đạt 72,88%; Số ca bỏ sót thấp nhất là $E_{6b}$ với chỉ 31 ca).

---

## 4. HÌNH 4.4: TRỰC QUAN HÓA KHÔNG GIAN ĐẶC TRƯNG BẰNG PCA

### Trước cải tiến (Bản v2 - Trang 42)
- Biểu đồ PCA biplot vẽ mờ nhạt, độ phân giải thấp.
- Toàn bộ nhãn trục và nhãn chú giải (Legend) dùng tiếng Anh viết tắt (`PC1`, `PC2`, `label 0, 1, 2`), cỡ chữ nhỏ li ti (6--7pt).
- Hai đám mây điểm của $C_0$ và $C_1$ hiển thị bằng màu sắc khó phân biệt khi in đen trắng.

### Sau cải tiến (Bản v3 - Trang 56)
- Vẽ lại hoàn toàn bằng mã nguồn Python 300 DPI (`overlap_pca_C0_C4.png`).
- Nhãn trục được chuẩn hóa tiếng Việt:
  - Trục hoành: `Thành phần chính 1 (PC1: 18.4% phương sai giải thích)`.
  - Trục tung: `Thành phần chính 2 (PC2: 12.1% phương sai giải thích)`.
- Chú giải rõ ràng: `Không trầm cảm (C0)`, `Nhẹ (C1)`, `Nặng (C2+)`.
- Phông chữ nâng lên 12--13pt, sử dụng bảng màu tương phản cao (Xanh dương đậm - Cam hổ phách - Đỏ thẫm) kết hợp ký hiệu hình học khác nhau (hình tròn, hình tam giác, hình dấu cộng), bảo đảm phân biệt rõ ràng ngay cả khi in thang độ xám.

---

## 5. HÌNH 4.7 VÀ HÌNH 4.8: CẶP ĐỒ THỊ ĐỐI CHIẾU HIỆU NĂNG VÀ SAI SỐ 2 BẬC

### Trước cải tiến (Bản v2 - Trang 54)
- **Hình 4.7:** Tiêu đề chú thích ghi sai lệch là *"Biểu đồ phân tán so sánh Macro-F1 và Class MAE"* trong khi bản chất là biểu đồ cột nhóm.
- **Hình 4.8:** Các nhãn số ca bên trên cột sai số 2 bậc bị quá nhỏ, không thể nhìn thấy số lượng ca bỏ sót và báo động giả.
- Khoảng cách giữa hai hình quá sát nhau, dòng ghi chú nguồn bị đè lên tiêu đề hình dưới.

### Sau cải tiến (Bản v3 - Trang 68)
- **Hình 4.7:** Sửa chú thích chính xác thành:  
  *`Hình 4.7. Biểu đồ cột nhóm so sánh Macro-F1 (cao hơn là tốt hơn) và Class MAE (thấp hơn là tốt hơn)`*
- **Hình 4.8:** Tạo lại đồ họa 300 DPI với nhãn số liệu to rõ (9pt đậm) dạng `Tổng cộng\n(Bỏ sót / Báo động)`, hiển thị trực quan số ca vi phạm nghiêm trọng (ví dụ: $E_1$ bỏ sót 156 ca, trong khi $E_{6b}$ chỉ bỏ sót 31 ca nhưng tăng báo động giả lên 1.340 ca).
- Hai hình được căn chỉnh chiều cao `0.78\textwidth`, bố trí cân đối và thanh lịch trên toàn bộ trang 68, có vạch ngăn cách nhẹ nhàng.

---

## 6. HÌNH 4.9 VÀ 4.10: MA TRẬN NHẦM LẪN OUT-OF-FOLD

### Trước cải tiến (Bản v2 - Trang 55)
- **Hiện tượng lỗi thảm họa:**
  - Nhồi nhét cả 7 ma trận nhầm lẫn ($E_0, E_1, E_5, E_{6b}, E_{7a}, E_{7b\text{-LR}}, E_{7b\text{-RF}}$) vào trong một tệp ảnh duy nhất.
  - Hình bị tràn lề phải tới **+120.4 pt**, rớt ra khỏi mép giấy in.
  - Kích thước từng ma trận quá bé, các con số trong từng ô (Confusion Matrix cells) chỉ đạt cỡ 4pt, bị nhòe vỡ hạt hoàn toàn, không thể đọc được số lượng mẫu phân loại đúng/sai.

### Sau cải tiến (Bản v3 - Trang 69)
- **Tách thành hai hình độc lập, sắc nét tuyệt đối (300 DPI):**
  - **Hình 4.9 (Nhóm Phẳng, Phân cấp cứng và Phân cấp mềm):** Bố cục lưới $2 \times 2$ cho $E_0, E_1, E_5, E_{6b}$. Số lượng mẫu trong mỗi ô được render với font Bold 15pt cực kỳ sắc nét.
  - **Hình 4.10 (Nhóm Phân loại thứ bậc Ordinal Classification):** Bố cục lưới ngang $1 \times 3$ cho $E_{7a}$ (Balanced), $E_{7b}$ (CumBin LR), $E_{7b}$ (CumBin RF). Font số trong ô đạt cỡ 14pt rõ nét.
- Toàn bộ trục tung (Nhãn thực tế $C_0, C_1, C_2^+$) và trục hoành (Nhãn dự đoán $C_0, C_1, C_2^+$) cùng thang màu (Colorbar) hiển thị chuẩn mực, trang nhã.
- Độ rộng cả hai hình đặt chính xác `0.95\textwidth`, lề phải dừng tại $538.58$ pt, không vượt 1 milimet nào.

---

## 7. CÔNG THỨC TOÁN HỌC VÀ ĐÁNH SỐ PHƯƠNG TRÌNH

### Trước cải tiến (Bản v2 - Trang 15)
- Phương trình (1.2) định nghĩa bài toán gộp lớp $T_2$ và $T_3$ có chứa các khối văn bản tiếng Việt dài ngoằng trong môi trường toán học:
  `Severe conv. / Nặng theo quy ước`
  làm phương trình bị đẩy dài quá khổ, tràn lề phải $+2.3$ pt ($x_1 = 540.9$ pt).

### Sau cải tiến (Bản v3 - Trang 16)
- Rút gọn định nghĩa toán học thành ký hiệu chuẩn:
  $$\text{Sev}_{\text{conv}} = \{C_2, C_3, C_4\}$$
- Toàn bộ diễn giải ngữ nghĩa lâm sàng được đưa vào câu văn phân tích ngay bên dưới.
- Phương trình nằm gọn trong khoảng hoành độ $[85.04, 480.0]$ pt, cách lề phải tới 58 pt, bảo đảm vẻ đẹp cân xứng chuẩn mực của văn bản khoa học.

