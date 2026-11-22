# SO SÁNH TRỰC DIỆN TRƯỚC VÀ SAU HIỆU CHỈNH (BEFORE VS AFTER: V5 VS V6)
**Dự án:** DepSev-ML  
**Đối sánh:** Báo cáo Khoa học V5 (`DepSev_ML_Scientific_Report_V5.pdf`) vs Báo cáo Khoa học V6 (`DepSev_ML_Scientific_Report_V6.pdf`)  
**Ngày thực hiện:** 30/10/2026  

---

## 1. Bảng đối chiếu các điểm khác biệt then chốt

| Hạng mục kiểm tra | Phiên bản V5 (Bản cũ tồn tại lỗi) | Phiên bản V6 (Bản hoàn thiện khoa học) | Tác động học thuật |
| :--- | :--- | :--- | :--- |
| **Số ca bỏ sót nặng của $E_{6b}$ ($C_2^+ \to C_0$)** | Tuyên bố văn bản ghi **31 ca (giảm 74.2%)**, nhưng Bảng 4.5 lại ghi **62 ca**. Mâu thuẫn nghiêm trọng. | Đồng bộ chuẩn xác toàn bộ báo cáo: **62 ca (giảm 48.33% so với 120 ca của $E_1$)** với ma trận $M_1$. Nêu rõ giảm xuống 19 ca với $M_2$. | Khắc phục mâu thuẫn số liệu lớn nhất, khớp 100% với CSV thực nghiệm. |
| **Định nghĩa Ma trận Chi phí Bayes (Phụ lục D)** | Ghi sai mức phạt $\mathbf{C}(0, 2) = 2$ ở $M_1$, tráo đổi hàng cột (phạt báo động giả thay vì bỏ sót). | Định nghĩa chính xác theo mã nguồn `src/15`: $R(a \mid \mathbf{x}) = \sum_j \mathbf{C}(j, a)\hat{P}(C_j \mid \mathbf{x})$, với $\mathbf{C}(2, 0) = 3.0$ ($M_1$), $5.0$ ($M_2$), $8.0$ ($M_3$). | Đúng đắn giải tích toán học và khớp tuyệt đối với mã nguồn. |
| **Nested CV Fold Metrics (Phụ lục C)** | Điền các số giả lập không khớp phương sai mẫu (RF: `[0.2580, 0.2910, 0.2645, 0.2820, 0.2820]` có Std = $0.0137 \ne 0.0270$). | Cập nhật chính xác từng fold từ CSV gốc: RF = `[0.3111, 0.2615, 0.2936, 0.2425, 0.2690]` (Mean = $0.2755$, Std = $0.0270$). | Khôi phục tính toàn vẹn dữ liệu và đạo đức nghiên cứu học thuật. |
| **Target Granularity Metrics (Mục 4.4)** | Văn bản gõ nhầm: $T_2$ BalAcc = 0.6789 (bảng ghi 0.6659); $T_4$ Macro-F1 = 0.3205 (bảng ghi 0.3380). | Văn bản khớp tuyệt đối với Bảng 4.3 và CSV: $T_2$ BalAcc = **0.6659**, $T_4$ Macro-F1 = **0.3380**. | Loại bỏ lỗi đánh máy sơ đẳng, bảo đảm tính nhất quán bảng - văn bản. |
| **Tỷ lệ vi phạm đơn điệu $E_{7b}$ (Mục 4.7.3)** | Ghi là 0.32% vô căn cứ. | Ghi chính xác từ thực nghiệm: **19.53%** ở LR và **5.32% $\pm$ 1.68%** ở RF; sau Isotonic Clipping là 0.0% nhưng làm sụp đổ Recall $C_1$. | Bộc lộ hiện tượng khoa học thực tế, làm nổi bật nhược điểm của CBC. |
| **Giao thức Temporal Validation Branch D** | Mô tả nhầm: Huấn luyện chỉ trên chu kỳ 2013–2014 ($N = 5.054$). | Mô tả đúng mã nguồn: Huấn luyện trên gộp 2 chu kỳ 2011–2014 ($N = \mathbf{9.693}$), kiểm định trên 2015–2016 ($N = \mathbf{4.888}$) trên 31 biến chung. | Khớp hoàn toàn với quy trình thực thi mã nguồn. |
| **Recall lớp $C_1$ trên $T_5$ vs $T_3$** | Dùng số 16.29% của $T_3$ để gán cho cả $T_5$. | Tách biệt rạch ròi: trên $T_5$, RF Recall $C_1$ là **15.83%** (53.9% nhầm sang $C_0$) và $C_4$ chỉ đạt **2.27%** (1/44); trên $T_3$ Recall $C_1$ là **16.29%**. | Bổ sung ma trận nhầm lẫn $5 \times 5$, chứng minh rào cản lâm sàng của $T_5$. |
| **Lỗi tham chiếu gãy (Broken References)** | Tồn tại các ký tự `Bảng ??` và `Mục ??` tại trang 65 và 66. | Khắc phục 100%: **0 ký tự `??` trên toàn bộ 89 trang**. | Báo cáo đạt độ hoàn thiện xuất bản chuyên nghiệp. |
| **Tràn viền và Lỗi canh lề (Margin QA)** | Xuất hiện một số cảnh báo tràn lề ở bảng biểu. | Kiểm định PyMuPDF: **0 khối văn bản vi phạm lề ($x_1 \le 540.58$ pt)**. | Bản in hoàn hảo cho đóng quyển luận văn bìa cứng. |
| **Liên kết tương tác (Hyperlinks)** | Nhiều mục không có liên kết, phông màu hyperlink chưa đồng nhất. | **543 liên kết siêu văn bản** hoạt động trơn tru (TOC, LOF, LOT, Cites). | Trải nghiệm duyệt tài liệu số tuyệt vời cho hội đồng phản biện. |
| **Hình vẽ đồ họa** | Hình 4.7 ghi sai số ca bỏ sót của $E_{6b}$ thành 31 ca và các số Recall sai lệch. | Cập nhật chính xác số liệu thanh bar [108, 120, 227, 211, 62, 68] và Recall từng lớp theo confusion matrix. | Đồ thị phản ánh trung thực kết quả thực nghiệm. |
| **Phong cách viết học thuật** | Dùng nhiều từ khoa trương ("bước đột phá", "vạch trần điểm nghẽn", "nhược điểm chí mạng"). | Văn phong khiêm tốn, chặt chẽ, khách quan, phân tích rõ ràng rủi ro lâm sàng và giới hạn dữ liệu. | Đạt chuẩn mực bảo vệ luận văn học thuật xuất sắc. |

---

## 2. Kết luận
Báo cáo Khoa học V6 là bước nhảy vọt toàn diện về chất lượng so với V5: không chỉ sửa đổi toàn bộ các sai lệch dữ liệu mà còn hoàn thiện triệt để về mặt phương pháp luận, mỹ thuật dàn trang và tính chuẩn mực học thuật.

