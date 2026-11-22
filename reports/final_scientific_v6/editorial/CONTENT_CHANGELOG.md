# NHẬT KÝ THAY ĐỔI NỘI DUNG (CONTENT CHANGELOG: V5 -> V6)
**Dự án:** DepSev-ML  
**Tệp nguồn:** `reports/final_scientific_v5/` -> `reports/final_scientific_v6/`  
**Ngày cập nhật:** 30/10/2026  

---

## 1. Thay đổi tại các Phần mở đầu (Front Matter)
- **Lời cam đoan & Lời cảm ơn:**
  - Chuẩn hóa đại từ nhân xưng thành "em" khiêm tốn, lịch thiệp, phù hợp phong cách bảo vệ học thuật.
  - Loại bỏ hoàn toàn các nhắc đến hỗ trợ công nghệ/AI.
  - Trang bìa 3 tiếng Anh: Chuẩn hóa tên tác giả thành *Nguyen Anh Son*, người hướng dẫn thành *Dr. Vu Thi Hong Nhan*.
- **Tóm tắt (Vietnamese Summary) & Abstract (English):**
  - Đồng bộ số liệu bỏ sót của mô hình $E_{6b}$ thành **62 ca** (thay vì 31 ca).
  - Khẳng định tính ổn định ngoại suy của $E_{7a}$ (Balanced OLR) trên chu kỳ tương lai $N = 4.888$ (Balanced Accuracy = $0.5073$).
  - Nhấn mạnh giới hạn dữ liệu cắt ngang và bản chất gián tiếp của chỉ số sinh lý ngoại vi.
- **Mục lục, Danh mục hình vẽ & Danh mục bảng biểu:**
  - Kích hoạt **543 liên kết siêu văn bản tương tác** (clickable hyperlinks) có màu sắc học thuật trang nhã.

---

## 2. Thay đổi tại Chương 1: Giới thiệu
- **Mục 1.4 (Tiến trình nghiên cứu hai giai đoạn $T_5 \to T_3$):**
  - Dẫn dắt mạch lạc quá trình nghiên cứu: Giai đoạn 1 khảo sát 5 bậc chuẩn PHQ-9 ($T_5$), làm rõ sự suy thoái độ nhạy trên các lớp triệu chứng (Recall $C_1 = 15.83\%$, Recall $C_4 = 2.27\%$).
  - Dẫn dắt sang Giai đoạn 2 tái cấu trúc sang bài toán phân tầng 3 nhóm lâm sàng $T_3$, giải thích nguyên lý huấn luyện Stage 2 chỉ trên các mẫu có nhãn thật $\text{PHQ-9} \ge 5$.
  - Nhấn mạnh nguyên tắc: $T_3$ là một bài toán phân loại độc lập với không gian nhãn được tái cấu trúc, không phải là kết quả cải thiện trực tiếp trên cùng không gian 5 lớp.
- **Mục 1.5 (Các đóng góp chính):**
  - Sửa đổi số ca bỏ sót ca nặng của $E_{6b}$ thành **62 ca** (giảm 48.33% với $M_1$), nêu rõ tiềm năng giảm tiếp xuống 19 ca với $M_2$.
  - Thay đổi ngôn từ khoa trương ("vạch trần điểm nghẽn chí mạng") thành văn phong khoa học chuẩn mực.

---

## 3. Thay đổi tại Chương 2: Cơ sở lý thuyết & Nghiên cứu liên quan
- **Mục 2.3 (Khảo sát thuật toán & Cơ sở lý thuyết Random Forest):**
  - Khảo sát súc tích 4 thuật toán cơ sở: Hồi quy Logistic, Cây quyết định, Random Forest, XGBoost (không viết thành giáo trình, không đưa công thức rườm rà).
  - Trình bày lý do khách quan, khiêm tốn khi lựa chọn Random Forest làm mô hình cơ sở dựa trên kết quả Nested CV (Macro-F1 = $0.2755 \pm 0.0270$).
  - Tập trung giải thích chuyên sâu và trực quan về Random Forest: nguyên lý Cây quyết định, phân tách nhị phân theo chỉ số Gini, hạn chế phương sai cao của cây đơn lẻ, kỹ thuật Bagging, phân tích mẫu Out-Of-Bag (OOB $\approx 36.8\%$), cơ chế ngẫu nhiên hóa đặc trưng (Random Subspace) và công thức giảm phương sai $\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$.
  - Làm rõ quy trình suy luận của một quan sát bệnh nhân qua 150 cây quyết định và cơ chế Soft Voting.
  - Phân tích chi tiết vai trò của từng siêu tham số trong bộ thông số đóng băng (*Frozen Random Forest Spec*).
- **Mục 2.4 (Các trường phái mô hình hóa):**
  - Chuẩn hóa ký hiệu hàm rủi ro Bayes: $R(a \mid \mathbf{x}) = \sum_j \mathbf{C}(j, a) \hat{P}(C_j \mid \mathbf{x})$.

---

## 4. Thay đổi tại Chương 3: Dữ liệu và Phương pháp nghiên cứu
- **Mục 3.4.4 (Lý thuyết quyết định nhạy chi phí Bayes):**
  - Chuẩn hóa ký hiệu phần tử ma trận chi phí $\mathbf{C}(j, a)$ với $j$ là nhãn thật và $a$ là hành động dự đoán, phạt lỗi bỏ sót nặng $\mathbf{C}(2, 0) = 3.0$ ở $M_1$ (không bị nhầm với $\mathbf{C}(0, 2)$).
- **Mục 3.6.2 (Giao thức kiểm định dòng thời gian Branch D):**
  - Sửa đổi mô tả giao thức đúng với mã nguồn: Huấn luyện trên gộp 2 chu kỳ NHANES 2011–2014 ($N = 9.693$), kiểm định ngoại suy trên chu kỳ tương lai NHANES 2015–2016 ($N = 4.888$) với không gian 31 đặc trưng chung ($F_{4\_\text{common31}}$).

---

## 5. Thay đổi tại Chương 4: Thực nghiệm và Thảo luận
- **Mục 4.4 (Phân tích Target Granularity):**
  - Sửa lỗi đánh máy: Balanced Accuracy của $T_2$ là **0.6659** (V5 ghi nhầm 0.6789), Macro-F1 của $T_4$ là **0.3380** (V5 ghi nhầm 0.3205).
  - Bổ sung phân tích ma trận nhầm lẫn chi tiết $5 \times 5$ của mô hình phẳng $T_5$ (3.016 mẫu $C_0$, 638 mẫu $C_1$, 243 mẫu $C_2$, 104 mẫu $C_3$, 44 mẫu $C_4$).
  - Chỉ ra sự sụp đổ lâm sàng: lớp $C_4$ chỉ phát hiện đúng 1/44 mẫu (Recall = 2.27%), lớp $C_1$ bị nhầm sang $C_0$ tới 53.92%.
- **Mục 4.6 & 4.7 (Kết quả Phân cấp cứng & Phân cấp mềm):**
  - Khắc phục triệt để lỗi tham chiếu gãy `Bảng ??` bằng cách đồng bộ đúng nhãn `\ref{tab:hard-hierarchical-results}` và `\ref{tab:soft-cost-results}`.
  - Sửa số liệu $E_{6b}$ ($M_1$): giảm số ca bỏ sót nặng từ 120 ca ($E_1$) xuống còn **62 ca** (giảm 48.33%).
  - Sửa số liệu vi phạm đơn điệu của $E_{7b}$ (CumBin): **19.53%** ở LR và **5.32% $\pm$ 1.68%** ở RF (thay cho số liệu 0.32% vô căn cứ).
- **Mục 4.8 (Phân tích đánh đổi lâm sàng):**
  - Sửa đổi tham chiếu `Mục ??` thành `\ref{sec:danh-doi-lam-sang}`.
  - Trình bày toàn diện chuỗi đánh đổi chi phí:
    - $M_0$ (Đồng nhất): Bỏ sót 211 ca, Báo động giả 223 ca.
    - $M_1$ (Vừa phải): Bỏ sót 62 ca, Báo động giả 1.150 ca (294 ca xếp vào $C_2^+$).
    - $M_2$ (Nghiêm ngặt): Bỏ sót 19 ca, Báo động giả 1.906 ca.
    - $M_3$ (Kịch khung): Bỏ sót 7 ca, Báo động giả 2.294 ca.
- **Mục 4.9 & 4.10 (Kiểm định dòng thời gian & Thảo luận tổng hợp):**
  - Sửa mô tả cỡ mẫu huấn luyện Branch D thành $N = 9.693$.
  - Cập nhật Bảng đối chuẩn quyết định (Executive Decision Matrix) phản ánh đúng số liệu 62 ca (-48.3%) và 98 ca của $E_2$.

---

## 6. Thay đổi tại Chương 5: Kết luận & Hướng phát triển
- Cập nhật số liệu và kết luận cho RQ4 và RQ5 tương thích 100% với Chương 4.
- Nêu rõ các khuyến nghị ứng dụng thực tế theo 4 mô hình tiêu biểu (The 4 Pillar Models).

---

## 7. Thay đổi tại Tài liệu tham khảo và Phụ lục
- **Tài liệu tham khảo:** Chuẩn hóa toàn bộ 19 mục tham khảo theo chuẩn IEEE, xóa các trích dẫn thừa.
- **Phụ lục C:** Cập nhật bảng Macro-F1 từng Fold cho cả 4 mô hình (RF, LR, DT, XGB) khớp chính xác tuyệt đối với kho dữ liệu thực nghiệm `results/tables/nested_cv_baseline_comparison.csv`.
- **Phụ lục D:** Định nghĩa chính xác 4 ma trận chi phí Bayes $M_0 \to M_3$ và giải thuật Isotonic Probability Clipping.

