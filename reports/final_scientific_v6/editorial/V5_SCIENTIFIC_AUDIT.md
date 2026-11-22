# BÁO CÁO KIỂM ĐỊNH KHOA HỌC BÁO CÁO V5 VÀ PHƯƠNG ÁN KHẮC PHỤC TRÊN V6 (V5 SCIENTIFIC AUDIT)
**Dự án:** DepSev-ML  
**Đối tượng kiểm định:** `DepSev_ML_Scientific_Report_V5.pdf` và mã nguồn LaTeX V5  
**Phiên bản hoàn thiện:** `DepSev_ML_Scientific_Report_V6.pdf` (V6)  
**Ngày kiểm định:** 30/10/2026  

---

## 1. Tổng quan kiểm định
Quá trình kiểm định khoa học độc lập đã rà soát toàn bộ văn bản, công thức toán học, bảng biểu, đồ thị và số liệu thực nghiệm trong báo cáo V5 đối chiếu trực tiếp với mã nguồn thực thi trong `src/`, các run manifests, và dữ liệu kết quả trong `results/` cùng `experiments/`.

Kiểm định đã phát hiện và xử lý triệt để 8 nhóm vấn đề then chốt (P0 Issues), cùng các lỗi diễn đạt học thuật và dàn trang.

---

## 2. Chi tiết 8 vấn đề then chốt (P0 Issues) và Giải pháp xử lý

### P0.1. Mâu thuẫn số ca bỏ sót ca nặng của mô hình $E_{6b}$
- **Hiện trạng sai lệch trong V5:**
  - Trong văn bản (Chương 1 dòng 122, Chương 4 dòng 213, 300, 332, 341; Chương 5 dòng 14, 23; Bảng tóm tắt quyết định): Báo cáo V5 tuyên bố mô hình $E_{6b}$ giảm số ca bỏ sót nặng từ 120 ca ($E_1$) xuống còn **31 ca** (tương đương mức giảm **74.2%**).
  - Tuy nhiên, ngay trong Bảng 4.5 (`table_soft_hierarchical.tex`) và Bảng 4.7 (`table_master_comparison.tex`), số liệu thực tế được ghi là **62 ca** và **294 ca báo động giả**.
- **Bằng chứng dữ liệu gốc:**
  - File `experiments/soft_hierarchical_t3/outputs/confusion_matrices/E6b_Soft_RF_LR_CostSensitive_confusion_matrix.csv`: Hàng nhãn thật $C_2^+$ bị mô hình dự đoán nhầm thành $C_0$ là đúng **62 ca**.
  - File `experiments/soft_hierarchical_t3/outputs/cost_sensitivity.csv`:
    - Cấu hình $M_1$ (Moderate Penalty): $C_2^+ \to C_0 = \mathbf{62}$ ca (giảm $58/120 = \mathbf{48.33\%}$); Báo động giả $C_0 \to \{C_1, C_2^+\} = \mathbf{1.150}$ ca.
    - Cấu hình $M_2$ (High Penalty): $C_2^+ \to C_0 = \mathbf{19}$ ca (giảm $\mathbf{84.17\%}$); Báo động giả = $\mathbf{1.906}$ ca.
    - Cấu hình $M_3$ (Extreme Penalty): $C_2^+ \to C_0 = \mathbf{7}$ ca (giảm $\mathbf{94.17\%}$); Báo động giả = $\mathbf{2.294}$ ca.
- **Xử lý trong V6:**
  - Điều chỉnh toàn bộ văn bản trong Chương 1, Chương 4, Chương 5 và Bảng tóm tắt điều hành: mô hình $E_{6b}$ với ma trận chi phí $M_1$ đạt Balanced Accuracy = **0.5025**, giảm số ca bỏ sót nặng từ 120 ca xuống còn **62 ca** (giảm **48.33%**).
  - Bổ sung phân tích chuỗi nhạy chi phí $M_0 \to M_3$ tại Mục 4.8, làm rõ khả năng giảm tiếp xuống 19 ca ($M_2$) và 7 ca ($M_3$) đi kèm sự đánh đổi gia tăng số ca báo động nhầm.

---

### P0.2. Sai lệch định nghĩa Ma trận Chi phí Bayes trong Phụ lục D
- **Hiện trạng sai lệch trong V5:**
  - Phụ lục D trong V5 mô tả ma trận $M_1$ phạt lỗi bỏ sót nặng $\mathbf{C}(0, 2) = 2$, ma trận $M_2$ phạt $\mathbf{C}(0, 2) = 3$, ma trận $M_3$ phạt $\mathbf{C}(0, 2) = 5$.
  - Đồng thời, ký hiệu $\mathbf{C}(0, 2)$ bị tráo đổi hàng và cột: viết là phạt bỏ sót ca nặng nhưng lại đặt chỉ số hàng là 0 (bình thường) và cột là 2 (nặng), tức là phạt báo động giả.
- **Bằng chứng mã nguồn gốc:**
  - Trong `src/15_soft_hierarchical_experiment.py` (dòng 70–94), hàm quyết định nhân ma trận `probs @ self.cost_matrix`, trong đó hàng là Nhãn thật ($j$) và cột là Hành động dự đoán ($a$).
  - Ma trận $M_0$: Đồng nhất ($\mathbf{C}(j, a) = 1.0$ với $j \ne a$).
  - Ma trận $M_1$: $\mathbf{C}(2, 0) = 3.0$; $\mathbf{C}(2, 1) = 1.5$; $\mathbf{C}(1, 0) = 1.5$; $\mathbf{C}(0, 2) = 1.5$.
  - Ma trận $M_2$: $\mathbf{C}(2, 0) = 5.0$; $\mathbf{C}(2, 1) = 2.0$; $\mathbf{C}(1, 0) = 2.0$; $\mathbf{C}(0, 2) = 2.0$.
  - Ma trận $M_3$: $\mathbf{C}(2, 0) = 8.0$; $\mathbf{C}(2, 1) = 3.0$; $\mathbf{C}(1, 0) = 2.5$; $\mathbf{C}(0, 2) = 2.5$.
- **Xử lý trong V6:**
  - Hiệu chỉnh chuẩn hóa định nghĩa toán học trong Chương 3 (Mục 3.4.4) và Phụ lục D: $R(a \mid \mathbf{x}) = \sum_j \mathbf{C}(j, a) \hat{P}(C_j \mid \mathbf{x})$.
  - Cập nhật đúng các giá trị hệ số phạt của cả 4 ma trận chi phí.

---

### P0.3. Số liệu từng Fold của Nested Cross-Validation trong Phụ lục C
- **Hiện trạng sai lệch trong V5:**
  - Bảng C.1 (`tab:app-fold-results`) trong Phụ lục C của V5 ghi các giá trị Macro-F1 theo từng Fold không khớp với giá trị trung bình và độ lệch chuẩn mẫu. Cụ thể: Random Forest ghi `[0.2580, 0.2910, 0.2645, 0.2820, 0.2820]` có độ lệch chuẩn thực tế chỉ là $0.0137$, nhưng cột trung bình lại ghi $0.2755 \pm 0.0270$.
- **Bằng chứng dữ liệu gốc:**
  - File `results/tables/nested_cv_baseline_comparison.csv` và `results/tables/ablation_fold_results_final_protocol.csv`:
    - **Random Forest ($F_4$):** Fold 1 = 0.3111, Fold 2 = 0.2615, Fold 3 = 0.2936, Fold 4 = 0.2425, Fold 5 = 0.2690 $\to$ **Mean = 0.2755, Std = 0.0270** (khớp hoàn hảo ddof=1).
    - **Logistic Regression ($F_4$):** Fold 1 = 0.2599, Fold 2 = 0.2198, Fold 3 = 0.2566, Fold 4 = 0.2382, Fold 5 = 0.2292 $\to$ **Mean = 0.2407, Std = 0.0173** (khớp hoàn hảo).
    - **Decision Tree ($F_4$):** Fold 1 = 0.2538, Fold 2 = 0.2404, Fold 3 = 0.2634, Fold 4 = 0.2239, Fold 5 = 0.2245 $\to$ **Mean = 0.2412, Std = 0.0175** (khớp hoàn hảo).
    - **XGBoost ($F_4$):** Fold 1 = 0.3017, Fold 2 = 0.2207, Fold 3 = 0.2925, Fold 4 = 0.2341, Fold 5 = 0.2615 $\to$ **Mean = 0.2621, Std = 0.0353** (khớp hoàn hảo).
- **Xử lý trong V6:**
  - Thay thế toàn bộ số liệu Bảng C.1 bằng các giá trị thực nghiệm gốc nói trên, đảm bảo tính đúng đắn toán học 100%.

---

### P0.4. Sai lệch số liệu Target Granularity giữa Bảng và Văn bản
- **Hiện trạng sai lệch trong V5:**
  - Tại Mục 4.4 dòng 106: Văn bản ghi *"Balanced Accuracy giảm từ 0.6789 xuống 0.4667"* và *"Macro-F1 suy giảm tuyến tính xuống 0.3205 và 0.2630"*.
  - Trong khi đó, Bảng 4.3 (`tab:granularity-results`) và file `results/tables/granularity_results.csv` ghi rõ: Balanced Accuracy của $T_2$ là **0.6659 $\pm$ 0.0122** và Macro-F1 của $T_4$ là **0.3380 $\pm$ 0.0177**.
- **Xử lý trong V6:**
  - Sửa lỗi đánh máy trong văn bản thành 0.6659 và 0.3380.
  - Bổ sung lập luận học thuật làm rõ: $T_2$ và $T_3$ có không gian nhãn mục tiêu khác nhau (nhị phân vs 3 lớp), do đó bước sụt giảm phản ánh độ phức tạp ranh giới quyết định gia tăng chứ không phải là mô hình bị suy thoái trên cùng một bài toán.

---

### P0.5. Sai lệch Tỷ lệ vi phạm đơn điệu xác suất trong $E_{7b}$ (CumBin)
- **Hiện trạng sai lệch trong V5:**
  - Mục 4.7.3 dòng 229 trong V5 ghi: *"tỷ lệ vi phạm tính đơn điệu xác suất xuất hiện trên 0.32% tổng số dự đoán"*.
- **Bằng chứng dữ liệu gốc:**
  - File `experiments/ordinal_t3/outputs/e7b_monotonicity_violations.csv`:
    - $E_{7b}$ CumBin Hồi quy Logistic: Tỷ lệ vi phạm qua 5 fold lần lượt là 18.91%, 20.89%, 18.29%, 17.92%, 21.63% $\to$ **Trung bình 19.53%** (790/4.045 mẫu).
    - $E_{7b}$ CumBin Random Forest: Tỷ lệ vi phạm qua 5 fold là 2.97%, 6.43%, 7.42%, 4.57%, 5.19% $\to$ **Trung bình 5.32% $\pm$ 1.68%** (215/4.045 mẫu).
    - Sau khi áp dụng giải thuật Isotonic Clipping: Tỷ lệ vi phạm đưa về đúng **0.0%**, nhưng làm triệt tiêu hoàn toàn độ nhạy của lớp trung gian $C_1$ ($0.00\%$ ở LR và $0.16\%$ ở RF).
- **Xử lý trong V6:**
  - Thay thế số liệu 0.32% vô căn cứ bằng đúng số liệu thực tế: 19.53% (LR) và 5.32% (RF), đồng thời phân tích sâu sắc tác động phụ của giải thuật Isotonic Clipping.

---

### P0.6. Mô tả sai Giao thức kiểm định ngoại suy theo dòng thời gian (Branch D)
- **Hiện trạng sai lệch trong V5:**
  - Mục 3.6 và Mục 4.9 trong V5 mô tả tập huấn luyện chỉ gồm chu kỳ NHANES 2013–2014 ($N = 5.054$).
- **Bằng chứng mã nguồn gốc:**
  - Mã nguồn trong `src/15_soft_hierarchical_experiment.py` (dòng 624–633) và `src/16_ordinal_t3_experiment.py` (dòng 585–594) thực thi:
    - `train_d = pd.concat([frames["2011_2012"], frames["2013_2014"]], ignore_index=True)` ($N = \mathbf{9.693}$).
    - `test_d = frames["2015_2016"]` ($N = \mathbf{4.888}$).
    - Không gian đặc trưng: 31 biến hài hòa hóa ($F_{4\_\text{common31}}$).
- **Xử lý trong V6:**
  - Điều chỉnh đồng bộ mô tả giao thức tại Chương 1, Chương 3, Chương 4 và Chương 5, phản ánh chính xác cỡ mẫu huấn luyện 9.693 đối tượng.

---

### P0.7. Nhầm lẫn Recall của lớp $C_1$ giữa $T_5$ và $T_3$
- **Hiện trạng sai lệch trong V5:**
  - Báo cáo V5 dùng số liệu Recall lớp $C_1$ = **16.29%** để mô tả cả bài toán 5 lớp phẳng $T_5$ và bài toán 3 lớp $T_3$.
- **Bằng chứng dữ liệu gốc:**
  - File `results/tables/granularity_per_class_results.csv`:
    - Trên $T_5$: Recall lớp $C_1$ thực tế là **15.83%** (101/638 mẫu), và lớp nặng nhất $C_4$ chỉ phát hiện đúng duy nhất 1 mẫu (**Recall = 2.27%**).
    - Trên $T_3$ (mô hình $E_0$ Flat RF): Recall lớp $C_1$ mới là **16.29%** (104/638 mẫu).
- **Xử lý trong V6:**
  - Phân định rạch ròi hai giá trị trên; bổ sung phân tích đầy đủ ma trận nhầm lẫn $5 \times 5$ của $T_5$ tại Mục 4.4 để chứng minh rào cản lâm sàng của bài toán 5 lớp.

---

### P0.8. Lỗi tham chiếu gãy (Broken LaTeX References "Bảng ??")
- **Hiện trạng trong V5:**
  - Hai trang trong Chương 4 (trang 65 và 66) xuất hiện các dấu `Bảng ??` và `Mục ??` do nhầm lẫn tên nhãn:
    - Gọi `\ref{tab:hard-hierarchical}` trong khi nhãn khai báo là `\label{tab:hard-hierarchical-results}`.
    - Gọi `\ref{tab:soft-hierarchical}` trong khi nhãn khai báo là `\label{tab:soft-cost-results}`.
    - Gọi `\ref{subsec:clinical-cost-tradeoff}` trong khi nhãn khai báo là `\label{sec:danh-doi-lam-sang}`.
- **Xử lý trong V6:**
  - Sửa đổi chính xác toàn bộ các thẻ tham chiếu. Kết quả kiểm tra bằng PyMuPDF xác nhận **0 lỗi tham chiếu gãy trên toàn bộ 89 trang**.

