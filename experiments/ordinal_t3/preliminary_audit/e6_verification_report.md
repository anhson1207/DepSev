# BÁO CÁO KIỂM ĐỊNH ĐỘC LẬP KẾT QUẢ THỰC NGHIỆM E4–E6 (SOFT HIERARCHICAL & COST-SENSITIVE)

**Dự án:** DepSev-ML — A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Chủ thể thực hiện:** Senior Machine Learning Research Engineer (DepSev-ML Team)  
**Ngày kiểm định:** 27 tháng 10 năm 2026  
**Thư mục lưu trữ:** [`experiments/ordinal_t3/preliminary_audit/`](file:///Users/nguyenanhson/DepSev-ML/experiments/ordinal_t3/preliminary_audit/)  
**Tập dữ liệu kiểm định:** Legacy Development Cohort (NHANES 2013–2014, $N=4,045$, 5-Fold Stratified CV)

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH KIỂM ĐỊNH

Trước khi triển khai hướng nghiên cứu mới **E7 – Ordinal Classification**, đợt kiểm định này được tiến hành nhằm xác minh độc lập:
1. Tính chuẩn xác toán học của công thức kết hợp xác suất (Probability Fusion) trong Soft Hierarchical (E4, E5).
2. Quy trình và nguồn gốc của các ma trận chi phí ($M_0, M_1, M_2, M_3$) trong Cost-Sensitive Decision Making (E6a, E6b).
3. Đảm bảo **Zero Data Leakage**: Xác minh quy tắc lựa chọn ma trận chi phí có thực hiện qua Inner CV hay dùng kết quả Outer Test.
4. Đối chiếu số liệu chi tiết về sự đánh đổi giữa việc giảm ca bệnh bỏ sót ($C_2^+ \to C_0$) và việc tăng báo động giả ($C_0 \to C_1/C_2^+$).
5. Phân định rõ kết quả nào đủ điều kiện làm baseline chuẩn và kết quả nào cần lưu ý cảnh báo học thuật.

---

## 2. KIỂM ĐỊNH TÍNH CHUẨN XÁC CỦA CÔNG THỨC XÁC SUẤT PHÂN CẤP (E4, E5)

### 2.1. Xác minh công thức toán học
Công thức kết hợp xác suất được triển khai trong `src/15_soft_hierarchical_experiment.py`:
$$\begin{aligned}
P(C_0 \mid X) &= 1 - p_1 \\
P(C_1 \mid X) &= p_1 \cdot (1 - p_2) \\
P(C_2^+ \mid X) &= p_1 \cdot p_2
\end{aligned}$$
với $p_1 = P(\text{Symptomatic} = 1 \mid X)$ từ Model 1 và $p_2 = P(\text{Severe} = 1 \mid \text{Symptomatic} = 1, X)$ từ Model 2.

### 2.2. Kết quả kiểm tra dữ liệu thực tế trên 4,045 mẫu Out-of-Fold
- **Khoảng giá trị xác suất:**
  - $E_4$ (Soft RF–RF): $P(C_k \mid X) \in [0.00657, 0.95539] \subset [0, 1]$.
  - $E_5$ (Soft RF–LR): $P(C_k \mid X) \in [0.00050, 0.95539] \subset [0, 1]$.
- **Tổng xác suất:**
  - Cực tiểu: $0.9999999999999997$
  - Cực đại: $1.0000000000000002$
  - Sai số lệch so với $1.0$ nhỏ hơn $3 \times 10^{-16}$ (sai số biểu diễn dấu phẩy động chuẩn IEEE 754).
- **Thứ tự lớp:** Ánh xạ nhãn cột chuẩn hóa chính xác $\{0: C_0, 1: C_1, 2: C_2^+\}$.
- **Tính độc lập của Stage 2:** Model 2 nhận đầu vào là toàn bộ ma trận đặc trưng $X$ của tập kiểm tra, không truy cập nhãn thật $y_{\text{true}}$, bảo đảm suy luận xác suất liên tục không bị gate cứng.

> [!NOTE]
> **Đánh giá kiểm định:** **PASS HOÀN TOÀN**. Công thức xác suất hợp lệ về mặt toán học và không có rò rỉ dữ liệu.

---

## 3. KIỂM ĐỊNH CÁC MA TRẬN CHI PHÍ (COST MATRICES) VÀ QUY TẮC CHỌN TRONG E6

### 3.1. Cấu trúc và ý nghĩa ngữ nghĩa của các ma trận chi phí
Các ma trận chi phí $K \in \mathbb{R}^{3 \times 3}$ được định nghĩa theo quy ước:
- **Hàng $i \in \{0, 1, 2\}$:** Nhãn thực tế ($0: C_0, 1: C_1, 2: C_2^+$).
- **Cột $j \in \{0, 1, 2\}$:** Nhãn dự đoán ($0: C_0, 1: C_1, 2: C_2^+$).
- **Đường chéo $K[i, i] = 0.0$:** Dự đoán đúng không bị phạt chi phí.

```text
1. M0_Uniform (0-1 Loss):
   [[0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]]
   --> Tương đương toán học với quy tắc argmax(P).

2. M1_Moderate (Phạt vừa phải bỏ sót ca nặng):
   [[0.0, 1.0, 1.5],
    [1.5, 0.0, 1.0],
    [3.0, 1.5, 0.0]]
   --> Phạt lỗi bỏ sót ca nặng (C2+ -> C0) gấp 3 lần báo động giả thông thường.

3. M2_High (Phạt nặng bỏ sót ca nặng):
   [[0.0, 1.0, 2.0],
    [2.0, 0.0, 1.0],
    [5.0, 2.0, 0.0]]
   --> Phạt lỗi bỏ sót ca nặng gấp 5 lần.

4. M3_Extreme (Phạt cực đoan bỏ sót ca nặng):
   [[0.0, 1.0, 2.5],
    [2.5, 0.0, 1.0],
    [8.0, 3.0, 0.0]]
   --> Phạt lỗi bỏ sót ca nặng gấp 8 lần.
```

---

### 3.2. Quy trình lựa chọn ma trận chi phí: Có Data Leakage không?
Chúng tôi đã kiểm tra hàm `optimize_cost_matrix_inner_cv()` tại dòng 197 của `src/15_soft_hierarchical_experiment.py`:
- **Cơ chế:** Trong mỗi outer fold, thuật toán chạy **Stratified 3-Fold CV lồng bên trong tập huấn luyện** ($X_{\text{train}}, s_{\text{train}}$).
- **Tiêu chí tối ưu:** Chọn ma trận chi phí có **Inner Macro-F1 cao nhất** trên out-of-fold của inner CV.
- **Kết quả lựa chọn thực tế:**
  - Ở cả 5 fold của `E6a` (RF–RF), Inner CV đều độc lập chọn **`M1_Moderate`**.
  - Ở cả 5 fold của `E6b` (RF–LR), Inner CV đều độc lập chọn **`M1_Moderate`**.
  - Các ma trận $M_2$ và $M_3$ **không được chọn làm mô hình chính E6** vì dù chúng giảm ca bỏ sót, nhưng lượng báo động giả quá lớn đã làm tụt Inner Macro-F1.

> [!IMPORTANT]
> **Kết luận kiểm định:** **KHÔNG CÓ DATA LEAKAGE**.
> - Ma trận `M1_Moderate` của E6a/E6b được chọn **hoàn toàn tự động thông qua Inner CV**, không sử dụng Outer Validation hay Test set.
> - Các kết quả của $M_2$ và $M_3$ được lưu lại trong `cost_sensitivity.csv` là một cuộc khảo sát độ nhạy tham số (Sensitivity Analysis / Pareto Curve), không phải chọn lọc thiên vị sau thực nghiệm.

---

## 4. BẢNG ĐỐI CHIẾU CHI TIẾT SỰ ĐÁNH ĐỔI GIỮA M0, M1, M2, M3

Dưới đây là bảng thống kê đối chiếu chi tiết số lượng mẫu và hiệu năng trên toàn bộ 4,045 đối tượng kiểm tra Out-of-Fold:

### Bảng 1: Phân tích đánh đổi trên nền RF + Logistic Regression (RF–LR)

| Ma trận chi phí | Trọng số phạt $C_2^+ \to C_0$ | Macro-F1 | Balanced Acc | Accuracy | Recall C1 | Recall C2+ | Precision C2+ | Lỗi $C_2^+ \to C_0$ (Bỏ sót nặng) | Lỗi $C_2^+ \to C_1$ (Đoán dưới mức) | Báo động giả $C_0 \to C_1$ | Báo động giả $C_0 \to C_2^+$ | Tổng báo động giả $C_0$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0 (Argmax)** | 1.0 | $0.4270 \pm 0.0118$ | $0.4506 \pm 0.0104$ | **$73.37\%$** | $3.45\%$ | $39.13\%$ | **$34.30\%$** | 211 / 391 (54.0%) | 27 ca | 47 ca | 176 ca | **223 ca (7.4%)** |
| **M1 (Moderate - E6b)**| **3.0** | **$0.4521 \pm 0.0190$** | **$0.5025 \pm 0.0218$** | $56.84\%$ | **$34.64\%$** | **$54.23\%$** | $31.22\%$ | **62 / 391 (15.9%)** | 117 ca | 856 ca | 294 ca | **1,150 ca (38.1%)** |
| **M2 (High)** | 5.0 | $0.3875 \pm 0.0215$ | $0.4776 \pm 0.0363$ | $41.19\%$ | $56.59\%$ | $49.88\%$ | $31.73\%$ | **19 / 391 (4.9%)** | 177 ca | 1,646 ca | 260 ca | **1,906 ca (63.2%)** |
| **M3 (Extreme)** | 8.0 | $0.3418 \pm 0.0187$ | $0.4750 \pm 0.0282$ | $33.08\%$ | **$61.76\%$** | **$56.79\%$** | $30.81\%$ | **7 / 391 (1.8%)** | 162 ca | 1,978 ca | 316 ca | **2,294 ca (76.1%)** |

### Bảng 2: Phân tích đánh đổi trên nền RF + Random Forest (RF–RF)

| Ma trận chi phí | Trọng số phạt $C_2^+ \to C_0$ | Macro-F1 | Balanced Acc | Accuracy | Recall C1 | Recall C2+ | Precision C2+ | Lỗi $C_2^+ \to C_0$ (Bỏ sót nặng) | Lỗi $C_2^+ \to C_1$ (Đoán dưới mức) | Báo động giả $C_0 \to C_1$ | Báo động giả $C_0 \to C_2^+$ | Tổng báo động giả $C_0$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M0 (Argmax)** | 1.0 | $0.4206 \pm 0.0104$ | $0.4309 \pm 0.0090$ | **$73.75\%$** | $5.18\%$ | $30.19\%$ | **$33.97\%$** | 227 / 391 (58.1%) | 46 ca | 45 ca | 139 ca | **184 ca (6.1%)** |
| **M1 (Moderate - E6a)**| **3.0** | **$0.4535 \pm 0.0149$** | **$0.4904 \pm 0.0165$** | $56.91\%$ | **$40.92\%$** | **$44.25\%$** | $33.16\%$ | **62 / 391 (15.9%)** | 156 ca | 933 ca | 215 ca | **1,148 ca (38.1%)** |
| **M2 (High)** | 5.0 | $0.3898 \pm 0.0152$ | $0.4756 \pm 0.0255$ | $40.87\%$ | $62.87\%$ | $43.99\%$ | $35.48\%$ | **19 / 391 (4.9%)** | 200 ca | 1,742 ca | 194 ca | **1,936 ca (64.2%)** |
| **M3 (Extreme)** | 8.0 | $0.3418 \pm 0.0184$ | $0.4712 \pm 0.0272$ | $32.71\%$ | **$67.72\%$** | **$50.65\%$** | $33.18\%$ | **8 / 391 (2.0%)** | 185 ca | 2,072 ca | 251 ca | **2,323 ca (77.0%)** |

---

## 5. PHÂN TÍCH VÀ CẢNH BÁO KHOA HỌC

1. **Số ca bỏ sót giảm mạnh là có thật, nhưng KHÔNG ĐỒNG NGHĨA M2/M3 tốt hơn:**
   - Dưới ma trận $M_3$, số ca $C_2^+$ bị bỏ sót thành $C_0$ giảm còn 7 ca (RF–LR) và 8 ca (RF–RF).
   - Tuy nhiên, **$76.1\% - 77.0\%$ người bình thường bị gán nhầm thành có bệnh** (hơn 2,290 ca báo động giả).
   - Macro-F1 sụt giảm nghiêm trọng xuống **$0.3418$** và Accuracy rơi xuống **$32.7\% - 33.1\%$** (thấp hơn cả đoán ngẫu nhiên đều 3 lớp).
   - **Cảnh báo học thuật:** Tuyệt đối không được chọn $M_2$ hay $M_3$ làm mô hình đề xuất chính thức. Chúng chỉ đóng vai trò minh họa cho đường biên Pareto cực đoan.
2. **Mô hình E6b ($M_1$) là điểm cân bằng tối ưu duy nhất:**
   - Đạt đỉnh Balanced Accuracy ($0.5025$), duy trì Macro-F1 ($0.4521$) ngang ngửa E1 ($0.4550$) và cao hơn E0 ($0.4354$).
   - Giảm số ca bỏ sót nặng từ 120 xuống 62 ca một cách bền vững.
3. **Giá trị sử dụng làm Baseline:**
   - **ĐỦ ĐIỀU KIỆN LÀM BASELINE:** Mô hình **E6b (Soft RF–LR + M1)** và **E6a (Soft RF–RF + M1)** được xác nhận là hoàn toàn hợp lệ, không rò rỉ dữ liệu, có kiểm định thống kê và có thể sử dụng làm đối trọng trực tiếp để so sánh với Ordinal Classification (E7).

