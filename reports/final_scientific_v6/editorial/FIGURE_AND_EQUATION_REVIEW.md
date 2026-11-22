# ĐÁNH GIÁ VÀ KIỂM ĐỊNH HÌNH ẢNH, BẢNG BIỂU VÀ CÔNG THỨC TOÁN HỌC (FIGURE AND EQUATION REVIEW)
**Dự án:** DepSev-ML  
**Tài liệu rà soát:** Báo cáo Khoa học V6 (`DepSev_ML_Scientific_Report_V6.pdf`)  
**Bộ sinh đồ họa:** `generate_perfect_figures.py` (Phiên bản V6 Master)  
**Ngày thực hiện:** 30/10/2026  

---

## 1. Kiểm định hệ thống 17 hình ảnh xuất bản (Publication Figures)

Toàn bộ 17 hình ảnh trong Báo cáo V6 đã được tái sinh tự động với độ phân giải cao 300 DPI, sử dụng bảng màu khoa học đồng nhất (Accessible Palette), phông chữ rõ ràng, bố cục rộng rãi, bo góc chuẩn mực và tuyệt đối không chồng đè văn bản.

| Số hiệu hình | Tệp hình ảnh | Tiêu đề và Nội dung kiểm định | Trạng thái V6 |
| :---: | :--- | :--- | :---: |
| **Hình 1.1** | `fig1_problem_formulation.png` | Sơ đồ bài toán toán học: từ ma trận đặc trưng $\mathbf{X} \in \mathbb{R}^{N \times d}$ đến 4 không gian mục tiêu $T_2, T_3, T_4, T_5$. | **HOÀN HẢO** |
| **Hình 1.2** | `fig_research_progression_t5_to_t3.png` | Sơ đồ tiến trình nghiên cứu hai giai đoạn: Giai đoạn 1 ($T_5$ phẳng) $\to$ Giai đoạn 2 ($T_3$ phân cấp hai bước). | **HOÀN HẢO** |
| **Hình 2.1** | `fig2_phq9_tiers.png` | Cấu trúc phân bậc thang đo PHQ-9 và các ngưỡng cắt can thiệp lâm sàng tương ứng. | **HOÀN HẢO** |
| **Hình 2.2** | `fig2_gut_brain_metabolic_axis.png` | Cơ chế sinh lý bệnh học: Trục ruột - não - chuyển hóa và các chỉ dấu sinh học ngoại vi trong NHANES. | **HOÀN HẢO** |
| **Hình 2.3** | `fig2_imbalance_overlap.png` | Hai rào cản thông tin y sinh cốt lõi: Mất cân bằng lớp cực đoan (68.5:1) và Chồng lấn phân phối đặc trưng. | **HOÀN HẢO** |
| **Hình 2.4** | `fig2_decision_tree_concept.png` | Nguyên lý hoạt động của Cây quyết định: rẽ nhánh nhị phân, chỉ số Gini Impurity và phân vùng siêu hộp. | **HOÀN HẢO** |
| **Hình 2.5** | `fig2_random_forest_architecture.png` | Kiến trúc Rừng ngẫu nhiên: Lấy mẫu Bootstrap, ngẫu nhiên hóa đặc trưng (Random Subspace) và Soft Voting. | **HOÀN HẢO** |
| **Hình 2.6** | `fig2_modeling_paradigms.png` | So sánh trực quan 3 trường phái mô hình hóa: Phân loại phẳng, Phân loại phân cấp và Phân loại thứ bậc. | **HOÀN HẢO** |
| **Hình 3.1** | `depsev_overall_architecture.png` | Kiến trúc tổng thể 7 module độc lập của khung làm việc học máy DepSev-ML. | **HOÀN HẢO** |
| **Hình 3.2** | `fig3_infold_cv_pipeline.png` | Quy trình tiền xử lý In-Fold bảo toàn nguyên tắc Zero Data Leakage tuyệt đối. | **HOÀN HẢO** |
| **Hình 3.3** | `fig3_3_feature_groups.png` | Cấu trúc phân tầng 4 không gian đặc trưng lũy tiến $F_1 \to F_4$ (từ 16 đến 34 biến). | **HOÀN HẢO** |
| **Hình 3.4** | `fig3_4_hard_hierarchical.png` | Kiến trúc phân cấp định tuyến cứng hai giai đoạn (Hard Routing) và điểm nghẽn lan truyền sai số. | **HOÀN HẢO** |
| **Hình 3.5** | `fig3_5_soft_cost_sensitive.png` | Kiến trúc phân cấp định tuyến mềm kết hợp tối ưu chi phí Bayes: **Cập nhật chính xác 120 ca $\to$ 62 ca ($M_1$, -48.3%)**, sửa ký hiệu $\mathbf{C}(2, 0)$. | **ĐÃ SỬA CHUẨN** |
| **Hình 3.6** | `fig3_6_ordinal_classification.png` | Nguyên lý phân loại thứ bậc: Ranh giới siêu phẳng song song (OLR) và ngưỡng cắt phân loại tích lũy (CBC). | **HOÀN HẢO** |
| **Hình 3.7** | `fig3_7_evaluation_protocol.png` | Hai tầng giao thức kiểm định độc lập: Nested 5-Fold CV (Legacy) và Temporal Validation (Branch D). | **HOÀN HẢO** |
| **Hình 4.1** | `fig4_algorithm_comparison_t5_t3.png` | Đối sánh hiệu năng các thuật toán học máy cơ sở trên $T_5$ và $T_3$. | **HOÀN HẢO** |
| **Hình 4.7** | `fig4_confusion_error_analysis.png` | Phân tích lỗi bỏ sót ca nặng và Recall từng lớp: **Cập nhật chính xác số ca bỏ sót [E0: 108, E1: 120, E4: 227, E5: 211, E6b: 62, E7a: 68]** và Recall từng lớp $E_1, E_{6b}, E_{7a}$. | **ĐÃ SỬA CHUẨN** |

---

## 2. Kiểm định các công thức toán học then chốt

### 2.1. Chỉ số bất thuần Gini (Gini Impurity)
- **Công thức:**
  $$G = 1 - \sum_{k=0}^{K-1} p_k^2$$
- **Đánh giá:** Chuẩn xác về mặt xác suất, $G = 0$ khi nút hoàn toàn thuần khiết.

### 2.2. Giới hạn mẫu ngoài túi (Out-Of-Bag Limit)
- **Công thức:**
  $$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368$$
- **Đánh giá:** Giải thích rõ ràng nguồn gốc con số $36.8\%$ mẫu OOB trong kỹ thuật đóng bao (Bagging).

### 2.3. Công thức triệt tiêu phương sai của Rừng ngẫu nhiên (Variance Reduction)
- **Công thức:**
  $$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
- **Đánh giá:** Minh chứng toán học sáng tỏ: khi $B \to \infty$, phương sai rừng bị chặn dưới bởi $\rho \sigma^2$; việc ngẫu nhiên hóa đặc trưng làm giảm $\rho$, giúp rừng có phương sai thấp hơn hẳn cây đơn lẻ mà không làm tăng bias.

### 2.4. Hàm rủi ro kỳ vọng và Quy tắc quyết định Bayes nhạy chi phí
- **Công thức chuẩn trong V6:**
  $$R(a \mid \mathbf{x}) = \sum_{j=0}^{K-1} \mathbf{C}(j, a) \hat{P}(C_j \mid \mathbf{x})$$
  $$\hat{y}^* = \arg\min_{a \in \{0, 1, \dots, K-1\}} R(a \mid \mathbf{x})$$
- **Đánh giá:** Khắc phục triệt để lỗi tráo đổi chỉ số hàng/cột của V5. Ký hiệu rõ ràng: $j$ là nhãn thực tế, $a$ là hành động dự đoán, $\mathbf{C}(j, a)$ là chi phí phạt khi thực tế là $j$ nhưng dự đoán là $a$.

### 2.5. Giải thuật Isotonic Probability Clipping
- **Công thức:**
  $$\tilde{P}_2 = \min\left(\hat{P}_1, \hat{P}_2\right)$$
  $$P(C_1 \mid \mathbf{x}) = \hat{P}_1 - \tilde{P}_2 \ge 0$$
- **Đánh giá:** Khẳng định tính không âm của phân phối xác suất suy luận, đồng thời giải thích rõ tại sao việc ép $\tilde{P}_2 = \hat{P}_1$ khiến Recall của $C_1$ bị suy biến về $0\%$.

