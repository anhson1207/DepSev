# BÁO CÁO KIỂM TOÁN SỐ LIỆU THỰC NGHIỆM (NUMERICAL AUDIT) — DEPSEV-ML V8

**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản kiểm định:** Báo cáo khoa học V8 (`DepSev_ML_Scientific_Report_V8.pdf`)  
**Ngày kiểm toán:** 30/10/2026  
**Người thực hiện:** Nhóm nghiên cứu DepSev-ML  

---

## 1. Mục tiêu kiểm toán số liệu

Mục tiêu của đợt kiểm toán số liệu cho phiên bản V8:
1. Đối soát 100% các giá trị định lượng trong toàn bộ các bảng kết quả và ma trận nhầm lẫn với dữ liệu thô trích xuất từ repository (`results/tables/`, `experiments/*/outputs/`).
2. Giải quyết dứt điểm các điểm mâu thuẫn số liệu tồn tại từ phiên bản V7:
   - Sai lệch giữa Bảng 4.6 (Phần B) và ma trận nhầm lẫn out-of-fold ở chỉ số Recall $C_0$ của các mô hình ($E_0 \to E_7$).
   - Sự khác biệt về mặt bản chất và số liệu giữa Random Forest mặc định trong Feature Ablation (Macro-F1 = 0.2232) và Random Forest tinh chỉnh trong Nested CV (Macro-F1 = 0.2755 ± 0.0270).
   - Xác định chính xác số ca nặng bị bỏ sót ($C_2^+ \to C_0$) ở mô hình $E_2$ là **98 ca** (không phải 119 hay 120 ca).
3. Đảm bảo tính nhất quán tuyệt đối giữa số liệu mô tả trong lời văn (prose), trong các bảng số liệu (LaTeX tables) và trong các biểu đồ đồ họa (figures).

---

## 2. Bảng đối soát chi tiết từng chỉ số số liệu

| Vị trí trong báo cáo | Tệp dữ liệu gốc (Repository Path) | Công thức / Phương thức tính | Giá trị V7 | Giá trị V8 | Trạng thái (Status) | Ghi chú khoa học |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bảng 4.6 (Phần B) - $E_0$ Recall $C_0$** | `results/tables/master_benchmark_t3.csv` & `experiments/hierarchical_t3/outputs/e0_flat_baseline_eval.json` | Tỷ lệ $C_0$ đoán đúng trên tập OOF: $2.304 / 3.016$ | 75.36% | **76.39%** | **PASS** | V7 tồn tại lỗi gõ phím từ V3 (75.36%). Giá trị đúng từ 5-fold mean là 76.39% (độ lệch với OOF matrix < 0.01%). |
| **Bảng 4.6 (Phần B) - $E_1$ Recall $C_0$** | `experiments/hierarchical_t3/outputs/e1_hard_rf_rf_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.478 / 3.016$ | 82.16% | **82.16%** | **PASS** | Khớp tuyệt đối giữa 5-fold mean (82.16%) và pooled matrix (82.16%). |
| **Bảng 4.6 (Phần B) - $E_2$ Recall $C_0$** | `experiments/hierarchical_t3/outputs/e2_hard_rf_rf_tuned_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.323 / 3.016$ | 76.99% | **77.02%** | **PASS** | Đồng bộ chuẩn xác theo giá trị trung bình 5 fold (77.02%). |
| **Bảng 4.6 (Phần B) - $E_2$ Bỏ sót $C_2^+ \to C_0$** | `experiments/hierarchical_t3/outputs/e2_hard_rf_rf_tuned_eval.json` (pooled confusion matrix) | Số mẫu $C_2^+$ bị Stage 1 gán thành $C_0$ | 98 ca (text có chỗ ghi 119) | **98 ca** (đồng nhất 100%) | **PASS** | Đã sửa toàn bộ các câu chữ trong Chương 4, Chương 5 và Bảng 4.7 về đúng 98 ca. |
| **Bảng 4.6 (Phần B) - $E_{3a}$ Recall $C_0$** | `experiments/hierarchical_t3/outputs/e3a_hard_rf_lr_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.478 / 3.016$ | 82.16% | **82.16%** | **PASS** | Do Stage 1 giống $E_1$ (RF default), Recall $C_0$ giống hệt $E_1$ (82.16%). |
| **Bảng 4.6 (Phần B) - $E_{3b}$ Recall $C_0$** | `experiments/hierarchical_t3/outputs/e3b_hard_rf_xgb_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.478 / 3.016$ | 82.16% | **82.16%** | **PASS** | Đồng nhất với $E_1$ và $E_{3a}$ tại Stage 1 (82.16%). |
| **Bảng 4.6 (Phần B) - $E_4$ Recall $C_0$** | `experiments/soft_hierarchical_t3/outputs/e4_soft_rf_rf_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.832 / 3.016$ | 93.90% | **93.90%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.6 (Phần B) - $E_5$ Recall $C_0$** | `experiments/soft_hierarchical_t3/outputs/e5_soft_rf_lr_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.793 / 3.016$ | 92.61% | **92.61%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.6 (Phần B) - $E_{6a}$ Recall $C_0$** | `experiments/soft_hierarchical_t3/outputs/e6a_soft_cost_rf_rf_m1_eval.json` | 5-fold mean Recall $C_0$ / OOF $1.868 / 3.016$ | 61.94% | **61.94%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.6 (Phần B) - $E_{6b}$ Recall $C_0$** | `experiments/soft_hierarchical_t3/outputs/e6b_soft_cost_rf_lr_m1_eval.json` | 5-fold mean Recall $C_0$ / OOF $1.866 / 3.016$ | 61.87% | **61.87%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.6 (Phần B) - $E_{7a}$ (Unweighted) Recall $C_0$** | `experiments/ordinal_t3/outputs/e7a_olr_unweighted_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.957 / 3.016$ | 98.04% | **98.04%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.6 (Phần B) - $E_{7a}$ (Balanced) Recall $C_0$** | `experiments/ordinal_t3/outputs/e7a_olr_balanced_eval.json` | 5-fold mean Recall $C_0$ / OOF $1.847 / 3.016$ | 61.24% | **61.24%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.6 (Phần B) - $E_{7b}$ (CumBin LR) Recall $C_0$** | `experiments/ordinal_t3/outputs/e7b_cumbin_lr_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.280 / 3.016$ | 75.60% | **75.60%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.6 (Phần B) - $E_{7b}$ (CumBin RF) Recall $C_0$** | `experiments/ordinal_t3/outputs/e7b_cumbin_rf_eval.json` | 5-fold mean Recall $C_0$ / OOF $2.701 / 3.016$ | 89.56% | **89.56%** | **PASS** | Khớp hoàn toàn. |
| **Bảng 4.1 (Feature Ablation) - RF trên $F_4$** | `results/tables/feature_group_ablation_5class.csv` | 5-fold mean Macro-F1 với siêu tham số mặc định | 0.2232 $\pm$ 0.0457 | **0.2232 $\pm$ 0.0457** | **PASS** | Giữ nguyên và làm rõ: đây là RF với tham số mặc định (default params). |
| **Bảng 4.2 (Nested CV) - RF trên $F_4$** | `results/tables/nested_cv_baseline_comparison.csv` | Outer 5-fold mean Macro-F1 với siêu tham số tối ưu | 0.2755 $\pm$ 0.0270 | **0.2755 $\pm$ 0.0270** | **PASS** | Giữ nguyên và giải thích: đây là RF đã qua tinh chỉnh (Tuned / Frozen Spec), tăng +0.0523 so với default RF. |
| **Hình 4.6 - Spearman $\rho$ biến liên tục với PHQ-9** | `results/tables/overlap_spearman_correlations.csv` | Hệ số tương quan hạng Spearman $\rho$ | (Chưa có biểu đồ riêng ở V7) | **Albumin: -0.164, PIR: -0.159, Sleep: -0.139, BMI: +0.114, Waist: +0.095, Pulse: +0.085** | **PASS** | Đã tạo biểu đồ 2 panel khớp 100% dữ liệu CSV. |
| **Hình 4.6 - Cramer's V biến phân loại với PHQ-9** | `results/tables/overlap_categorical_associations.csv` | Hệ số liên hệ Cramer's V | (Chưa có biểu đồ riêng ở V7) | **HUQ010: 0.195, Gender: 0.144, PAQ665: 0.125, BPQ020: 0.109, DIQ010: 0.100** | **PASS** | Đã đưa vào Panel (b) Hình 4.6 và phân tích chi tiết. |
| **Hình 4.7 - Boxplots 6 biến đặc trưng** | `results/tables/overlap_numerical_summary.csv` | Trung vị và tứ phân vị (IQR) theo 5 mức PHQ-9 | 15 boxplot nhỏ dính chữ | **Lưới 2x3 gồm 6 biến: BMI, PIR, Sleep, Albumin, Pulse, Sedentary** | **PASS** | Số liệu vẽ trực tiếp từ bảng tóm tắt phân vị thực nghiệm. |
| **Hình 4.8 - Tỷ lệ phương sai giải thích của PCA** | `results/tables/overlap_pca_coordinates.csv` | Tỷ lệ phương sai thành phần chính PC1 và PC2 | (Chưa ghi rõ \% trong text) | **PC1: 12.3%, PC2: 8.8%, Tổng: 21.1%** | **PASS** | Đồng bộ chính xác giữa nhãn trục biểu đồ và văn bản phân tích. |

---

## 3. Kết luận kiểm toán số liệu

- Toàn bộ 20 hạng mục số liệu trọng yếu đã được kiểm tra, chuẩn hóa và đối soát thành công 100% với các bảng số liệu thực nghiệm gốc trong repository.
- Sai lệch legacy ở $E_0$ Recall $C_0$ (75.36% $\to$ 76.39%) và số ca bỏ sót ở $E_2$ (119 $\to$ 98 ca) đã được sửa dứt điểm, không còn bất kỳ mâu thuẫn nội tại nào giữa các chương và các bảng biểu.
- Đã phân định minh bạch hai giá trị hiệu năng của Random Forest trên không gian $F_4$: Macro-F1 = 0.2232 (mô hình mặc định trong thực nghiệm Feature Ablation) và Macro-F1 = 0.2755 (mô hình tinh chỉnh siêu tham số trong Nested CV).

