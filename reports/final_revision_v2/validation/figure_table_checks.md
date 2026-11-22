# FIGURE AND TABLE VERIFICATION CHECKLIST

## 1. Table Inventory and Verification

| Table ID | Title / Content | File Location / Mechanism | Underlying Ground Truth Data Source | Verification Status |
|---|---|---|---|---|
| **Table 3.1** | Phân bố các lớp triệu chứng PHQ-9 trên Legacy Cohort ($N = 5.054$) | `chapters/chapter3.tex` | `results/tables/class_distribution.csv` | **VERIFIED** |
| **Table 3.2** | Phân bố các lớp triệu chứng trên các tập dữ liệu Multi-cycle chính thức | `chapters/chapter3.tex` | `results/tables/multicycle_class_distribution.csv` | **VERIFIED** |
| **Table 3.3** | Danh mục hệ thống 34 đặc trưng trong không gian $F_4$ | `tables/table_feature_inventory.tex` | `results/tables/ablation_feature_inventory_final_protocol.csv` | **VERIFIED** |
| **Table 4.1** | Kết quả thực nghiệm Feature Ablation ($N = 4.045$, $T_5$) | `tables/table_ablation_results.tex` | `results/tables/feature_ablation_results_final_protocol.csv` | **VERIFIED** |
| **Table 4.2** | Hiệu năng mô hình Random Forest theo độ chi tiết nhãn ($T_2 \to T_5$) | `tables/table_granularity_results.tex` | `results/tables/granularity_results.csv` | **VERIFIED** |
| **Table 4.3** | So sánh Flat Classification và Hard Hierarchical trên nhiệm vụ $T_3$ | `tables/table_hard_hierarchical.tex` | `experiments/hierarchical_t3/outputs/model_comparison.csv` | **VERIFIED** |
| **Table 4.4** | Hiệu năng mô hình Soft Hierarchical và Quyết định Nhạy chi phí ($T_3$) | `tables/table_soft_hierarchical.tex` | `experiments/soft_hierarchical_t3/outputs/model_comparison.csv` | **VERIFIED** |
| **Table 4.5** | Kết quả thực nghiệm các mô hình Ordinal Classification ($T_3$) | `tables/table_ordinal_results.tex` | `experiments/ordinal_t3/outputs/model_comparison.csv` | **VERIFIED** |
| **Table 4.6** | Bảng so sánh tổng hợp toàn diện các mô hình ($E_0 \to E_7$) trên Legacy Cohort | `tables/table_master_comparison.tex` | Integrated CV comparisons + Majority Baseline | **VERIFIED** |
| **Table 4.7** | Kết quả kiểm định tổng quát hóa theo dòng thời gian (Branch D, $N = 4.888$) | `tables/table_multicycle_results.tex` | `experiments/ordinal_t3/outputs/multicycle_comparison.csv` | **VERIFIED** |

---

## 2. Figure Inventory and Verification

| Figure ID | Description | Asset Path | Status in PDF |
|---|---|---|---|
| **Figure 4.1** | So sánh hiệu năng Macro-F1 của 4 thuật toán trên các nhóm $F_1 \to F_4$ | `figures/feature_group_comparison_final_protocol.png` | **RENDERED** |
| **Figure 4.2** | Đường cong suy giảm hiệu năng theo độ chi tiết nhãn mục tiêu ($T_2 \to T_5$) | `figures/granularity_performance.png` | **RENDERED** |
| **Figure 4.3** | Ma trận nhầm lẫn Out-of-fold của nhiệm vụ 3 lớp $T_3$ (Flat RF $E_0$) | `figures/granularity_T3_confusion.png` | **RENDERED** |
| **Figure 4.4** | Trực quan hóa PCA: Sự chồng lấn phân phối sinh lý giữa các mức độ | `figures/overlap_pca_C0_C4.png` | **RENDERED** |
| **Figure 4.5** | Đường cong đánh đổi giữa việc giảm bỏ sót nặng và gia tăng báo động giả ($M_0 \to M_3$) | `figures/cost_tradeoff_curve.png` | **RENDERED** |
| **Figure 4.6** | Tỷ lệ vi phạm tính đơn điệu ($q_2 > q_1$) trên 5 folds của phương pháp Frank & Hall ($E_{7b}$) | `figures/monotonicity_violation_distribution.png` | **RENDERED** |
| **Figure 4.7** | Biểu đồ phân tán so sánh Macro-F1 và Class MAE | `figures/macro_f1_vs_mae.png` | **RENDERED** |
| **Figure 4.8** | So sánh số ca sai lệch nghiêm trọng 2 mức ($C_2^+ \to C_0$ vs $C_0 \to C_2^+$) | `figures/2step_error_comparison.png` | **RENDERED** |
| **Figure 4.9** | So sánh ma trận nhầm lẫn Out-of-fold của các mô hình đại diện then chốt | `figures/confusion_matrices_comparison.png` | **RENDERED** |

---

## 3. Typography and Layout Integrity

- **Tectonic Compilation**: Exited with code 0 (no fatal errors, no missing glyphs).
- **Page Count**: Exactly 70 pages.
- **Reference Integrity**: 0 undefined `\ref` or `\cite` warnings.
- **Table Formatting**: Booktabs horizontal rules (`\toprule`, `\midrule`, `\bottomrule`), text wrapping with explicit column widths, zero table overspill beyond margins.
- **LaTeX Math KaTeX / TeX Compliance**: Equations written with proper display math and subscripts.

