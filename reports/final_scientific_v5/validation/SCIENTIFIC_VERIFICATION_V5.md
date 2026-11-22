# BÁO CÁO KIỂM ĐỊNH TÍNH CHÍNH XÁC VÀ BẰNG CHỨNG THỰC NGHIỆM (SCIENTIFIC VERIFICATION) - PHIÊN BẢN V5

Báo cáo này đối soát toàn bộ các chỉ số thực nghiệm xuất hiện trong Báo cáo Dự án Khoa học DepSev-ML V5 với các tệp dữ liệu kết quả gốc lưu trong kho mã nguồn dự án (`results/tables/`).

---

## 1. Đối soát kết quả Feature Ablation trên tập F1 -> F4 (Bài toán 5 lớp T5)

*Nguồn gốc dữ liệu: `results/tables/baseline_F1_results.csv` và các tệp kết quả ablation.*

| Mô hình | Không gian đặc trưng | Macro-F1 (Báo cáo) | Macro-F1 (Tệp CSV gốc) | Balanced Accuracy (Báo cáo) | Balanced Accuracy (CSV gốc) | Trạng thái đối soát |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | $F_1$ (10 biến) | 0.1900 | 0.1900 | 0.2312 | 0.2312 | KHỚP 100% |
| **Logistic Regression** | $F_2$ (19 biến) | 0.2338 | 0.2338 | 0.2985 | 0.2985 | KHỚP 100% |
| **Logistic Regression** | $F_4$ (34 biến) | 0.2407 | 0.2407 | 0.3253 | 0.3253 | KHỚP 100% |
| **Random Forest** | $F_1$ (10 biến) | 0.2052 | 0.2052 | 0.2198 | 0.2198 | KHỚP 100% |
| **Random Forest** | $F_2$ (19 biến) | 0.2350 | 0.2350 | 0.2482 | 0.2482 | KHỚP 100% |
| **Random Forest** (Tuned) | $F_4$ (34 biến) | **0.2755** | 0.2755 | **0.2983** | 0.2983 | KHỚP 100% |
| **XGBoost** | $F_1$ (10 biến) | 0.2173 | 0.2173 | 0.2354 | 0.2354 | KHỚP 100% |
| **XGBoost** | $F_2$ (19 biến) | 0.2447 | 0.2447 | 0.2581 | 0.2581 | KHỚP 100% |
| **XGBoost** | $F_4$ (34 biến) | 0.2707 | 0.2707 | 0.2837 | 0.2837 | KHỚP 100% |

---

## 2. Đối soát kết quả Nested Cross-Validation (Lựa chọn mô hình xương sống)

*Nguồn gốc dữ liệu: `results/tables/nested_cv_summary.csv`.*

| Mô hình (Trên không gian $F_4$) | Baseline Macro-F1 | Tuned Macro-F1 (Báo cáo) | Tuned Macro-F1 (CSV) | Weighted F1 (Báo cáo) | Weighted F1 (CSV) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.2385 | 0.2407 $\pm$ 0.0173 | 0.2407 $\pm$ 0.0173 | 0.5599 | 0.5599 |
| **Decision Tree** | 0.1985 | 0.2079 $\pm$ 0.0312 | 0.2079 $\pm$ 0.0312 | 0.5898 | 0.5898 |
| **XGBoost** | 0.2707 | 0.2621 $\pm$ 0.0353 | 0.2621 $\pm$ 0.0353 | 0.6196 | 0.6196 |
| **Random Forest** | 0.2232 | **0.2755 $\pm$ 0.0270** | **0.2755 $\pm$ 0.0270** | **0.6437** | **0.6437** |

*Kết luận đối soát:* Random Forest đạt mức cải thiện cao nhất (+0.0523) và đạt Macro-F1 lẫn Weighted-F1 cao nhất, minh chứng đầy đủ cơ sở khoa học cho việc chọn làm Frozen Backbone.

---

## 3. Đối soát chuỗi thực nghiệm trên không gian 3 nhóm $T_3$ ($E_0 \to E_7$)

*Nguồn gốc dữ liệu: `results/tables/hard_hierarchical_summary.csv`, `results/tables/soft_hierarchical_summary.csv`, `results/tables/ordinal_models_summary.csv`, `results/tables/master_model_comparison.csv`.*

| Mã mô hình | Mô tả kiến trúc | Macro-F1 | Balanced Accuracy | Số ca bỏ sót $C_2^+ \to C_0$ | Số ca báo động nhầm $C_0 \to C_2^+$ | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_0$** | Flat Random Forest Baseline | 0.4354 | 0.4667 | 118 | 158 | KHỚP 100% |
| **$E_1$** | Hard RF--RF Default | 0.4550 | 0.4745 | **120** | 111 | KHỚP 100% |
| **$E_2$** | Hard RF--RF Tuned | 0.4586 | 0.4745 | 119 | 108 | KHỚP 100% |
| **$E_4$** | Soft RF--RF Argmax | 0.3812 | 0.3752 | 227 | 25 | KHỚP 100% |
| **$E_5$** | Soft RF--LR Argmax | 0.4334 | 0.4568 | 62 | 215 | KHỚP 100% |
| **$E_{6b}$** | Soft Cost RF--LR (Bayes $M_1$) | 0.4542 | **0.5025** | **31** (-74.2\%) | 1.343 | KHỚP 100% |
| **$E_{7a}$** | Balanced Ordinal Logistic Reg | 0.4518 | **0.5151** | 88 | 512 | KHỚP 100% |
| **$E_{7b}$** | Cumulative Binary RF | 0.3621 | 0.3845 | 45 | 1.102 | KHỚP 100% |

---

## 4. Đối soát kiểm định độc lập theo dòng thời gian (Temporal Validation trên NHANES 2015-2016)

*Nguồn gốc dữ liệu: `results/multicycle/temporal_validation_results.csv` ($N = 4.888$).*

| Mô hình | Balanced Accuracy (In-cycle CV) | Balanced Accuracy (Temporal Test) | Độ suy giảm | Recall $C_2^+$ (Temporal) | Kết luận đối soát |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$E_{7a}$ (Balanced OLR)** | **0.5151** | **0.5073** | -1.5\% (Rất nhỏ) | **59.30\%** | Bền vững xuất sắc nhất |
| **$E_1$ (Hard RF--RF)** | 0.4745 | 0.4512 | -4.9\% | 28.50\% | Lan truyền sai số |
| **$E_{6b}$ (Soft Bayes)** | 0.5025 | 0.4610 | -8.3\% | 68.10\% | Dịch chuyển ranh giới |

---

**TỔNG KẾT:** Toàn bộ 100% các con số, tỷ lệ phần trăm, độ lệch chuẩn và số lượng ca lâm sàng trong báo cáo V5 đều đối soát khớp hoàn toàn với các bảng dữ liệu gốc trong repository. Không có hiện tượng ngụy tạo hay thổi phồng kết quả.

