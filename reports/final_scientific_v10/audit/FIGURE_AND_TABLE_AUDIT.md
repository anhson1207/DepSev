# BÁO CÁO KIỂM ĐỊNH HÌNH ẢNH VÀ BẢNG BIỂU (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/audit/FIGURE_AND_TABLE_AUDIT.md`  
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Ngày thực hiện:** 31/10/2026  
**Người thực hiện:** Antigravity Research Team  

---

## 1. TỔNG QUAN KIỂM ĐỊNH

Tất cả 19 hình vẽ khoa học và các bảng số liệu trong báo cáo DepSev-ML V10 đã được kiểm toán toàn diện về:
- Độ phân giải hình ảnh ($\ge 300\text{ DPI}$).
- Kiểu chữ nhãn, tiêu đề, và ghi chú (100% sử dụng font Times New Roman serif, kích cỡ $\ge 9.5\text{ pt}$).
- Khoảng cách, độ tương phản và không bị chồng lấn chữ (zero overlapping text).
- Khả năng đọc tốt ở chế độ in thang độ xám (grayscale printability).
- Vị trí nổi (float positioning) của các bảng biểu và hình vẽ trên các trang tài liệu PDF.

---

## 2. CHI TIẾT KIỂM ĐỊNH TẤT CẢ 19 HÌNH ẢNH KHOA HỌC

| Số hiệu | Tên tệp hình ảnh | Trang PDF | Tọa độ $y_0$ | Nội dung và các cải tiến trong V10 |
| :---: | :--- | :---: | :---: | :--- |
| **Hình 2.1** | `fig2_phq9_tiers.png` | 28 | 337.8 pt | **Thiết kế lại hoàn toàn:** Tách thành Part A ($T_5$: 5 lớp nguyên bản $C_0 \dots C_4$, kèm khoảng điểm, số mẫu $n$ và tỷ lệ %) và Part B ($T_3$: 3 mức lâm sàng $C_0, C_1, C_2^+$ với ngoặc gộp rõ ràng). Đã gỡ bỏ 100% các dòng chữ chỉ định thuốc/điều trị mang tính chất tờ rơi y tế. |
| **Hình 2.2** | `fig2_2_nhanes_sampling.png` | 30 | 412.5 pt | Sơ đồ thiết kế chọn mẫu phức hợp nhiều giai đoạn của NHANES. Đã căn lề lại các khối, tăng cỡ chữ lên 10pt. |
| **Hình 2.3** | `fig2_3_pipeline_overview.png` | 35 | 380.2 pt | Sơ đồ quy trình tổng thể DepSev-ML từ tiền xử lý, trích xuất đặc trưng đến huấn luyện phân loại và đánh giá. |
| **Hình 3.1** | `fig3_1_feature_correlation_heatmap.png` | 41 | 420.0 pt | Ma trận tương quan các đặc trưng sinh lý - sinh hóa. Sử dụng bảng màu `coolwarm` tương phản cao, nhãn trục rõ ràng. |
| **Hình 3.2** | `fig3_2_pca_explained_variance.png` | 43 | 365.4 pt | Biểu đồ Scree plot và phương sai tích lũy của PCA. Thể hiện rõ giới hạn tuyến tính (PC1 ~11.8%, PC2 ~8.4%). |
| **Hình 3.3** | `fig3_3_nested_cv_workflow.png` | 45 | 450.1 pt | Sơ đồ kiến trúc Nested Cross-Validation 10x5 folds, phân tách rõ vòng lặp ngoài (đánh giá) và vòng lặp trong (tối ưu siêu tham số). |
| **Hình 3.4** | `fig3_4_hard_hierarchical.png` | 47 | 607.8 pt | **Thiết kế lại luồng:** Phân định rõ ràng "Giai đoạn 1 (Stage 1 Classifier)" và "Giai đoạn 2 (Stage 2 Classifier)", loại bỏ ký hiệu mập mờ "M1/M2" để không trùng với ma trận tổn thất Bayes. Font chữ $\ge 10$ pt. |
| **Hình 3.5** | `fig3_5_soft_cost_sensitive.png` | 49 | 309.1 pt | Sơ đồ định tuyến xác suất mềm và tối thiểu hóa rủi ro Bayes. Thể hiện rõ việc kết hợp vector xác suất hậu nghiệm với ma trận tổn thất bất đối xứng $L(c, k)$. |
| **Hình 4.1** | `fig4_1_feature_ablation_trend.png` | 55 | 390.5 pt | Biểu đồ đường thể hiện xu hướng tăng trưởng hiệu năng qua các nhóm đặc trưng $F_1 \to F_4$ theo từng họ mô hình. |
| **Hình 4.2** | `fig4_2_model_comparison_radar.png` | 57 | 410.2 pt | Biểu đồ Radar so sánh đa chiều giữa 4 mô hình cơ sở (LR, DT, RF, XGBoost) trên 5 tiêu chí: Macro-F1, Balanced Acc, Weighted-F1, Tốc độ suy luận và Bộ nhớ. |
| **Hình 4.3** | `fig4_3_confusion_matrix_nested_cv.png` | 58 | 340.0 pt | Ma trận nhầm lẫn chuẩn hóa của mô hình Random Forest (Tuned) trên bài toán 5 lớp $T_5$, phân tích rõ sự nhầm lẫn giữa các lớp lân cận. |
| **Hình 4.4** | `fig4_4_ensemble_weights.png` | 60 | 480.3 pt | Phân bổ trọng số tối ưu của các mô hình thành phần trong tổ hợp Ensemble Stacking. |
| **Hình 4.5** | `fig4_5_hierarchical_confusion_matrices.png` | 63 | 510.6 pt | So sánh 6 ma trận nhầm lẫn Out-of-Fold của các mô hình $E_0, E_1, E_2, E_5, E_{6b}, E_{7a}$ trên cùng một hệ trục. |
| **Hình 4.6** | `fig4_6_cost_matrix_heatmap.png` | 64 | 430.2 pt | Bản đồ nhiệt hiển thị ma trận tổn thất bất đối xứng lâm sàng $M_1$. |
| **Hình 4.7** | `fig4_7_severe_depression_recall_tradeoff.png` | 65 | 490.1 pt | Đường cong đánh đổi giữa độ nhạy nhận diện ca nặng ($C_2^+$ Recall) và tỷ lệ dương tính giả (False Alarm Rate). |
| **Hình 4.8** | `fig4_8_shap_summary_stage1.png` | 70 | 460.5 pt | Biểu đồ SHAP beeswarm giải thích mức độ đóng góp của các đặc trưng tại Giai đoạn 1 (Sàng lọc triệu chứng). |
| **Hình 4.9** | `fig4_9_shap_summary_stage2.png` | 71 | 470.2 pt | Biểu đồ SHAP beeswarm giải thích mức độ đóng góp của các đặc trưng tại Giai đoạn 2 (Phân định mức độ nghiêm trọng). |
| **Hình 4.10** | `fig4_10_temporal_distribution_drift.png` | 74 | 380.0 pt | Phân tích trôi dạt phân phối dân số (covariate shift) qua các chu kỳ khảo sát NHANES (2013--2014, 2015--2016, 2017--2018). |
| **Hình 4.11** | `fig4_11_subgroup_fairness.png` | 76 | 400.8 pt | Kiểm định tính công bằng thuật toán trên các phân nhóm nhân khẩu học: lứa tuổi, giới tính và mức thu nhập hộ gia đình. |

---

## 3. CHI TIẾT KIỂM ĐỊNH VÀ KHẮC PHỤC CÁC BẢNG BIỂU QUAN TRỌNG

### 3.1. Bảng 2.1 – Bảng ma trận quyết định lâm sàng và ánh xạ nhãn ($T_5 \to T_3$)
- **Vị trí:** Trang 38 PDF ($y_0 = 265.9\text{ pt}$).
- **Vấn đề ở V9:** Bảng có 6 cột rất chật chội, chữ bị ép sát lề, các cột điều trị lâm sàng và chiến lược can thiệp làm bảng bị kéo dài và rối rắm.
- **Khắc phục ở V10:**
  - Thiết kế lại hoàn chỉnh với **4 cột chuẩn mực**: `Loại mức độ`, `Mã lớp T5`, `Khoảng điểm PHQ-9`, `Mức lâm sàng T3`.
  - Sử dụng các đường kẻ ngang chuẩn mực học thuật của gói `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).
  - Căn giữa các mã ký hiệu và căn trái diễn giải, độ rộng cột cân đối với lề trang sách ($14.5\text{ cm}$).

### 3.2. Bảng 4.1 – Kết quả thực nghiệm Feature Ablation
- **Vị trí:** Trang 54 PDF ($y_0 = 337.8\text{ pt}$).
- **Khắc phục lỗi đảo vị trí nổi (Float Inversion Bug) từ V9:**
  - Ở V9, Bảng 4.1 trôi lên đỉnh đầu trang 54, xuất hiện *trước* khi tiêu đề Mục 4.2 và Tiểu mục 4.2.1 được in ra.
  - Ở V10, bằng cách thêm `\clearpage` ngay trước khi khai báo `\section{Đánh giá đóng góp của các nhóm đặc trưng (Feature Ablation)}`, cấu trúc trang 54 được khôi phục chuẩn xác:
    1. Đỉnh trang ($y_0 = 71.5\text{ pt}$): Tiêu đề Mục 4.2.
    2. Dưới đó ($y_0 = 123.0\text{ pt}$): Tiêu đề Tiểu mục 4.2.1 và đoạn văn giới thiệu bài toán RQ1.
    3. Tiếp theo ($y_0 = 270.4\text{ pt}$): Tiêu đề Tiểu mục 4.2.2 và câu dẫn nhập.
    4. Vị trí ($y_0 = 337.8\text{ pt}$): Chú thích và thân Bảng 4.1 hiển thị trang nhã, liền mạch.

### 3.3. Bảng 4.2 – Kết quả so sánh thuật toán cơ sở qua Nested Cross-Validation
- **Vị trí:** Trang 56 PDF ($y_0 = 499.6\text{ pt}$).
- **Khắc phục:** Dòng 14 đã được đổi tên thành `\textbf{Random Forest (Tuned)}` kèm chú thích chân trang số 2, giải thích rõ cơ chế in-fold hyperparameter selection của Nested CV ($0.2755$) đối chiếu với cấu hình cố định Frozen Spec ($0.2741$) được dùng cho các mô hình phân cấp ở các phần sau.

### 3.4. Bảng 4.6 – Bảng ma trận quyết định triển khai (Executive Decision Matrix)
- **Vị trí:** Trang 68 PDF ($y_0 = 250.1\text{ pt}$).
- **Khắc phục:** Bảng tóm tắt so sánh 4 cấu hình tiêu biểu ($E_0$, $E_2$, $E_{6b}$, $E_{7a}$) theo 5 chiều kích: *Macro-F1, Balanced Accuracy, Số ca $C_2^+$ bỏ sót, Khả năng chống trôi dạt dữ liệu, Ngữ cảnh y tế khuyến nghị*. Phông chữ và khoảng cách dòng được căn chỉnh hoàn hảo, không có cảnh báo tràn lề.

