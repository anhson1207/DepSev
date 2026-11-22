# BÁO CÁO KIỂM ĐỊNH HÌNH ẢNH TRONG BÁO CÁO DEPSEV-ML V7
*(Figure Quality Assurance and Verification Report)*

Tài liệu này ghi nhận kết quả kiểm tra chất lượng của tất cả 19 hình vẽ trong báo cáo DepSev-ML V7.

---

## 1. DANH SÁCH 19 HÌNH VẼ VÀ TRẠNG THÁI HIỂN THỊ

| Số thứ tự hình | Vị trí trang | Tên tập tin hình | Tiêu đề chú thích | Kiểm tra chất lượng & Nhãn | Trạng thái |
| :---: | :---: | :--- | :--- | :--- | :---: |
| **Hình 1.1** | Trang 20 | `figures/fig1_depsev_flowchart.png` | Sơ đồ tổng quan phát biểu bài toán và luồng xử lý của khung học máy DepSev-ML | Rõ nét, không tràn lề, chuẩn tiếng Việt | ĐẠT |
| **Hình 1.2** | Trang 22 | `figures/research_workflow_t5_to_t3.png` | Sơ đồ quy trình nghiên cứu: Từ khảo sát 5 mức độ $T_5$ đến mô hình phân cấp $T_3$ | Rõ nét, luồng $T_5 \to T_3$ mạch lạc | ĐẠT |
| **Hình 2.1** | Trang 26 | `figures/fig1_phq9_clinical_mapping.png` | Thang đo PHQ-9 (0–27 điểm), 5 khoảng mức độ triệu chứng và ánh xạ sang các bài toán | Thiết kế vector sắc nét, nhãn chuẩn | ĐẠT |
| **Hình 2.2** | Trang 27 | `figures/fig1_biological_mechanisms.png` | Các cơ chế y sinh học liên kết chỉ số sinh lý ngoại vi với mức độ triệu chứng trầm cảm | Minh họa sơ đồ y sinh mạch lạc | ĐẠT |
| **Hình 2.3** | Trang 28 | `figures/fig1_class_imbalance_distribution.png` | (a) Mất cân bằng lớp trên tập dữ liệu và (b) Hiện tượng chồng lấn phân phối sinh lý | Subplot (a) & (b) cân đối, màu sắc rõ | ĐẠT |
| **Hình 2.4** | Trang 30 | `figures/fig2_decision_tree_concept.png` | Sơ đồ nguyên lý phân nhánh của Decision Tree trên không gian đặc trưng | Đã đổi nhãn thành "Decision Tree" | ĐẠT |
| **Hình 2.5** | Trang 31 | `figures/fig2_random_forest_architecture.png` | Kiến trúc Random Forest: Bootstrap Sampling, ngẫu nhiên hóa đặc trưng và Soft Voting | Đã đổi nhãn thành "Decision Tree 1..B" | ĐẠT |
| **Hình 2.6** | Trang 33 | `figures/fig2_paradigms_comparison.png` | Ba trường phái mô hình hóa: (a) Phân loại phẳng, (b) Phân loại phân cấp và (c) Phân loại thứ bậc | 3 nhánh phân loại hiển thị rõ | ĐẠT |
| **Hình 3.1** | Trang 38 | `figures/fig3_depsev_framework_pipeline.png` | Kiến trúc tổng thể 7 module của khung học máy thực nghiệm DepSev-ML | Kiến trúc tổng quan hoàn chỉnh | ĐẠT |
| **Hình 3.2** | Trang 41 | `figures/fig3_preprocessing_pipeline.png` | Quy trình tiền xử lý dữ liệu in-fold và nguyên tắc cách ly bảo toàn Zero Data Leakage | Sơ đồ chia fold và tiền xử lý độc lập | ĐẠT |
| **Hình 3.3** | Trang 42 | `figures/fig3_feature_spaces_pyramid.png` | Cấu trúc 4 không gian đặc trưng lũy tiến $F_1 \to F_4$ trong DepSev-ML | Dạng kim tự tháp lũy tiến rõ ràng | ĐẠT |
| **Hình 3.4** | Trang 45 | `figures/two_stage_data_flow.png` | Kiến trúc phân loại phân cấp định tuyến cứng hai giai đoạn và điểm nghẽn sai số một chiều | Đã xóa môi trường trùng lặp ở V6 | ĐẠT |
| **Hình 3.5** | Trang 47 | `figures/fig3_cost_sensitive_routing.png` | Kiến trúc phân loại phân cấp mềm kết hợp lý thuyết quyết định nhạy chi phí Bayes | Quy trình tích hợp ma trận chi phí rõ | ĐẠT |
| **Hình 3.6** | Trang 48 | `figures/fig3_ordinal_architectures.png` | Nguyên lý mô hình hóa thứ bậc: (a) Ordinal Logistic Regression và (b) Cumulative Binary | Minh họa ranh giới siêu phẳng song song | ĐẠT |
| **Hình 3.7** | Trang 49 | `figures/fig3_validation_schemes.png` | Hai tầng giao thức kiểm thử: Nested 5-Fold Stratified CV và Temporal Validation | Sơ đồ lồng nhau và chia chu kỳ rõ | ĐẠT |
| **Hình 4.1** | Trang 53 | `figures/fig4_ablation_f1_comparison.png` | So sánh hiệu năng Macro-F1 của 4 thuật toán trên các nhóm đặc trưng lũy tiến $F_1 \to F_4$ | Đồ thị so sánh đối chuẩn rõ nét | ĐẠT |
| **Hình 4.2** | Trang 55 | `figures/fig4_rf_vs_complex_models.png` | So sánh hiệu năng giữa mô hình đơn nhất Frozen Random Forest và các mô hình kết hợp | Phân tích trực quan vai trò của RF | ĐẠT |
| **Hình 4.3** | Trang 56 | `figures/fig4_granularity_degradation.png` | Đồ thị thể hiện sự suy thoái của Macro-F1 và Balanced Accuracy từ $T_2$ đến $T_5$ | Thể hiện rõ điểm rơi ở $T_2 \to T_3$ | ĐẠT |
| **Hình 4.4** | Trang 57 | `figures/fig4_class_distribution_by_granularity.png` | Biến động cấu trúc phân bố các lớp khi tăng độ phân giải nhãn mục tiêu từ $T_2$ đến $T_5$ | Phân bố tỷ lệ mẫu trực quan | ĐẠT |
| **Hình 4.5** | Trang 58 | `figures/granularity_t5_confusion_matrix.png` | Ma trận nhầm lẫn của Random Forest trên bài toán 5 mức độ $T_5$ (Out-of-Fold) | **Hình mới bổ sung**: Heatmap 5x5 có cả số lượng mẫu và phần trăm theo hàng | ĐẠT |
| **Hình 4.6** | Trang 59 | `figures/fig4_correlation_matrix.png` | Ma trận tương quan giữa các đặc trưng sinh lý, lâm sàng chính và điểm PHQ-9 | Heatmap tương quan Spearman rõ | ĐẠT |
| **Hình 4.7** | Trang 60 | `figures/fig4_feature_distributions_by_class.png` | Biểu đồ hộp thể hiện sự tương đồng và chồng lấn phân phối của các biến sinh lý qua 5 lớp | Boxplot chuẩn khoa học | ĐẠT |
| **Hình 4.8** | Trang 61 | `figures/fig4_pca_latent_space.png` | Hình chiếu PCA 2D không gian đặc trưng thể hiện sự chồng lấn phân phối giữa các lớp | Minh họa sự chồng lấn phân phối | ĐẠT |
| **Hình 4.9** | Trang 62 | `figures/multicycle_cohort_comparison.png` | So sánh Macro-F1 và Balanced Accuracy theo độ chi tiết nhãn giữa chu kỳ đơn lẻ, gộp và thời gian | **Hình mới thay thế**: Đúng nội dung đối sánh $T_2 \to T_5$ qua Branch A, B, D | ĐẠT |
| **Hình 4.10** | Trang 63 | `figures/fig4_benchmark_f1_balacc.png` | Đối sánh trực quan hiệu năng thuật toán trên bài toán 5 lớp chuẩn $T_5$ (Nested CV) | Thể hiện RF Tuned đạt mốc cao nhất | ĐẠT |
| **Hình 4.11** | Trang 66 | `figures/fig4_confusion_error_analysis.png` | (a) Số ca nặng bị bỏ sót thành bình thường và (b) So sánh Recall từng lớp giữa $E_1, E_{6b}$ và $E_{7a}$ | Minh họa mức giảm từ 120 xuống 62 ca | ĐẠT |
| **Hình 4.12** | Trang 67 | `figures/monotonicity_violation_distribution.png` | Phân bố vi phạm tính đơn điệu xác suất trong mô hình phân loại nhị phân tích lũy $E_{7b}$ | Thể hiện tỷ lệ vi phạm 19.53% (LR) | ĐẠT |
| **Hình 4.13** | Trang 69 | `figures/legacy_model_comparison_e0_to_e6.png` | So sánh trực quan hiệu năng giữa các mô hình từ $E_0$ đến $E_6$ trên tập Legacy | Bức tranh tổng thể các mô hình $T_3$ | ĐẠT |
| **Hình 4.14** | Trang 69 | `figures/macro_f1_vs_mae.png` | Mối quan hệ đánh đổi giữa Macro-F1 và Class MAE giữa các trường phái mô hình | Đồ thị phân tán đánh đổi rõ | ĐẠT |
| **Hình 4.15** | Trang 70 | `figures/2step_error_comparison.png` | So sánh số lượng lỗi nhảy 2 bước ($C_2^+ \to C_0$) giữa các mô hình | Thể hiện $E_2$ bỏ sót 98 ca, $E_1$ 120 ca | ĐẠT |
| **Hình 4.16** | Trang 71 | `figures/confusion_matrices_group1.png` | Ma trận nhầm lẫn của 4 mô hình đại diện: $E_0$ (Flat RF), $E_1$ (Hard RF--RF), $E_5$ (Soft RF--LR) và $E_{6b}$ | Lưới 2x2 rõ nét, số liệu chuẩn | ĐẠT |
| **Hình 4.17** | Trang 72 | `figures/confusion_matrices_group2.png` | Ma trận nhầm lẫn của 2 mô hình phân loại thứ bậc: $E_{7a}$ (Balanced OLR) và $E_{7b}$ (CumBin RF) | Ma trận thứ bậc chuẩn | ĐẠT |
| **Hình 4.18** | Trang 72 | `figures/cost_tradeoff_curve.png` | Đường cong đánh đổi giữa tỷ lệ nhận diện ca nặng ($C_2^+$ Recall) và tỷ lệ báo động nhầm ($C_0$ FAR) | Đường cong Pareto thực nghiệm rõ | ĐẠT |
| **Hình 4.19** | Trang 74 | `figures/multicycle_soft_comparison.png` | So sánh hiệu năng các mô hình phân cấp mềm và nhạy chi phí giữa Branch A, B và D | Đã chuyển về đúng vị trí kiểm định thời gian | ĐẠT |

