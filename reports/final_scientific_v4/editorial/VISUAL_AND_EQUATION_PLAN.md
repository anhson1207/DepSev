# KẾ HOẠCH HỆ THỐNG TRỰC QUAN HÓA VÀ TINH GỌN CÔNG THỨC TOÁN HỌC (V4)

## 1. KẾ HOẠCH TINH GỌN CÔNG THỨC TOÁN HỌC

Mục tiêu: Giảm thiểu số lượng công thức từ >35 công thức trong các bản cũ xuống **12 công thức cốt lõi** trong phần thân bài viết. Mọi công thức được giữ lại đều phải có vai trò trực tiếp trong việc định nghĩa bài toán, giải thích cơ chế phân loại hoặc tính toán độ đo thực nghiệm.

### Bảng phân loại toàn bộ công thức:

| STT | Ký hiệu / Tên công thức | Trạng thái | Vị trí đề xuất | Lý do và vai trò |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $\mathcal{D} = \{(\mathbf{x}_i, s_i)\}_{i=1}^N$ (Tập dữ liệu) | **KEEP** | Chương 1 (1.1) | Định nghĩa toán học nền tảng của bài toán học có giám sát. |
| 2 | $s_i = \sum_{j=1}^9 \text{DPQ}_{ij}$ (Tổng điểm PHQ-9) | **KEEP** | Chương 1 (1.2) | Xác lập công thức tính điểm gốc từ 9 câu hỏi khảo sát. |
| 3 | Ánh xạ mục tiêu $\Phi_{T_2}, \Phi_{T_3}, \Phi_{T_4}, \Phi_{T_5}$ | **KEEP (Tóm tắt)** | Chương 1 (1.3--1.6) | Định nghĩa 4 không gian phân loại mục tiêu; trình bày cô đọng trong 1 nhóm công thức. |
| 4 | $\text{Macro-F1} = \frac{1}{K}\sum_k \text{F1}_k$ | **KEEP** | Chương 1 (1.7) | Hàm mục tiêu đánh giá chính của toàn bộ đề tài trên dữ liệu mất cân bằng. |
| 5 | $\text{Balanced Accuracy} = \frac{1}{K}\sum_k \text{Recall}_k$ | **KEEP** | Chương 1 (1.8) | Độ đo phản ánh năng lực nhận diện đồng đều trên các lớp. |
| 6 | Trọng số lớp cân bằng $w_k = \frac{N}{K \cdot N_k}$ | **KEEP** | Chương 2 (2.1) | Cơ chế toán học để khắc phục thiên kiến lớp đa số trong huấn luyện. |
| 7 | Vùng lỗi Bayes $\Omega_{\text{overlap}} = \int \min(P_0 p_0, P_1 p_1) d\mathbf{x}$ | **SIMPLIFY** | Chương 2 (2.2) | Giải thích nguyên nhân lý thuyết của sự suy giảm hiệu năng do chồng lấn sinh lý. |
| 8 | Xác suất Sigmoid / Softmax của Hồi quy Logistic | **REMOVE** | Bỏ khỏi Chương 2 | Kiến thức giáo trình đại cương; không cần thiết trong báo cáo khoa học. |
| 9 | Hàm mất mát Cross-Entropy và đạo hàm gradient | **REMOVE** | Bỏ khỏi Chương 2 | Kiến thức giáo trình cơ bản; đã được tối ưu hóa chuẩn tắc trong scikit-learn. |
| 10 | Độ vẩn đục Gini / Entropy và tiêu chuẩn chia nhánh Cây | **REMOVE** | Bỏ khỏi Chương 2 | Lý thuyết cây quyết định phổ thông; không cần thiết. |
| 11 | Khai triển Taylor bậc hai của XGBoost và Hessian | **REMOVE** | Bỏ khỏi Chương 2 | Lý thuyết giải thuật Chen & Guestrin 2016; không phục vụ lập luận của đề tài. |
| 12 | Công thức xác suất toàn phần của Soft Hierarchical | **KEEP** | Chương 3 (3.1--3.3) | Cốt lõi của mô hình phân cấp mềm $E_4 \to E_6$; bảo toàn tính chất xác suất. |
| 13 | Quy tắc quyết định tối thiểu hóa rủi ro Bayes $\hat{y}^* = \arg\min_i \sum_j \mathbf{C}(i,j) \hat{P}_j$ | **KEEP** | Chương 3 (3.4) | Cơ sở toán học của mô hình Cost-Sensitive $E_6$; điều khiển sự đánh đổi lâm sàng. |
| 14 | Xác suất tích lũy OLR $P(Y \le k \mid \mathbf{x}) = \sigma(\theta_{k+1} - \mathbf{w}^T \mathbf{x})$ | **KEEP** | Chương 3 (3.5) | Cốt lõi của mô hình thứ bậc $E_{7a}$; chứng minh các siêu phẳng quyết định song song. |
| 15 | Suy luận xác suất từng lớp trong Cumulative Binary (CBC) | **SIMPLIFY** | Chương 3 (3.6) | Hiệu số xác suất tích lũy; dùng để phân tích vi phạm tính đơn điệu. |
| 16 | Đạo hàm hàm hợp lý OLR và ma trận thông tin Fisher | **APPENDIX** | Chuyển Phụ lục D | Chi tiết thuật toán tối ưu L-BFGS-B; dành cho người muốn tái lập cài đặt. |
| 17 | Công thức tính Class MAE và Quadratic Weighted Kappa (QWK) | **KEEP** | Chương 3 (3.7--3.8) | Định nghĩa toán học của hai độ đo thứ bậc thực nghiệm. |

*Tổng kết:* Toàn bộ báo cáo V4 chỉ giữ lại đúng **11 công thức trọng tâm** trong phần nội dung chính, giảm hơn 65% so với bản cũ nhưng tăng độ mạch lạc và dễ đọc.

---

## 2. KẾ HOẠCH HỆ THỐNG HÌNH ẢNH (15 HÌNH CHẤT LƯỢNG CAO)

Tất cả các hình ảnh được tạo mới hoặc tinh chỉnh ở định dạng vector / PNG 300 DPI, sử dụng font sans-serif đồng nhất, chú thích tiếng Việt rõ ràng, màu sắc trang nhã (bảng màu khoa học Navy / Teal / Amber / Crimson):

| Mã hình | Tên tệp hình ảnh | Vị trí | Mục đích & Thông điệp cốt lõi | Nguồn gốc tạo hình |
| :--- | :--- | :--- | :--- | :--- |
| **Hình 1.1** | `fig1_problem_formulation.png` | Mục 1.2 | Sơ đồ toàn cảnh: Dữ liệu vào $\to$ Pipeline tiền xử lý $\to$ Đa trường phái mô hình $\to$ Phân tầng nguy cơ PHQ-9. | Matplotlib diagram chuẩn tắc, 300 DPI |
| **Hình 2.1** | `fig2_phq9_tiers.png` | Mục 2.1 | Thang đo PHQ-9 từ 0–27 điểm, 5 mức độ lâm sàng và ánh xạ sang không gian 3 nhóm $T_3$. | Matplotlib vector diagram |
| **Hình 2.2** | `fig2_gut_brain_metabolic_axis.png` | Mục 2.1 | Cơ chế y sinh học liên kết 4 hệ thống ngoại vi (viêm mạn tính, lipid, huyết áp, lối sống) với trầm cảm. | Matplotlib diagram y sinh |
| **Hình 2.3** | `fig2_imbalance_overlap.png` | Mục 2.2 | (a) Biểu đồ mất cân bằng lớp cực đoan (IR = 68.5:1); (b) Minh họa vùng chồng lấn phân phối Bayes giữa $C_0$ và $C_1$. | Dữ liệu thật $N=4.045$ & đường cong phân phối lý thuyết |
| **Hình 2.4** | `fig2_modeling_paradigms.png` | Mục 2.3 | So sánh cấu trúc hình học và cơ chế ra quyết định của 3 trường phái: Phẳng, Phân cấp và Thứ bậc. | Matplotlib comparative schema |
| **Hình 3.1** | `depsev_overall_architecture.png` | Mục 3.1 | Kiến trúc hệ thống tổng thể 7 module của khung học máy thực nghiệm DepSev-ML. | Sơ đồ kiến trúc khoa học |
| **Hình 3.2** | `fig3_2_data_pipeline.png` | Mục 3.2 | Quy trình 5 bước tiền xử lý in-fold và nguyên tắc cô lập bảo toàn Zero Data Leakage. | Matplotlib workflow diagram |
| **Hình 3.3** | `fig3_3_feature_groups.png` | Mục 3.3 | Sơ đồ phân tầng 4 nhóm đặc trưng lũy tiến $F_1 \to F_4$ (10, 19, 29 và 34 biến). | Matplotlib hierarchy diagram |
| **Hình 3.4** | `fig3_4_hard_hierarchical.png` | Mục 3.5 | Cơ chế định tuyến cứng 2 giai đoạn và vạch trần điểm nghẽn lan truyền sai số một chiều ($C_2^+ \to C_0$). | Matplotlib error-propagation schema |
| **Hình 3.5** | `fig3_5_soft_cost_sensitive.png` | Mục 3.5 | Kiến trúc phân cấp mềm kết hợp tích phân xác suất toàn phần và ma trận quyết định nhạy chi phí Bayes. | Matplotlib Bayes decision schema |
| **Hình 3.6** | `fig3_6_ordinal_classification.png` | Mục 3.5 | (a) Biến tiềm ẩn và ngưỡng cắt song song trong OLR; (b) Minh họa nguy cơ vi phạm tính đơn điệu trong CBC. | Matplotlib ordinal function plot |
| **Hình 3.7** | `fig3_7_evaluation_protocol.png` | Mục 3.6 | Sơ đồ hai tầng kiểm định: Nested 5-Fold Stratified CV trên Legacy và Temporal Validation trên 2015–2016. | Matplotlib protocol workflow |
| **Hình 4.1** | `feature_group_comparison_final_protocol.png` | Mục 4.2 | Biểu đồ cột thể hiện sự tăng trưởng Macro-F1 của 4 thuật toán qua 4 nhóm đặc trưng lũy tiến $F_1 \to F_4$. | `feature_ablation_results_final_protocol.csv` |
| **Hình 4.2** | `granularity_performance.png` | Mục 4.4 | Đồ thị suy giảm đột ngột của Macro-F1 và Balanced Accuracy khi tăng độ chi tiết nhãn từ $T_2$ đến $T_5$. | `results/final_report_granularity.md` |
| **Hình 4.3** | `overlap_pca_C0_C4.png` | Mục 4.5 | Hình chiếu PCA 2D không gian đặc trưng chứng minh sự bao phủ dày đặc giữa lớp bình thường $C_0$ và lớp nhẹ $C_1$. | Trích xuất từ 34 đặc trưng $N=4.045$ |
| **Hình 4.4** | `cost_tradeoff_curve.png` | Mục 4.8 | Đường cong đánh đổi lâm sàng sống còn giữa số ca bỏ sót bệnh nặng ($C_2^+ \to C_0$) và số ca báo động nhầm ($C_0 \to C_2^+$). | Trích xuất từ dự đoán $E_5, E_6, E_{6b}$ |
| **Hình 4.5** | `confusion_matrices_group1.png` | Mục 4.7 | Ma trận nhầm lẫn 4 mô hình đối xứng: $E_0$ (Flat RF), $E_1$ (Hard RF--RF), $E_5$ (Soft RF--LR) và $E_{6b}$ (Cost-Sensitive). | Out-of-fold predictions |
| **Hình 4.6** | `confusion_matrices_group2.png` | Mục 4.7 | Ma trận nhầm lẫn 2 mô hình thứ bậc: $E_{7a}$ (Balanced OLR) và $E_{7b}$ (Cumulative Binary RF). | Out-of-fold predictions |

---

## 3. KẾ HOẠCH BẢNG BIỂU (7 BẢNG TRỌNG TÂM TRONG THÂN BÀI)

| Mã bảng | Tên bảng | Vị trí | Các cột cần thiết | Nguồn dữ liệu thực nghiệm |
| :--- | :--- | :--- | :--- | :--- |
| **Bảng 2.1** | Tổng hợp so sánh các nghiên cứu liên quan | Mục 2.4 | Nghiên cứu, Nguồn dữ liệu, Không gian đích, Phương pháp, Hạn chế chính | Tổng hợp y văn quốc tế |
| **Bảng 3.1** | Phân bố lớp triệu chứng PHQ-9 trên Legacy Cohort | Mục 3.2 | Phân hoạch, Tổng N, $C_0$, $C_1$, $C_2$, $C_3$, $C_4$ | NHANES 2013–2014 ($N=5.054$) |
| **Bảng 3.2** | Quy mô và phân bố mẫu trên các chu kỳ NHANES | Mục 3.2 | Chu kỳ / Phân nhánh, Tổng N, Tỷ lệ \% từng lớp ($C_0 \to C_4$) | NHANES 2011–2016 ($N=14.581$) |
| **Bảng 3.3** | Tóm tắt cấu trúc 4 không gian đặc trưng $F_1 \to F_4$ | Mục 3.3 | Không gian, Số biến, Các nhóm biến thành phần, Ý nghĩa lâm sàng | `ablation_feature_inventory.csv` |
| **Bảng 4.1** | Kết quả thực nghiệm Feature Ablation ($F_1 \to F_4$ trên $T_5$) | Mục 4.2 | Mô hình, Không gian ($F_1 \to F_4$), Macro-F1, BalAcc, Weighted-F1, Acc | `feature_ablation_results.csv` |
| **Bảng 4.2** | Hiệu năng mô hình Random Forest theo độ chi tiết nhãn ($T_2 \to T_5$) | Mục 4.4 | Không gian mục tiêu, Số lớp $K$, Macro-F1, BalAcc, Accuracy | `final_report_granularity.md` |
| **Bảng 4.3** | Bảng so sánh tổng hợp Master Comparison ($E_0 \to E_7$) | Mục 4.7 | **Phần A:** Trường phái, Mô hình, Macro-F1, BalAcc, Acc, Class MAE, QWK.<br>**Phần B:** Recall $C_0, C_1, C_2^+$, Bỏ sót $C_2^+ \to C_0$, Báo động $C_0 \to C_2^+$ | `DepSev_Verified_Results_E0-E7.csv` |
| **Bảng 4.4** | Kết quả kiểm định ngoại suy theo dòng thời gian (Temporal Test) | Mục 4.9 | Mô hình ($T_3$), Macro-F1, BalAcc, Accuracy, Recall $C_0, C_1, C_2^+$, Class MAE | NHANES 2015–2016 ($N=4.888$) |

*Ghi chú:* Các bảng danh mục chi tiết 34 đặc trưng kèm mô tả mã biến, siêu tham số chi tiết và bảng số liệu theo từng fold kiểm định chéo được chuyển sang **Phụ lục A, B, C** để đảm bảo phần thân bài đạt độ súc tích cao nhất.

