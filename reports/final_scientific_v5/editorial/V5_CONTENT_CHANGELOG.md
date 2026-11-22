# GHI CHÉP THAY ĐỔI NỘI DUNG VÀ BIÊN TẬP HỌC THUẬT (PHIÊN BẢN V5)

Bản ghi chép này lưu trữ toàn bộ các chỉnh sửa lớn về mặt nội dung, phương pháp luận và trình bày giữa Phiên bản V4 và Phiên bản V5 của Báo cáo Dự án Khoa học DepSev-ML.

---

## 1. Mạch dẫn dắt và Logic nghiên cứu ($T_5 \to T_3$)

- **Chương 1 (Mục 1.5):** 
  - Bổ sung cấu trúc 2 giai đoạn kế tiếp nhau của đề tài: Giai đoạn 1 ($T_5$) khảo sát trần hiệu năng của bài toán 5 lớp chuẩn; Giai đoạn 2 ($T_3$) tái cấu trúc sang không gian 3 nhóm lâm sàng thực dụng và phát triển kiến trúc phân cấp hai bước.
  - Bổ sung Hình 1.2 (`fig_research_progression_t5_to_t3.png`) làm sơ đồ chỉ đường (roadmap) trực quan cho toàn bộ báo cáo.
  - Nhấn mạnh nguyên tắc: Stage 2 được huấn luyện chỉ trên các mẫu có nhãn thật $\text{PHQ-9} \ge 5$, và $T_3$ là bài toán độc lập với không gian nhãn tái cấu trúc, không phải là kết quả cải thiện trực tiếp trên cùng không gian 5 lớp.
- **Chương 4 (Mục 4.6 mới):** 
  - Bổ sung tiểu mục mang tính bước ngoặt: `Mục 4.6: Chuyển dịch nghiên cứu: Từ rào cản thông tin T5 đến tái cấu trúc bài toán T3`.
  - Tổng kết 4 bằng chứng thực nghiệm trên $T_5$ (Ablation, Nested CV, Ensembles, Multi-cycle) để lý giải vì sao mô hình 5 lớp phẳng bị bão hòa quanh Macro-F1 = 0.27.
  - Phân tích nhu cầu sàng lọc y tế thực tế để dẫn dắt chuyển dịch sang $T_3$.

---

## 2. Giới thiệu các thuật toán học máy cơ sở (Chương 2)

- Bổ sung `Mục 2.3: Các thuật toán học máy cơ sở trong DepSev-ML`.
- Trình bày 4 thuật toán theo cấu trúc 4 phần thống nhất:
  1. **Decision Tree (Cây quyết định):** Nguyên lý rẽ nhánh Gini Impurity, ưu điểm diễn giải logic `IF-THEN`, nhược điểm phương sai cao và dễ quá khớp, vai trò làm weak learner nền tảng. Kèm Hình 2.4 minh họa.
  2. **Logistic Regression (Hồi quy Logistic):** Nguyên lý hàm Sigmoid/Softmax, ưu điểm tính toán nhanh và diễn giải Odds Ratio, nhược điểm giới hạn tuyến tính, vai trò làm baseline chuẩn trong y học chứng cứ.
  3. **Random Forest (Rừng ngẫu nhiên):** Nguyên lý Bagging, Bootstrap, Random Subspace và Soft Voting; ưu điểm kiểm soát phương sai và chống quá khớp; vai trò làm backbone baseline xuyên suốt đề tài. Kèm Hình 2.5 minh họa.
  4. **XGBoost (eXtreme Gradient Boosting):** Nguyên lý Gradient Boosting, tối ưu phần dư Taylor bậc 2; ưu điểm xấp xỉ phi tuyến mạnh; nhược điểm nhạy cảm mất cân bằng lớp; vai trò kiểm tra trần hiệu năng của boosting.

---

## 3. Quy trình 8 bước áp dụng Random Forest (Chương 3)

- Bổ sung `Mục 3.5: Xây dựng mô hình Random Forest cơ sở`.
- Chi tiết hóa 8 bước thực nghiệm liên hoàn:
  - **Bước 1:** Chuẩn bị dữ liệu NHANES ($N = 4.045$, $d = 34$ biến $F_4$), phân phối nhãn 5 lớp, quy trình điền khuyết in-fold bảo toàn Zero Data Leakage.
  - **Bước 2:** Phân chia dữ liệu Stratified 5-Fold cố định seed 42 và Nested CV (5 outer, 3 inner folds).
  - **Bước 3:** Huấn luyện cây con, tiêu chí dừng sớm (`max_depth=8`, `min_samples_split=10`, `min_samples_leaf=8`).
  - **Bước 4:** Từ cây đến rừng: 150 cây qua Bootstrap và ngẫu nhiên hóa đặc trưng `max_features=0.5`, tổng hợp Soft Voting.
  - **Bước 5:** Khóa cấu hình chuẩn Frozen Random Forest Spec với `class_weight='balanced_subsample'`.
  - **Bước 6:** Tinh chỉnh và chọn mô hình cơ sở: Lý do chọn Macro-F1 thay vì Accuracy; đối sánh Nested CV với LR (0.2407), DT (0.2079), XGB (0.2621) và RF (0.2755).
  - **Bước 7:** Đánh giá hiệu năng OOF và hạn chế: Phân tích sự sụp đổ của lớp $C_1$ (Recall 16.29\%) do chồng lấn sinh lý.
  - **Bước 8:** Chuyển giao RF sang kiến trúc phân cấp $T_3$: Làm baseline phẳng $E_0$, bộ phân loại Stage 1 và Stage 2 trong phân cấp cứng $E_1$; phân tích điểm nghẽn 120 ca nặng bị bỏ sót và tiền đề cho $E_2 \to E_6$.

---

## 4. Tái thiết kế toàn bộ hệ thống hình ảnh (8 hình trọng tâm)

- Đồ họa tối giản, chuẩn mực, loại bỏ các đoạn văn bản dài dòng bên trong hình, chỉ sử dụng các từ khóa súc tích.
- Font chữ $\ge 9\text{pt}$ đọc rõ trên bản in A4, độ phân giải 300 DPI.
- Tông màu học thuật: Xanh dương (#1A365D, #2563EB), Xanh lá (#047857, #10B981), Vàng cam (#B45309, #F59E0B), Đỏ cảnh báo (#B91C1C, #EF4444).
- 8 hình gồm:
  1. `fig_research_progression_t5_to_t3.png` (Hình 1.2)
  2. `fig2_decision_tree_concept.png` (Hình 2.4)
  3. `fig2_random_forest_architecture.png` (Hình 2.5)
  4. `fig3_infold_cv_pipeline.png` (Hình 3.2)
  5. `fig3_4_hard_hierarchical.png` (Hình 3.5)
  6. `fig3_5_soft_cost_sensitive.png` (Hình 3.6)
  7. `fig4_algorithm_comparison_t5_t3.png` (Hình 4.6)
  8. `fig4_confusion_error_analysis.png` (Hình 4.7)

---

## 5. Rà soát và lược bỏ công thức toán

- Loại bỏ các dẫn xuất vi tích phân sách giáo khoa cơ bản (như đạo hàm gradient descent cơ bản, khai triển Taylor chi tiết từng biến, định nghĩa xác suất sơ cấp).
- Giữ lại các công thức toán cốt lõi:
  - Tỷ số mất cân bằng và trọng số nghịch đảo tần số lớp.
  - Chuỗi xác suất toàn phần trong phân cấp mềm.
  - Quy tắc quyết định Bayes với ma trận chi phí.
  - Hàm liên kết tích lũy trong Hồi quy Logistic thứ bậc (OLR).
  - Giải thuật isotonic clipping xử lý vi phạm đơn điệu trong CBC.

