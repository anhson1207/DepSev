# BÁO CÁO KIỂM TOÁN SỐ LIỆU VÀ TÍNH ĐÚNG ĐẮN KHOA HỌC (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/audit/SCIENTIFIC_NUMERICAL_AUDIT.md`  
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Ngày thực hiện:** 31/10/2026  
**Người thực hiện:** Antigravity Research Team  

---

## 1. TỔNG QUAN VÀ MỤC ĐÍCH KIỂM TOÁN

Trong các phiên bản trước (từ V4 đến V9), báo cáo xuất hiện một số mâu thuẫn số liệu giữa các chương, sự thiếu nhất quán giữa các bảng thực nghiệm, và các phát biểu diễn dịch quá mức về mặt lâm sàng/toán học. Bản kiểm toán này ghi nhận toàn bộ quá trình rà soát, đối chiếu số liệu từ mã nguồn thực nghiệm (`results/` và các log thực nghiệm) sang văn bản báo cáo V10, bảo đảm tính trung thực học thuật và chuẩn mực khoa học cao nhất.

---

## 2. ĐỐI CHIẾU VÀ HÒA GIẢI CÁC MÂU THUẪN SỐ LIỆU TRỌNG YẾU

### 2.1. Hòa giải giá trị hiệu năng Random Forest: Nested CV vs. Frozen Model vs. Granularity

* **Hiện tượng tại V9:**
  - Bảng so sánh thuật toán cơ sở ghi: `Random Forest (Frozen) = 0.2755 ± 0.0270`.
  - Bảng nghiên cứu Ensemble ghi: `RF Baseline = 0.2741 ± 0.0252`.
  - Bảng phân tích phân giải mục tiêu ($T_5$) ghi: `Random Forest = 0.2630`.
  - Điều này gây thắc mắc cho người phản biện: tại sao cùng một mô hình Random Forest lại có ba điểm số Macro-F1 khác nhau?

* **Nguyên nhân kỹ thuật:**
  1. Trong thực nghiệm so sánh thuật toán (Nested Cross-Validation 10x5 folds), mô hình Random Forest được điều chỉnh siêu tham số độc lập trong từng fold nội (`Random Forest (Tuned)`). Giá trị out-of-fold đạt **Macro-F1 = 0.2755 ± 0.0270**, **Balanced Accuracy = 0.2983 ± 0.0325**, **Weighted-F1 = 0.6437 ± 0.0087**.
  2. Sau khi Nested CV chỉ ra Random Forest có cấu hình tốt nhất xung quanh bộ siêu tham số (`n_estimators = 150`, `max_depth = 8`, `max_features = 0.5`, `min_samples_split = 10`, `min_samples_leaf = 8`, `class_weight = 'balanced_subsample'`), cấu hình này được "đóng băng" (Frozen Spec) làm baseline cố định cho các nghiên cứu tiếp theo (Ensemble và Hierarchical) để tiết kiệm chi phí tính toán và bảo đảm tính tái lập đồng nhất. Khi đánh giá mô hình đóng băng này trên 10-fold CV chuẩn của nghiên cứu Ensemble, Macro-F1 đạt **0.2741 ± 0.0252**.
  3. Khi đánh giá trên một phân chia phân tầng cố định trong nghiên cứu Granularity ($T_5$), Macro-F1 đạt **0.2630**, Balanced Accuracy = **0.2889**, Weighted-F1 = **0.6380**.

* **Biện pháp xử lý trong V10:**
  - Trong [Bảng 4.2](file:///Users/nguyenanhson/DepSev-ML/reports/final_scientific_v10/source/tables/table_nested_cv_comparison.tex#L16), dòng 14 đã được đổi tên chuẩn xác thành: `\textbf{Random Forest (Tuned)}`.
  - Thêm chú thích chân trang giải thích rõ sự khác biệt giữa phiên bản Tuned in-fold trong Nested CV ($0.2755$) và phiên bản Frozen Spec ($0.2741$) dùng làm mô hình cơ sở cho các tầng phân cấp.
  - Đồng bộ hóa các đoạn diễn giải văn bản trong Mục 4.3 và 4.4, loại bỏ hoàn toàn sự mập mờ thuật ngữ.

---

### 2.2. Kiểm định thực nghiệm Feature Ablation ($F_1 \to F_4$)

* **Hiện tượng tại V9:**
  - Báo cáo so sánh chéo kết quả giữa các mô hình khác nhau (ví dụ: so sánh Logistic Regression ở $F_1$ với XGBoost ở $F_4$), tạo cảm giác ngụy biện về sự cải thiện của nhóm đặc trưng.

* **Dữ liệu thực nghiệm xác thực (10-fold CV trên Legacy Development, $N=4{,}880$):**
  - **Logistic Regression:**
    - $F_1$ (10 đặc trưng nhân khẩu học): Macro-F1 = $0.1900 \pm 0.0164$, Balanced Acc = $0.2420 \pm 0.0299$.
    - $F_2$ (+9 sinh trắc & huyết áp): Macro-F1 = $0.2338 \pm 0.0041$, Balanced Acc = $0.2942 \pm 0.0299$.
    - $F_3$ (+10 huyết học CBC): Macro-F1 = $0.2390 \pm 0.0084$, Balanced Acc = $0.3194 \pm 0.0284$.
    - $F_4$ (+5 sinh hóa & lipid): Macro-F1 = $0.2399 \pm 0.0148$, Balanced Acc = $0.3216 \pm 0.0508$.
    - *Mức cải thiện nội bộ LR:* $+0.0499$ Macro-F1 ($+26.3\%$).
  - **Decision Tree:**
    - $F_1$: Macro-F1 = $0.2072 \pm 0.0131$, Balanced Acc = $0.2227 \pm 0.0177$.
    - $F_4$: Macro-F1 = $0.2255 \pm 0.0101$, Balanced Acc = $0.2348 \pm 0.0121$.
    - *Mức cải thiện nội bộ DT:* $+0.0183$ Macro-F1 ($+8.8\%$).
  - **Random Forest:**
    - $F_1$: Macro-F1 = $0.2052 \pm 0.0156$, Balanced Acc = $0.2195 \pm 0.0178$.
    - $F_2$: Macro-F1 = $0.2163 \pm 0.0172$, Balanced Acc = $0.2308 \pm 0.0194$.
    - $F_3$: Macro-F1 = $0.2215 \pm 0.0253$, Balanced Acc = $0.2349 \pm 0.0282$.
    - $F_4$: Macro-F1 = $0.2232 \pm 0.0279$, Balanced Acc = $0.2367 \pm 0.0308$.
    - *Mức cải thiện nội bộ RF:* $+0.0180$ Macro-F1 ($+8.8\%$).
  - **XGBoost:**
    - $F_1$: Macro-F1 = $0.2173 \pm 0.0085$, Balanced Acc = $0.2366 \pm 0.0140$.
    - $F_2$: Macro-F1 = $0.2458 \pm 0.0099$, Balanced Acc = $0.2647 \pm 0.0132$.
    - $F_3$: Macro-F1 = $0.2589 \pm 0.0145$, Balanced Acc = $0.2812 \pm 0.0189$.
    - $F_4$: Macro-F1 = $0.2707 \pm 0.0210$, Balanced Acc = $0.2954 \pm 0.0265$.
    - *Mức cải thiện nội bộ XGBoost:* $+0.0534$ Macro-F1 ($+24.6\%$).

* **Kết luận khoa học đã chuẩn hóa trong V10:**
  - Sự đóng góp của các nhóm đặc trưng cận lâm sàng ($F_2 \to F_4$) được kết luận dựa trên so sánh *cùng họ mô hình* (within-model comparison).
  - Khẳng định rõ: Nhóm cận lâm sàng mang lại thông tin gia tăng có ý nghĩa thống kê đối với các mô hình phi tuyến có khả năng khai thác tương tác phức tạp (XGBoost tăng $0.0534$), nhưng bản thân các chỉ số này không đủ để tạo bước nhảy vọt độc lập nếu thiếu các thông tin tâm lý học hành vi.

---

### 2.3. Kiểm định số liệu bỏ sót ca nặng lâm sàng ($C_2^+ \to C_0$)

* **Hiện tượng tại V9:**
  - Có đoạn văn ghi "giảm $74.2\%$ số ca bỏ sót, chỉ còn 31 ca", trong khi bảng ma trận nhầm lẫn out-of-fold của toàn bộ tập dữ liệu lại thể hiện số liệu khác.

* **Đối chiếu số liệu out-of-fold thực tế ($N = 4{,}880$, số ca $C_2^+ = 422$):**
  - **Mô hình $E_1$ (Hard RF--RF Default):**
    - Số ca $C_2^+$ bị dự đoán nhầm thành $C_0$: **120 ca** ($28.44\%$ tổng số ca nặng).
  - **Mô hình $E_2$ (Hard RF--RF Tuned, tối ưu Macro-F1 = 0.4586):**
    - Số ca $C_2^+$ bị dự đoán nhầm thành $C_0$: **98 ca** ($23.22\%$ tổng số ca nặng).
  - **Mô hình $E_{6b}$ (Soft Bayes Risk với ma trận tổn thất $M_1$):**
    - Số ca $C_2^+$ bị dự đoán nhầm thành $C_0$: **62 ca** ($14.69\%$ tổng số ca nặng).
    - So với $E_1$: Giảm từ 120 xuống 62 ca (giảm **48.33%**, tương đương 58 ca được cứu khỏi việc bị bỏ sót hoàn toàn).
    - So với $E_2$: Giảm từ 98 xuống 62 ca (giảm **36.73%**, tương đương 36 ca).

* **Biện pháp xử lý trong V10:**
  - Đã chỉnh sửa toàn bộ các câu chữ trong Chương 4 và Chương 5 để phản ánh chính xác con số thực nghiệm: **62 ca bỏ sót** (giảm 48.33% so với $E_1$ và 36.73% so với $E_2$).
  - Tuyệt đối không sử dụng các con số ước lượng thiếu kiểm chứng (31 ca / 74.2%).

---

### 2.4. Phân phối mẫu và dữ liệu phân tầng (NHANES Cohort)

* **Tổng cỡ mẫu nghiên cứu:** $N = 4{,}880$ quan sát từ NHANES (2013--2018).
* **Phân chia tập dữ liệu:**
  - Tập huấn luyện & phát triển nội bộ: $N_{\text{train}} = 3{,}904$ ($80.0\%$).
  - Tập kiểm thử độc lập: $N_{\text{test}} = 976$ ($20.0\%$).
* **Phân bố 5 lớp ($T_5$):**
  - $C_0$ (Không trầm cảm, 0--4 điểm): $n = 3{,}784$ ($77.54\%$).
  - $C_1$ (Nhẹ, 5--9 điểm): $n = 674$ ($13.81\%$).
  - $C_2$ (Vừa, 10--14 điểm): $n = 255$ ($5.23\%$).
  - $C_3$ (Trung bình nặng, 15--19 điểm): $n = 116$ ($2.38\%$).
  - $C_4$ (Nặng, 20--27 điểm): $n = 51$ ($1.05\%$).
* **Phân bố 3 mức lâm sàng ($T_3$):**
  - $C_0$ (Không trầm cảm): $n = 3{,}784$ ($77.54\%$).
  - $C_1$ (Triệu chứng nhẹ): $n = 674$ ($13.81\%$).
  - $C_2^+$ (Cần can thiệp lâm sàng, $C_2 \cup C_3 \cup C_4$): $n = 422$ ($8.65\%$).
* **Phân bố tập kiểm thử độc lập ($N_{\text{test}} = 976$):**
  - $C_0$: $n = 757$ ($77.56\%$).
  - $C_1$: $n = 135$ ($13.83\%$).
  - $C_2^+$: $n = 84$ ($8.61\%$).

---

## 3. KẾT QUẢ CÁC HƯỚNG MÔ HÌNH PHÂN CẤP VÀ THỨ BẬC ($E_0 \to E_7$)

Bảng đối chiếu tổng hợp các cấu hình trên bài toán $T_3$ (10-fold CV out-of-fold):

| Ký hiệu | Tên cấu hình / Thuật toán | Macro-F1 (Mean $\pm$ SD) | Balanced Acc (Mean $\pm$ SD) | Weighted F1 (Mean $\pm$ SD) | $C_2^+ \to C_0$ (Ca bỏ sót) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| $E_0$ | Flat Random Forest Baseline | $0.4497 \pm 0.0142$ | $0.4908 \pm 0.0181$ | $0.7410 \pm 0.0075$ | 114 |
| $E_1$ | Hard Hierarchical (RF--RF Default) | $0.4431 \pm 0.0135$ | $0.4851 \pm 0.0176$ | $0.7385 \pm 0.0080$ | 120 |
| $E_2$ | Hard Hierarchical (RF--RF Tuned) | $\mathbf{0.4586 \pm 0.0124}$ | $0.4962 \pm 0.0169$ | $\mathbf{0.7512 \pm 0.0068}$ | 98 |
| $E_3$ | Hard Hierarchical (RF--XGBoost) | $0.4521 \pm 0.0130$ | $0.4912 \pm 0.0174$ | $0.7450 \pm 0.0072$ | 104 |
| $E_4$ | Hard Hierarchical (RF--Ridge) | $0.4385 \pm 0.0148$ | $0.4795 \pm 0.0188$ | $0.7320 \pm 0.0085$ | 128 |
| $E_5$ | Soft Hierarchical Routing | $0.4489 \pm 0.0138$ | $0.4920 \pm 0.0171$ | $0.7425 \pm 0.0074$ | 110 |
| $E_{6a}$ | Soft Hierarchical + Bayes Risk ($M_0$) | $0.4450 \pm 0.0141$ | $0.4950 \pm 0.0168$ | $0.7390 \pm 0.0078$ | 92 |
| $E_{6b}$ | Soft Hierarchical + Bayes Risk ($M_1$) | $0.4215 \pm 0.0150$ | $0.5025 \pm 0.0163$ | $0.7180 \pm 0.0089$ | $\mathbf{62}$ |
| $E_{7a}$ | Balanced Ordinal Logistic Regression | $0.4312 \pm 0.0145$ | $\mathbf{0.5151 \pm 0.0152}$ | $0.7240 \pm 0.0082$ | 74 |
| $E_{7b}$ | Cost-Sensitive Ordinal Logistic Reg. | $0.4180 \pm 0.0155$ | $0.5085 \pm 0.0159$ | $0.7115 \pm 0.0091$ | 68 |

---

## 4. TIÊU CHUẨN KHOA HỌC ĐƯỢC THIẾT LẬP TRONG BẢN BÁO CÁO V10

1. **Minh bạch về khoảng tin cậy:** Mọi chỉ số đánh giá tổng kết đều được ghi kèm độ lệch chuẩn qua 10 fold ($\text{Mean} \pm \text{SD}$), giúp hội đồng phản biện phân định rõ ràng giữa biến thiên ngẫu nhiên và sự cải thiện thực chất.
2. **Phân định rõ Fold-Mean vs. Pooled OOF:** Báo cáo ghi rõ: chỉ số trung bình lấy qua các fold kiểm định chéo phản ánh độ ổn định của thuật toán, trong khi ma trận nhầm lẫn tổng hợp trên toàn bộ mẫu Out-of-Fold phản ánh hành vi phân phối thực tế của hệ thống.
3. **Tiết chế các khẳng định khoa học:**
   - Không tuyên bố bất kỳ mô hình nào "chiến thắng toàn diện" hay "tối ưu tuyệt đối".
   - Trình bày rõ sự đánh đổi (Trade-off): $E_2$ tối ưu Macro-F1 ($0.4586$) nhưng bỏ sót 98 ca nặng; $E_{6b}$ bảo vệ an toàn ca nặng (chỉ sót 62 ca) nhưng phải đánh đổi bằng việc giảm Macro-F1 xuống $0.4215$; $E_{7a}$ bảo đảm thứ bậc thống kê và đạt Balanced Accuracy cao nhất ($0.5151$).
4. **Trung thực về dữ liệu cận lâm sàng:** Thừa nhận thẳng thắn trong Chương 5 rằng các chỉ số sinh lý - hóa sinh đơn lẻ không thể thay thế cho đánh giá tâm thần chuyên sâu, và sai số phân loại một phần xuất phát từ bản chất nhãn tự báo cáo (self-reported) của thang đo PHQ-9.

