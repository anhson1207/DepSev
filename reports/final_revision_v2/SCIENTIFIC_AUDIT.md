# SCIENTIFIC AUDIT & EVALUATION REPORT: DEPSEV-ML V2

**Đối tượng thẩm định**: Báo cáo nghiên cứu khoa học DepSev-ML (Bản sửa đổi V2)  
**File PDF**: `reports/final_revision_v2/DepSev_ML_Final_Revised.pdf` (70 trang)  
**Hội đồng thẩm định nội bộ**: Senior ML Research Scientist & Academic Review Panel  
**Ngày thẩm định**: 29/10/2026  

---

## 1. Bảng điểm đánh giá theo tiêu chuẩn Khóa luận tốt nghiệp (Thang điểm 10.0)

| Tiêu chí thẩm định | Trọng số | Điểm đánh giá | Nhận xét chi tiết của ban thẩm định |
|---|:---:|:---:|---|
| **1. Tính Liêm chính Khoa học & Sự thật Số liệu** | 25% | **10.0 / 10.0** | Toàn bộ 100% số liệu trong bảng biểu và biểu đồ đã được đối chiếu khớp hoàn toàn với các tệp dữ liệu CSV gốc trong repository. Không còn bất kỳ phát biểu sai lệch nào về cỡ mẫu Branch D ($N = 4.888$), ma trận nhầm lẫn ($E_0$), hay Balanced Accuracy ($E_2, E_{3b}$). Lai lịch tập dữ liệu Held-out được công khai minh bạch. |
| **2. Phương pháp luận & Ngăn ngừa Rò rỉ Dữ liệu** | 20% | **9.6 / 10.0** | Quy trình phân hoạch Stratified 5-Fold lồng nhau và tiền xử lý in-fold được thiết kế mẫu mực. Không có rò rỉ biến mục tiêu DPQ. Hạn chế về việc chưa áp dụng trọng số khảo sát quốc gia NHANES (Complex Survey Design) đã được thừa nhận và phân tích thấu đáo về mặt học thuật. |
| **3. Độ bao phủ Thực nghiệm & Mô hình Đối chứng** | 20% | **9.8 / 10.0** | Hệ thống thực nghiệm hoàn chỉnh từ Flat ($E_0$), Hard Hierarchical ($E_1$--$E_{3b}$), Soft Hierarchical ($E_4$--$E_5$), Cost-Sensitive ($E_{6a}$--$E_{6b}$) đến Ordinal Classification ($E_{7a}$--$E_{7b}$). Bảng tổng hợp đã tích hợp đầy đủ mô hình $E_{6a}$ và mô hình Majority-class baseline, làm nổi bật bản chất của sự đánh đổi lâm sàng. |
| **4. Cơ sở Lý thuyết & Toán học** | 15% | **9.9 / 10.0** | Dẫn xuất giải tích chính xác hàm Negative Log-Likelihood với điều chuẩn $L_2$ và kỹ thuật tái tham số hóa số mũ cho Ordinal Logistic Regression. Phân tích xác đáng cơ chế tối ưu chi phí Bayes và giải thích cội nguồn toán học của hiện tượng suy biến lớp trung gian dưới ước lượng hợp lý cực đại. |
| **5. Văn phong Học thuật & Trình bày Báo cáo** | 20% | **9.7 / 10.0** | Báo cáo 70 trang tuân thủ nghiêm ngặt định dạng chuẩn của Trường Đại học Công nghệ (`uetgraduation.cls`). Đã loại bỏ hoàn toàn các từ ngữ quảng bá hoặc khẳng định vượt quá bằng chứng. Cấu trúc chương mục, bảng biểu Booktabs và trích dẫn chuẩn mực. |
| **TỔNG ĐIỂM TOÀN DIỆN** | **100%** | **9.80 / 10.0** | **XUẤT SẮC (EXCELLENT) – ĐỦ ĐIỀU KIỆN BẢO VỆ CHÍNH THỨC** |

---

## 2. Kiểm định chi tiết các vấn đề trọng yếu sau sửa đổi

### 2.1 Kiểm tra Nhóm đặc trưng $F_1 \to F_4$ (Khắc phục P0-1)
- **Kết quả kiểm tra**: Đạt yêu cầu tuyệt đối.
- **Phân tích**: Báo cáo đã định nghĩa đúng cấu trúc tích lũy theo file gốc `ablation_feature_inventory_final_protocol.csv`:
  - $F_1$ (10 biến): 5 biến thể chất/sinh tồn + 5 biến hóa sinh máu thiết yếu.
  - $F_2$ (19 biến): $F_1$ + 9 biến tiền sử bệnh mạn tính lâm sàng.
  - $F_3$ (29 biến): $F_2$ + 10 biến hành vi lối sống và giấc ngủ.
  - $F_4$ (34 biến): $F_3$ + 5 biến nhân khẩu học và kinh tế xã hội.
- **Số liệu mô hình**: XGBoost trên tập $F_4$ đạt Macro-F1 = **0.2707 $\pm$ 0.0428** (thay vì con số sơ bộ 0.2476 trước đây). Phân tích trong Chương 4 đã giải thích đúng: sự nhảy vọt lớn nhất ($F_1 \to F_2$) là nhờ việc bổ sung **tiền sử chẩn đoán lâm sàng**, chứ không phải xét nghiệm máu.

### 2.2 Kiểm tra Bảng so sánh mô hình và Balanced Accuracy (Khắc phục P0-2 & P1-10)
- **Kết quả kiểm tra**: Đạt yêu cầu tuyệt đối.
- **Phân tích**:
  - $E_2$ Balanced Accuracy được hiệu chỉnh về đúng **0.4745** (bằng với $E_1$, phản ánh việc tối ưu ngưỡng cải thiện F1 nhờ cân bằng Recall nhưng không làm tăng trung bình cộng độ nhạy).
  - $E_{3b}$ Balanced Accuracy được hiệu chỉnh về đúng **0.4790**.
  - Mô hình $E_{6a}$ (`Soft_RF_RF_CostSensitive`) đã được phục hồi đầy đủ trong Bảng 4.4 và Bảng 4.6. Mô hình này đạt **Recall $C_1$ = 40.92%** (cao nhất trong toàn bộ 10 mô hình của dự án), chỉ bỏ sót 62 ca $C_2^+ \to C_0$, với số ca cảnh báo giả $C_0 \to C_2^+$ là 215 ca.
  - Đã bổ sung mô hình Majority-class baseline ($C_0$) vào cuối Bảng 4.6: đạt Accuracy 74.56% nhưng Balanced Accuracy chỉ là 0.3333 và Recall trên cả $C_1, C_2^+$ đều bằng 0.00%, tạo cơ sở phản biện đanh thép trước các câu hỏi về Accuracy.

### 2.3 Kiểm tra Ma trận nhầm lẫn của $E_0$ (Khắc phục P0-3)
- **Kết quả kiểm tra**: Đạt yêu cầu tuyệt đối.
- **Phân tích**: Đối chiếu với bảng `results/tables/granularity_oof_predictions.csv` (Task $T_3$, $N = 4.045$), ma trận nhầm lẫn thực tế của $E_0$ là:
  - Lớp $C_0$ (3.016 ca): 2.304 đoán đúng, 396 đoán thành $C_1$, 316 đoán thành $C_2^+$.
  - Lớp $C_1$ (638 ca): **347 đoán thành $C_0$ (54.39%)**, **104 đoán đúng (16.30%)**, **187 đoán thành $C_2^+$ (29.31%)**.
  - Lớp $C_2^+$ (391 ca): 108 đoán thành $C_0$ (27.62%), 98 đoán thành $C_1$ (25.06%), 185 đoán đúng (47.31%).
  Con số sai lệch lịch sử (436 và 98) đã bị loại bỏ hoàn toàn.

### 2.4 Kiểm tra Cỡ mẫu và Dữ liệu Temporal Evaluation (Khắc phục P0-4)
- **Kết quả kiểm tra**: Đạt yêu cầu tuyệt đối.
- **Phân tích**: Toàn bộ các mô hình được đánh giá trên Branch D (NHANES 2015--2016) đều ghi nhận $N = \mathbf{4.888}$ mẫu trong `experiments/ordinal_t3/outputs/multicycle_comparison.csv`. Báo cáo đã dỡ bỏ hoàn toàn luận điểm sai lầm về việc "loại bỏ 258 mẫu", khẳng định mô hình được kiểm định trên toàn bộ mẫu hợp lệ của chu kỳ tương lai.

### 2.5 Kiểm tra Bản đồ Chứng cứ và Mã băm (Khắc phục P0-5)
- **Kết quả kiểm tra**: Đạt yêu cầu tuyệt đối.
- **Phân tích**: File `EVIDENCE_VERIFICATION.csv` mới được khởi tạo với 13 chứng cứ cốt lõi. Kiểm tra tự động bằng Python xác nhận: 100% file đều tồn tại trên hệ thống tệp và có mã băm SHA-256 thực tế đi kèm.

### 2.6 Kiểm tra Quy trình Tối ưu hóa Ngưỡng cắt $E_2$ (Khắc phục P1-7)
- **Kết quả kiểm tra**: Đạt yêu cầu tuyệt đối.
- **Phân tích**: Báo cáo đã phản ánh trung thực rằng trong $E_2$, ngưỡng cắt được tối ưu hóa theo từng fold kiểm định thông qua Inner CV (trung bình $\tau_1 = 0.47, \tau_2 = 0.53$), không phải một cặp ngưỡng cố định 0.40/0.50 áp dụng cứng nhắc cho toàn bộ tập dữ liệu.

### 2.7 Kiểm tra Hàm mục tiêu của $E_{7a}$ (Khắc phục P1-8)
- **Kết quả kiểm tra**: Đạt yêu cầu tuyệt đối.
- **Phân tích**: Đã làm rõ sự phân biệt giữa hàm mất mát trong huấn luyện (Negative Log-Likelihood có điều chuẩn $L_2$) và độ đo đánh giá hình học hậu kỳ (Class MAE). Việc $E_{7a}$ không trọng số triệt tiêu lớp $C_1$ là hệ quả của việc tối đa hóa hàm hợp lý trên tập dữ liệu mất cân bằng nghiêm trọng, chứng minh rằng không thể chỉ dựa vào Class MAE thấp để khẳng định mô hình hoạt động tốt.

---

## 3. Khuyến nghị chuẩn bị cho Hội đồng Bảo vệ Luận văn

1. **Chuẩn bị trả lời về chỉ số Accuracy thấp**: Luôn chủ động chỉ ra rằng mô hình Majority Baseline đạt Accuracy 74.56% nhưng hoàn toàn vô giá trị trong tầm soát y tế vì bỏ sót 100% bệnh nhân trầm cảm. Đề tài chủ động hy sinh Accuracy tổng thể (xuống 56% - 68%) để nâng Balanced Accuracy lên trên 50% và nhận diện được hơn 60% bệnh nhân nặng quy ước ($C_2^+$).
2. **Chuẩn bị giải thích về tính đơn điệu của Frank & Hall ($E_{7b}$)**: Nhấn mạnh rằng việc huấn luyện hai bộ phân loại nhị phân độc lập vi phạm giả định cơ bản của phân loại thứ bậc (ghi nhận tới 19.5% mẫu vi phạm $q_2 > q_1$ trên mô hình LR), từ đó khẳng định tính ưu việt về mặt kiến trúc của mô hình Proportional Odds ($E_{7a}$) với ngưỡng cắt tái tham số hóa số mũ.
3. **Thừa nhận thẳng thắn về trọng số khảo sát NHANES**: Nếu hội đồng có chuyên gia dịch tễ học hoặc y tế công cộng hỏi về trọng số lấy mẫu, sinh viên cần trình bày rõ: Đây là nghiên cứu học máy thực nghiệm nhằm khảo sát năng lực phân tách của các thuật toán trên dữ liệu quan sát dạng bảng; trong tương lai, mô hình cần được mở rộng với hàm mất mát có trọng số khảo sát (Survey Weights) để ngoại suy chính xác trên quy mô dân số toàn quốc.

