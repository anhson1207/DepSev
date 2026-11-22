# BÁO CÁO CÁC GIỚI HẠN VÀ VẤN ĐỀ TỒN ĐỌNG (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/audit/UNRESOLVED_ISSUES.md`  
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Ngày thực hiện:** 31/10/2026  
**Người thực hiện:** Antigravity Research Team  

---

## 1. MỤC ĐÍCH TÀI LIỆU

Một nghiên cứu khoa học trung thực và chuẩn mực bắt buộc phải nêu rõ các giới hạn phương pháp luận, phạm vi ứng dụng thực tế và những câu hỏi mở chưa thể giải quyết triệt để trong khuôn khổ đề tài. Báo cáo này tổng hợp đầy đủ các vấn đề còn tồn đọng nhằm cung cấp cho hội đồng phản biện cái nhìn khách quan, đa chiều về dự án DepSev-ML.

---

## 2. GIỚI HẠN BẢN CHẤT CỦA BỘ DỮ LIỆU NHANES

### 2.1. Thiết kế nghiên cứu cắt ngang (Cross-Sectional Design)
- **Thực trạng:** Dữ liệu NHANES được thu thập theo phương pháp cắt ngang qua các chu kỳ khảo sát độc lập (2013--2014, 2015--2016, 2017--2018). Mỗi cá nhân chỉ được lấy mẫu sinh học và trả lời bảng câu hỏi tại một thời điểm duy nhất, không có theo dõi dọc (longitudinal follow-up).
- **Hệ quả nghiên cứu:** Mặc dù nghiên cứu đã thực hiện kiểm định ngoại mẫu theo thời gian (Temporal Out-of-Sample Validation) để đánh giá khả năng chống trôi dạt phân phối (distribution drift), mô hình hiện tại **chỉ dự đoán trạng thái triệu chứng tại thời điểm đo**, không thể suy diễn quan hệ nhân quả (causality) hay dự báo diễn tiến phục hồi/tái phát bệnh của bệnh nhân trong tương lai.

### 2.2. Tính nhiễu của nhãn tự báo cáo (Self-Reported PHQ-9)
- **Thực trạng:** Nhãn mục tiêu được xây dựng dựa trên bảng khảo sát PHQ-9 gồm 9 câu hỏi tự đánh giá trong vòng 2 tuần qua.
- **Hệ quả nghiên cứu:** Bảng tự khai có thể bị ảnh hưởng bởi thiên kiến nhận thức, tâm lý e ngại kỳ thị xã hội (stigma), hoặc biến động cảm xúc nhất thời tại ngày phỏng vấn. PHQ-9 là công cụ sàng lọc dịch tễ học, không phải chẩn đoán xác định từ bác sĩ tâm thần thông qua phỏng vấn lâm sàng có cấu trúc (như SCID hoặc MINI). Đây là một nguồn nhiễu nội tại (label noise) khó có thể triệt tiêu hoàn toàn bằng các thuật toán học máy.

---

## 3. GIỚI HẠN CỦA CÁC ĐẶC TRƯNG CẬN LÂM SÀNG DẠNG BẢNG

### 3.1. Hiện tượng "trần hiệu năng" (Performance Ceiling)
- **Quan sát thực nghiệm:** Dù đã áp dụng nhiều mô hình hiện đại (Nested CV Random Forest, XGBoost, Stacking Ensemble, Hierarchical Classifiers, Ordinal Regression), Macro-F1 trên bài toán 5 lớp ($T_5$) dao động trong khoảng $0.27$--$0.28$, và trên bài toán 3 mức ($T_3$) dao động trong khoảng $0.44$--$0.46$.
- **Bản chất y sinh:** Các chỉ số sinh lý thông thường (công thức máu toàn phần CBC, lipid máu, huyết áp, chỉ số khối cơ thể BMI) chỉ phản ánh phản ứng viêm nhẹ mạn tính hoặc rối loạn chuyển hóa thứ phát đi kèm với stress kéo dài. Bản thân các chỉ số này không chứa đặc trưng sinh học thần kinh trực tiếp (direct neurobiological markers) của bệnh trầm cảm. Do đó, việc kỳ vọng một mô hình Machine Learning đạt độ chính xác hoàn hảo chỉ dựa trên máu và sinh trắc học thông thường là phi thực tế về mặt sinh học.

---

## 4. SỰ ĐÁNH ĐỔI BẤT KHẢ KHÁNG GIỮA CÁC MỤC TIÊU LÂM SÀNG

Không tồn tại một mô hình "toàn năng" thỏa mãn đồng thời tất cả các tiêu chí y tế:
1. **Tối ưu điểm số thống kê thuần túy:** Mô hình $E_2$ (Hard RF--RF Tuned) đạt Macro-F1 cao nhất ($0.4586$) và Weighted-F1 cao nhất ($0.7512$). Tuy nhiên, mô hình này vẫn bỏ sót **98 ca trầm cảm nặng** ($C_2^+$) bị phân loại nhầm thành không có triệu chứng ($C_0$).
2. **Ưu tiên bảo vệ an toàn bệnh nhân:** Mô hình $E_{6b}$ (Soft Bayes Risk với ma trận tổn thất $M_1$) giảm số ca nặng bị bỏ sót xuống chỉ còn **62 ca** (giảm $48.33\%$ so với $E_1$ và $36.73\%$ so với $E_2$). Tuy nhiên, cái giá phải trả là Macro-F1 giảm xuống $0.4215$ do mô hình chủ động chấp nhận nhiều cảnh báo giả hơn ở nhóm triệu chứng nhẹ để không bỏ lọt ca nguy hiểm.
3. **Tuân thủ bản chất thứ bậc và chống trôi dạt:** Mô hình $E_{7a}$ (Balanced Ordinal Logistic Regression) đạt Balanced Accuracy cao nhất ($0.5151$) và suy giảm hiệu năng thấp nhất khi kiểm định ngoại mẫu theo thời gian ($0.5073$), nhưng Macro-F1 ở mức vừa phải ($0.4312$).

*Kết luận:* Quyết định lựa chọn mô hình triển khai thực tế phải do hội đồng y tế và các nhà quản lý chính sách y tế công cộng đưa ra dựa trên cán cân giữa chi phí thăm khám chuyên sâu và hậu quả của việc bỏ sót bệnh nhân nặng trong cộng đồng.

---

## 5. TÌNH TRẠNG KIỂM ĐỊNH TÀI LIỆU THAM KHẢO NGOẠI VI

- Toàn bộ các công trình nền tảng về thang đo PHQ-9 (*Kroenke et al., 2001*), thuật toán Random Forest (*Breiman, 2001*), XGBoost (*Chen & Guestrin, 2016*), mô hình giải thích SHAP (*Lundberg & Lee, 2017*), lý thuyết rủi ro Bayes (*Duda, Hart & Stork, 2001*), và quy trình thiết kế mẫu NHANES (*CDC/NCHS, 2013--2018*) đã được kiểm chứng trực tiếp, bảo đảm trích dẫn chính xác tác giả, năm xuất bản và nguồn phát hành.
- Báo cáo không trích dẫn các liên kết web tạm thời hoặc các bản thảo chưa qua bình duyệt (unverified preprints) không rõ nguồn gốc.

