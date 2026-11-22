# BÁO CÁO TỰ PHÊ BÌNH VÀ CÁC GIỚI HẠN CÒN TỒN TẠI (V7)
*(Honest Self-Reflection on Remaining Limitations and Methodological Constraints)*

Nhằm bảo đảm tính trung thực khoa học cao nhất của một công trình nghiên cứu Khoa học Máy tính, tài liệu này chỉ rõ những giới hạn cố hữu mà đề tài chưa thể khắc phục hoàn toàn trong phạm vi dữ liệu và tài nguyên hiện tại.

---

## 1. GIỚI HẠN VỀ DỮ LIỆU THỰC NGHIỆM

1. **Bản chất dữ liệu cắt ngang (Cross-sectional Study)**:
   - Bộ dữ liệu NHANES được thu thập tại một thời điểm duy nhất cho mỗi đối tượng tham gia. Nghiên cứu do đó không thể theo dõi diễn tiến động học (longitudinal dynamics) của các chỉ số sinh lý trong quá trình triệu chứng xuất hiện, thuyên giảm hoặc tái phát.
2. **Tính gián tiếp của các biến sinh lý ngoại vi**:
   - Dữ liệu đầu vào chỉ bao gồm các xét nghiệm máu, nhân trắc học và chức năng chuyển hóa ngoại vi. Đây là các phản ứng thứ cấp của cơ thể trước tình trạng viêm mạn tính hoặc stress kéo dài, không phản ánh trực tiếp hoạt động sinh học thần kinh trung ương (như tín hiệu điện não đồ EEG, hình ảnh học cộng hưởng từ chức năng fMRI hay nồng độ chất dẫn truyền thần kinh tại synap).
3. **Tính chất chủ quan của thang đo tự điền PHQ-9**:
   - Nhãn tham chiếu của bài toán dựa trên 9 câu hỏi tự đánh giá cảm nhận trong 2 tuần gần nhất của người tham gia. Điểm số này có thể chịu sai số do tâm lý tự vệ, thiên kiến nhớ lại (recall bias) hoặc sự khác biệt về văn hóa khi diễn giải mức độ suy sụp tinh thần.

---

## 2. GIỚI HẠN VỀ MÔ HÌNH HỌC MÁY VÀ THUẬT TOÁN

1. **Hiệu năng nhận diện lớp trung gian $C_1$ (Mức độ nhẹ)**:
   - Trên không gian $T_3$, lớp nhẹ ($C_1$, PHQ-9 từ 5 đến 9) có hiệu năng nhận diện (Recall) thấp nhất ở hầu hết các mô hình (chỉ đạt từ $15\%$ đến $35\%$). Điều này phản ánh sự chồng lấn sinh lý học sâu sắc: một người có triệu chứng mệt mỏi nhẹ hoặc rối loạn giấc ngủ tạm thời có thể có các chỉ số xét nghiệm tương đồng với người bình thường ($C_0$).
2. **Sự đánh đổi tất yếu trong tối ưu chi phí Bayes ($E_{6b}$)**:
   - Dù mô hình phân cấp mềm nhạy chi phí $E_{6b}$ với ma trận $M_1$ giúp giảm $48.33\%$ số ca nặng bị bỏ sót (từ 120 ca xuống 62 ca), cái giá phải trả là số ca bình thường bị báo động nhầm ($C_0 \to \{C_1, C_2^+\}$) tăng từ 538 ca lên 1.150 ca. Khi nâng mức phạt lên $M_2$ (chỉ bỏ sót 19 ca nặng), số báo động nhầm tăng lên 1.906 ca. Đây là sự đánh đổi không thể triệt tiêu nếu không có thêm thông tin đặc trưng mới.
3. **Hiện tượng vi phạm tính đơn điệu trong Cumulative Binary Classification ($E_{7b}$)**:
   - Phương pháp hiệu chỉnh cắt ngưỡng (Isotonic Probability Clipping) tuy bảo đảm xác suất hợp lệ về mặt giải tích ($0 \le P(C_1) \le 1$), nhưng khi ép $\tilde{P}_2 = \hat{P}_1$ tại các điểm vi phạm, nó đã triệt tiêu xác suất gán nhãn cho lớp trung gian, khiến Recall của $C_1$ bị suy giảm gần như về 0.

---

## 3. GIỚI HẠN VỀ TRIỂN KHAI THỰC TẾ

1. **Mô hình chỉ mang tính chất sàng lọc sơ bộ (Screening Tool)**:
   - DepSev-ML được thiết kế như một công cụ hỗ trợ gợi ý sàng lọc trong các đợt khám sức khỏe tổng quát cộng đồng. Kết quả dự đoán tuyệt đối không thể thay thế cho quy trình chẩn đoán tâm thần học chuyên khoa của bác sĩ lâm sàng.

