# TỔNG KẾT VẤN ĐỀ TỒN ĐỌNG VÀ TỰ PHẢN BIỆN KHOA HỌC (UNRESOLVED ISSUES & SCIENTIFIC REFLECTIONS) — DEPSEV-ML V8

**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản:** Báo cáo khoa học V8 (`DepSev_ML_Scientific_Report_V8.pdf`)  
**Ngày thực hiện:** 30/10/2026  

---

## 1. Giới thiệu

Với tinh thần liêm chính học thuật và tính trung thực khoa học của một sinh viên ngành Khoa học Máy tính, tài liệu này ghi nhận một cách khách quan những giới hạn nghiên cứu nội tại mà phiên bản V8 (cũng như dữ liệu hiện có) chưa thể giải quyết triệt để. 

Đề tài không tìm cách che giấu các kết quả khiêm tốn hoặc thổi phồng khả năng của mô hình học máy, mà nhìn nhận thẳng thắn các thách thức thực tế của bài toán dự đoán sức khỏe tâm thần từ dữ liệu sinh lý ngoại vi.

---

## 2. Các giới hạn khoa học nội tại của đề tài

### 2.1. Giới hạn bản chất của nguồn dữ liệu NHANES
1. **Thiết kế nghiên cứu cắt ngang (Cross-sectional Study):**
   - Bộ dữ liệu NHANES khảo sát các chỉ số sinh lý và bảng câu hỏi PHQ-9 tại một thời điểm duy nhất cho mỗi cá nhân.
   - Do đó, các mô hình học máy chỉ có thể học mối liên hệ tương quan tĩnh (static association) giữa các biến số, hoàn toàn không có khả năng suy diễn mối quan hệ nhân quả (causality) hoặc theo dõi sự biến thiên động (dynamic trajectory) của triệu chứng trầm cảm theo thời gian.
2. **Nhãn mục tiêu dựa trên khảo sát tự báo cáo (Self-reported Target):**
   - Điểm số PHQ-9 được tính từ 9 câu hỏi tự điền, vốn chịu ảnh hưởng mạnh mẽ từ nhận thức chủ quan, định kiến xã hội (stigma) và trạng thái tâm lý nhất thời của người trả lời tại thời điểm phỏng vấn.
   - Nhãn PHQ-9 không phải là chẩn đoán xác định bệnh học lâm sàng (gold standard clinical diagnosis) bởi bác sĩ chuyên khoa tâm thần.

### 2.2. Giới hạn tín hiệu của các chỉ số sinh lý ngoại vi
1. **Độ đặc hiệu sinh học thấp:**
   - Các biến số trong NHANES bao gồm huyết áp, chỉ số khối cơ thể (BMI), cholesterol, albumin và HbA1c là các chỉ dấu sinh học ngoại vi phản ánh tình trạng chuyển hóa và thể chất chung, không phải là các dấu ấn sinh học đặc hiệu thần kinh (neuro-specific biomarkers).
   - Phân tích tương quan thực nghiệm đã chỉ ra không có biến số đơn lẻ nào có hệ số tương quan vượt quá $|r| > 0.25$ với điểm PHQ-9.
2. **Hiện tượng chồng lấn phân phối sâu sắc (Severe Distribution Overlap):**
   - Biểu đồ hộp và hình chiếu PCA 2D đã chứng minh phân phối của nhóm triệu chứng nhẹ ($C_1$) lọt hoàn toàn vào bên trong không gian mật độ của nhóm triệu chứng tối thiểu ($C_0$).
   - Vì lý do này, việc gia tăng kích thước mẫu dữ liệu từ $5.054$ lên $14.581$ đối tượng qua 3 chu kỳ NHANES không giúp cải thiện Macro-F1 trên bài toán 5 lớp phẳng $T_5$ (chỉ dao động quanh mức $\approx 0.26$--$0.27$). Giới hạn này xuất phát từ bản chất thông tin của tập đặc trưng, không phải do thiếu hụt dữ liệu huấn luyện.

### 2.3. Sự đánh đổi không thể tránh khỏi trong các kiến trúc mô hình hóa trên $T_3$
1. **Phân cấp định tuyến cứng ($E_1, E_2$):**
   - Đạt Macro-F1 cao nhất ($0.4586$ ở $E_2$), nhưng tồn tại điểm nghẽn lan truyền sai số một chiều: có tới **98 ca nặng ($C_2^+$) bị Stage 1 gán nhầm thành bình thường ($C_0$)** và bị loại bỏ vĩnh viễn, không bao giờ được Stage 2 tiếp cận.
2. **Phân cấp mềm tối ưu chi phí Bayes ($E_{6a}, E_{6b}$):**
   - Giúp giảm đáng kể số ca nặng bị bỏ sót (từ 120 ca ở $E_1$ xuống còn 62 ca với ma trận $M_1$, và 19 ca với $M_2$).
   - Tuy nhiên, sự bảo vệ này phải đánh đổi bằng việc số lượng ca bình thường bị báo động nhầm ($C_0 \to \{C_1, C_2^+\}$) tăng lên mạnh mẽ (từ 538 ca lên 1.150 ca với $M_1$ và 1.906 ca với $M_2$). Trong ứng dụng thực tế, sự đánh đổi này phụ thuộc hoàn toàn vào nguồn lực tiếp nhận và năng lực xét nghiệm khẳng định của hệ thống y tế cơ sở.
3. **Phân loại nhị phân tích lũy ($E_{7b}$):**
   - Do hai mô hình $P(Y \ge 1)$ và $P(Y \ge 2)$ huấn luyện độc lập, hiện tượng vi phạm đơn điệu xác suất xuất hiện trên $5.32\%$ mẫu (với RF) và $19.53\%$ mẫu (với LR).
   - Sau khi áp dụng giải thuật Isotonic Probability Clipping, việc ép xác suất khiến toàn bộ các mẫu vi phạm bị gán xác suất lớp trung gian $P(C_1) = 0$, dẫn đến Recall của lớp $C_1$ bị suy thoái nghiêm trọng ($0.16\%$ ở RF và $0.00\%$ ở LR).

---

## 3. Định hướng nghiên cứu mở rộng trong tương lai

1. **Thu thập dữ liệu chuỗi thời gian liên tục từ thiết bị đeo (Wearables):**
   - Thay thế các chỉ số đo cắt ngang một lần bằng tín hiệu quang thể tích (PPG), độ biến thiên nhịp tim (HRV) 24/7 và gia tốc kế đo lường chất lượng giấc ngủ để phát hiện các biến đổi vi mô theo chu kỳ sinh học.
2. **Tích hợp dữ liệu đa phương thức (Multimodal Learning):**
   - Kết hợp các đặc trưng sinh lý dạng bảng với dữ liệu âm thanh giọng nói (acoustic features), biến đổi ngữ điệu hoặc biểu cảm nét mặt trong quá trình trả lời phỏng vấn để bổ sung thông tin cảm xúc trực tiếp.
3. **Phát triển mạng nơ-ron học thứ bậc sâu (Deep Ordinal Architectures):**
   - Xây dựng mô hình End-to-End ràng buộc cấu trúc đơn điệu tự nhiên ngay trong hàm mất mát, kết hợp với các kỹ thuật giải thích mô hình (XAI / SHAP) nhằm tăng độ tin cậy khi triển khai hỗ trợ quyết định y tế.

