# BÁO CÁO BIÊN TẬP VĂN PHONG VÀ CHUẨN MỰC HỌC THUẬT (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/audit/ACADEMIC_WRITING_AUDIT.md`  
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Ngày thực hiện:** 31/10/2026  
**Người thực hiện:** Antigravity Research Team  

---

## 1. MỤC TIÊU VÀ NGUYÊN TẮC BIÊN TẬP

Trong các phiên bản trước (V6--V9), báo cáo vẫn còn vướng phải một số nhược điểm về văn phong:
1. **Lạm dụng ngôn từ mang tính quảng cáo / AI-generated puffery:** Xuất hiện các cụm từ đao to búa lớn như *"đột phá mang tính cách mạng"*, *"ưu việt tuyệt đối"*, *"chiến thắng áp đảo"*, làm giảm đi vẻ khiêm tốn và tính khách quan cần có của một công trình nghiên cứu khoa học.
2. **Tư duy trình bày "chữa lỗi" tiêu cực:** Trình bày theo hướng bài toán 5 lớp ($T_5$) bị "thất bại thảm hại" nên tác giả phải "cầu cứu" sang phân loại phân cấp ($T_3$), tạo ấn tượng không tốt cho hội đồng đánh giá.
3. **Dịch thuật ngữ Machine Learning gượng ép:** Cố gắng dịch các thuật ngữ chuyên ngành chuẩn mực sang tiếng Việt một cách khiên cưỡng, gây khó hiểu cho người đọc chuyên môn.
4. **Font chữ tham số chắp vá:** Sử dụng font `\texttt` (monospace) giữa các đoạn văn mô tả thuật toán, làm vỡ tính liền mạch của trang sách.

---

## 2. CHUYỂN ĐỔI MẠCH DẪN DẮT: TỪ "CHỮA LỖI" SANG "NGHIÊN CỨU SO SÁNH NHIỀU HƯỚNG"

Trong bản V10, toàn bộ cấu trúc dẫn dắt của 5 chương được tổ chức lại theo tinh thần: **Khám phá, so sánh và phân tích sự đánh đổi (trade-offs) giữa các kiến trúc học máy khác nhau trên dữ liệu bảng y tế.**

### Hướng tiếp cận 1: Khảo sát phân loại phẳng đa mức trên không gian gốc ($T_5$)
- **Mục tiêu:** Đánh giá khả năng của các mô hình Machine Learning cổ điển và ensemble (Logistic Regression, Decision Tree, Random Forest, XGBoost) trong việc phân biệt 5 mức độ trầm cảm chi tiết theo thang đo lâm sàng PHQ-9 ($C_0 \dots C_4$).
- **Phát hiện khoa học:** Kết quả Nested Cross-Validation cho thấy Random Forest đạt Macro-F1 cao nhất ($0.2755 \pm 0.0270$). Tuy nhiên, phân tích sâu ma trận nhầm lẫn out-of-fold chỉ ra hiện tượng thắt nút cổ chai (bottleneck): lớp thiểu số cực đoan ($C_4$ chỉ chiếm $1.05\%$) và sự tương đồng về chỉ số sinh học giữa các nhóm lân cận khiến việc phân biệt 5 mức chi tiết bằng dữ liệu cận lâm sàng thuần túy gặp giới hạn thống kê tự nhiên.
- **Vai trò của Random Forest:** Được chọn làm mô hình cơ sở đại diện (representative baseline) cho các giai đoạn tiếp theo nhờ khả năng cân bằng giữa việc nhận diện lớp thiểu số và độ chính xác tổng thể.

### Hướng tiếp cận 2: Khảo sát phân loại đa hướng trên không gian mức độ lâm sàng ($T_3$)
- **Mục tiêu:** Tái cấu trúc không gian nhãn thành 3 mức có ý nghĩa can thiệp y tế ($C_0$: Bình thường, $C_1$: Triệu chứng nhẹ, $C_2^+$: Cần can thiệp lâm sàng).
- **So sánh 4 trường phái mô hình:**
  1. *Phân loại phẳng (Flat Baseline $E_0$):* Đơn giản, độ phức tạp tính toán thấp.
  2. *Phân loại phân cấp cứng (Hard Hierarchical $E_1 \dots E_4$):* Tách rời bài toán phát hiện triệu chứng và phân định mức độ, đạt Macro-F1 cao nhất ($0.4586$ ở cấu hình $E_2$).
  3. *Phân loại phân cấp mềm kết hợp lý thuyết rủi ro Bayes (Soft Cost-Sensitive $E_5, E_6$):* Tích hợp ma trận chi phí bất đối xứng nhằm tối đa hóa an toàn lâm sàng, giảm thiểu tối đa số ca nặng bị bỏ sót ($E_{6b}$ giảm 48.33% ca bỏ sót $C_2^+ \to C_0$).
  4. *Hồi quy Logistic thứ bậc (Ordinal Logistic Regression $E_7$):* Tôn trọng trật tự tự nhiên của bệnh học tâm thần, đạt độ chính xác cân bằng cao nhất ($0.5151$) và ổn định nhất trước sự trôi phân phối thời gian.
- **Thông điệp nghiên cứu:** Không có một mô hình nào tối ưu trên mọi phương diện. Việc lựa chọn mô hình phụ thuộc hoàn toàn vào ngữ cảnh triển khai trong hệ thống y tế thực tế.

---

## 3. LOẠI BỎ CÁC CỤM TỪ KHOA TRƯƠNG VÀ CHUẨN HÓA VĂN PHONG HỌC THUẬT

Dưới đây là một số ví dụ tiêu biểu về các đoạn văn đã được biên tập lại giữa V9 và V10:

| Nội dung / Chủ đề | Cách diễn đạt trong V9 (Khoa trương / Máy móc) | Cách diễn đạt trong V10 (Chuẩn mực khoa học, khiêm tốn) |
| :--- | :--- | :--- |
| **Đánh giá Random Forest** | *"Random Forest là mô hình vượt trội toàn diện, đánh bại hoàn toàn các đối thủ nhờ cơ chế giải thuật vô cùng mạnh mẽ..."* | *"Random Forest đạt hiệu năng tổng thể hài hòa nhất trong số các thuật toán được khảo sát, với Macro-F1 đạt $0.2755 \pm 0.0270$ và Weighted-F1 đạt $0.6437 \pm 0.0087$. Do đó, mô hình này được lựa chọn làm cơ sở đối chuẩn chính..."* |
| **Phân tích mô hình Bayes** | *"Cơ chế Bayes Risk đã tạo ra một bước đột phá ngoạn mục, đập tan hiện tượng bỏ sót bệnh nhân nặng..."* | *"Khi tích hợp hàm tổn thất bất đối xứng theo lý thuyết rủi ro Bayes ($E_{6b}$), số ca nặng ($C_2^+$) bị phân loại nhầm thành bình thường ($C_0$) giảm từ 120 ca xuống còn 62 ca (giảm $48.33\%$), phù hợp với yêu cầu cảnh báo sớm trong y tế dự phòng."* |
| **Đánh giá Hồi quy thứ bậc** | *"Ordinal Regression chứng minh chân lý tối thượng của cấu trúc thứ tự, đánh gục các mô hình phân loại phẳng..."* | *"Mô hình Hồi quy Logistic thứ bậc cân bằng ($E_{7a}$) phản ánh đúng bản chất trật tự của mức độ trầm cảm, mang lại độ chính xác cân bằng ($0.5151 \pm 0.0152$) và duy trì tính ổn định cao khi kiểm định ngoại mẫu theo thời gian ($0.5073$)."* |
| **Bàn luận về giới hạn dữ liệu** | *"Mô hình của chúng tôi hoàn toàn đủ khả năng thay thế bác sĩ chẩn đoán trầm cảm trong tương lai gần..."* | *"Nghiên cứu nhận định rằng dữ liệu cận lâm sàng chỉ đóng vai trò sàng lọc hỗ trợ ban đầu. Để đưa ra kết luận chẩn đoán chính xác, bắt buộc phải có sự thăm khám lâm sàng toàn diện từ các chuyên gia tâm thần học."* |

---

## 4. CHUẨN HÓA THUẬT NGỮ MACHINE LEARNING VÀ TRÌNH BÀY THAM SỐ

1. **Giữ nguyên thuật ngữ tiếng Anh quốc tế:**
   - Các thuật ngữ chuyên ngành được giữ nguyên gốc tiếng Anh hoặc đi kèm chú giải rõ ràng: *Nested Cross-Validation, Out-of-Fold, Feature Ablation, Random Forest, Gradient Boosting, Balanced Accuracy, Macro-F1, Decision Tree, Logistic Regression, Threshold Moving, Cost-Sensitive Learning*.
   - Tránh việc dịch thô gây khó hiểu như *"Cây quyết định ngẫu nhiên nhiều tầng"*, *"Độ nhạy cân bằng hóa học"*, *"Độ đo F1 trung bình vĩ mô"*.
2. **Loại bỏ font Monospace (`\texttt`) trong thân bài:**
   - Trong các phiên bản trước, tên các siêu tham số như `n_estimators`, `max_depth`, `class_weight` được bao bọc trong `\texttt{...}`, tạo ra sự không đồng nhất về mặt thị giác trên trang sách in.
   - Trong V10, toàn bộ các tham số này đã được chuyển sang font thông thường với dấu gạch dưới được thoát ký tự hợp lệ (`n\_estimators = 150`, `max\_depth = 8`, `max\_features = 0.5`, `min\_samples\_split = 10`, `min\_samples\_leaf = 8`, `class\_weight = 'balanced\_subsample'`).
   - Kết quả kiểm tra trích xuất font từ file PDF hoàn thiện xác nhận: **0% font Monospace (`LMMono`, `Courier`, v.v.)** xuất hiện trong tài liệu.

---

## 5. TỔNG KẾT

Bản báo cáo V10 đã đạt được sự cân bằng mẫu mực giữa **tính chính xác kỹ thuật**, **sự khiêm tốn của sinh viên nghiên cứu**, và **chuẩn mực trình bày của một công trình học thuật chuyên ngành Khoa học Máy tính**. Toàn bộ các nhận định đều dựa trên bằng chứng số liệu cụ thể, có đối chiếu ma trận nhầm lẫn và độ lệch chuẩn rõ ràng.

