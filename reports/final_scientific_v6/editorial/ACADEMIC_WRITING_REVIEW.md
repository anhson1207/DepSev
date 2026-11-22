# ĐÁNH GIÁ PHONG CÁCH VIẾT HỌC THUẬT VÀ CHUẨN MỰC KHOA HỌC (ACADEMIC WRITING REVIEW)
**Dự án:** DepSev-ML  
**Tài liệu rà soát:** Báo cáo Khoa học V6 (`DepSev_ML_Scientific_Report_V6.pdf`)  
**Tiêu chí đánh giá:** Chuẩn mực học thuật quốc tế, tính khiêm tốn khoa học, tính chính xác y sinh, loại bỏ biệt ngữ thương mại/khoa trương  
**Ngày thực hiện:** 30/10/2026  

---

## 1. Nguyên tắc biên tập học thuật trong V6
Một báo cáo nghiên cứu khoa học Machine Learning chất lượng cao không chỉ đòi hỏi sự đúng đắn về số liệu mà còn yêu cầu cách diễn đạt khách quan, khiêm tốn, chặt chẽ và tôn trọng các giới hạn thực tế của dữ liệu.

Trong phiên bản V6, toàn bộ văn bản đã được rà soát và chuẩn hóa theo 5 nguyên tắc biên tập cốt lõi:
1. **Loại bỏ ngôn từ khoa trương, giật gân (Anti-Hyperbole):** Tuyệt đối không dùng các từ ngữ cảm tính như *"bước đột phá vĩ đại"*, *"giảm ngoạn mục"*, *"vạch trần điểm nghẽn"*, *"nhược điểm chí mạng"*, *"triệt tiêu hoàn toàn rủi ro"*, *"mô hình an toàn nhất thế giới"*. Thay thế bằng các mệnh đề miêu tả khách quan: *"mức độ cải thiện có ý nghĩa thống kê"*, *"giảm tỷ lệ bỏ sót"*, *"chỉ ra rủi ro lâm sàng"*, *"hạn chế phương sai"*.
2. **Khiêm tốn về vai trò của Random Forest:** Không khẳng định Random Forest là "thuật toán tốt nhất trên mọi phương diện". Trình bày rõ ràng: RF đạt Macro-F1 trung bình cao nhất trên tập kiểm tra ngoài mẫu của Nested CV ($0.2755 \pm 0.0270$), nhưng Hồi quy Logistic vẫn có lợi thế về độ trơn siêu phẳng quyết định (Balanced Accuracy $0.3253$), và việc lựa chọn RF làm xương sống là dựa trên sự cân bằng giữa khả năng nắm bắt phi tuyến và tính ổn định phương sai.
3. **Chính xác hóa ngôn ngữ y sinh học và bối cảnh sàng lọc:**
   - Khẳng định rõ thang đo PHQ-9 là **công cụ sàng lọc triệu chứng tâm lý** (psychological symptom screening tool), không thay thế cuộc phỏng vấn chẩn đoán lâm sàng có cấu trúc của bác sĩ chuyên khoa tâm thần.
   - Điểm số $0$--$4$ là quy ước mức độ "triệu chứng tối thiểu" (Minimal symptoms) trong sàng lọc cộng đồng, không khẳng định đối tượng là "hoàn toàn khỏe mạnh" hay "bình thường tuyệt đối".
   - Ngưỡng $C_2^+$ (điểm PHQ-9 $\ge 10$) là quy ước nghiên cứu và thực hành sàng lọc y tế cộng đồng để kích hoạt đánh giá chuyên khoa, không phải là ngưỡng chẩn đoán bệnh lý tự động.
4. **Tránh kết luận quá đà về mặt toán học từ biểu đồ PCA:**
   - Biểu đồ PCA 2 chiều là phép chiếu trực giao tuyến tính (linear orthogonal projection).
   - Sự chồng lấn phân phối của các lớp trên không gian PCA 2D phản ánh sự tương đồng của các biến số sinh lý trong không gian tuyến tính chiều thấp, minh họa trực quan cho độ khó của bài toán, nhưng không phải là "bằng chứng toán học tuyệt đối chứng minh không thể phân tách bằng bất kỳ hàm phi tuyến nào".
5. **Đại từ nhân xưng chuẩn mực:**
   - Toàn bộ lời cam đoan và lời cảm ơn xưng "em" khiêm tốn, lịch sự, tôn trọng hội đồng và giảng viên hướng dẫn (TS. Vũ Thị Hồng Nhạn).

---

## 2. Bảng đối chiếu các chỉnh sửa câu chữ tiêu biểu

| Vị trí | Câu văn cũ trong V5 (Thiếu chuẩn mực / Khoa trương) | Câu văn mới trong V6 (Chuẩn mực học thuật / Khiêm tốn) |
| :--- | :--- | :--- |
| **Chương 1 (Mục 1.5)** | *"vạch trần điểm nghẽn nghiêm trọng khi loại bỏ nhầm 120 ca nặng... giảm ngoạn mục 74.2%"* | *"chỉ ra điểm nghẽn lan truyền sai số có nguy cơ lâm sàng... giảm số ca bỏ sót ca nặng sang bình thường từ 120 ca xuống còn 62 ca (giảm 48.33% với ma trận chi phí $M_1$)"* |
| **Chương 2 (Mục 2.3.1)** | *"Random Forest là thuật toán vượt trội nhất, đánh bại hoàn toàn các mô hình khác"* | *"Rừng ngẫu nhiên đạt chỉ số Macro-F1 trung bình cao nhất trong nhóm các mô hình phẳng được khảo sát (0.2755 ± 0.0270)... mang lại sự cân bằng tốt nhất giữa khả năng nắm bắt phi tuyến và độ bền vững phương sai"* |
| **Chương 3 (Mục 3.4.2)** | *"Đây là nhược điểm chí mạng của định tuyến cứng trong ứng dụng lâm sàng"* | *"Đây là rủi ro lâm sàng nghiêm trọng của cơ chế định tuyến cứng"* |
| **Chương 3 (Mục 3.4.3)** | *"Để triệt tiêu hiện tượng nghẽn sai số một chiều"* | *"Để khắc phục hiện tượng nghẽn sai số một chiều"* |
| **Chương 4 (Mục 4.4)** | *"sự sụp đổ hoàn toàn của lớp triệu chứng nhẹ khi Recall chỉ đạt 16.29%"* | *"sự suy giảm độ nhạy của các lớp triệu chứng: trên $T_5$ lớp nhẹ $C_1$ có Recall chỉ đạt 15.83% (53.92% bị nhầm sang $C_0$) và lớp rất nặng $C_4$ chỉ phát hiện đúng 1/44 ca (Recall = 2.27%)"* |
| **Chương 4 (Mục 4.7.2)** | *"cắt giảm ngoạn mục 74.2% số ca bỏ sót... mô hình an toàn nhất"* | *"cắt giảm 48.33% số ca trầm cảm nặng bị bỏ sót (từ 120 ca xuống còn 62 ca với ma trận $M_1$), giúp giảm tỷ lệ bỏ sót ca nặng từ 30.69% xuống 15.86%"* |
| **Chương 4 (Mục 4.11)** | *"vạch trần điểm nghẽn của phân cấp cứng"* | *"làm rõ rủi ro bỏ sót ca nặng của phân cấp cứng"* |
| **Chương 5 (Mục 5.1)** | *"Mô hình phân cấp mềm tối ưu chi phí Bayes giải quyết triệt để điểm nghẽn... giảm 74.2%... chỉ còn 31 ca"* | *"Mô hình phân cấp mềm kết hợp tối ưu chi phí Bayes giải quyết hiệu quả rủi ro bỏ sót lâm sàng: với ma trận chi phí $M_1$, mô hình giảm 48.33% số ca bỏ sót nặng thành bình thường (từ 120 ca xuống còn 62 ca)"* |

---

## 3. Đánh giá tổng kết
Báo cáo V6 đạt được sự chín chắn cao về văn phong khoa học: khiêm tốn nhưng lập luận đanh thép, tôn trọng sự thật khách quan của dữ liệu, minh bạch các hạn chế thực tế và tuân thủ tuyệt đối đạo đức nghiên cứu học thuật.

