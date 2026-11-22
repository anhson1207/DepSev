# BÁO CÁO CÁC VẤN ĐỀ CÒN TỒN TẠI VÀ HẠN CHẾ KHOA HỌC (REMAINING ISSUES & LIMITATIONS)
**Dự án:** DepSev-ML – Khung học máy dự đoán mức độ triệu chứng trầm cảm từ dữ liệu sinh lý  
**Tài liệu xuất bản:** `reports/final_revision_v3/DepSev_ML_Final_Polished.pdf`  
**Ngày lập báo cáo:** 29/10/2026  

Báo cáo này ghi nhận một cách trung thực, khách quan và minh bạch toàn bộ các hạn chế khoa học, rào cản kỹ thuật và định hướng hoàn thiện tiếp theo của công trình nghiên cứu DepSev-ML, đóng vai trò là tài liệu tham khảo quan trọng cho hội đồng chấm khóa luận và các nhà nghiên cứu kế thừa.

---

## 1. GIỚI HẠN BẢN CHẤT CỦA DỮ LIỆU SINH LÝ DẠNG BẢNG (DATA BOTTLENECK)

1. **Bản chất đo lường chủ quan của thang PHQ-9 vs Dấu ấn sinh học:**
   - Thang đo PHQ-9 phản ánh trải nghiệm tâm lý chủ quan của đối tượng trong vòng hai tuần qua (cảm giác buồn bã, mất ngủ, mệt mỏi, chán ăn, mất tập trung).
   - Ngược lại, 34 đặc trưng sinh lý trong $F_4$ (huyết áp, BMI, chỉ số mỡ máu, công thức máu toàn phần) đo lường trạng thái chuyển hóa thực thể và thể chất.
   - **Thực tế thực nghiệm:** Phân tích PCA (Hình 4.4) và kiểm định ma trận nhầm lẫn OOF đã chứng minh một ranh giới sinh lý mờ nhạt: Một người có 4 điểm PHQ-9 (phân loại vào nhóm $C_0$ -- Không trầm cảm) và một người có 5 điểm PHQ-9 (phân loại vào nhóm $C_1$ -- Trầm cảm nhẹ) sở hữu phân phối dấu ấn sinh học gần như đồng nhất. Không có bất kỳ mô hình học máy nào (dù là Flat RF, XGBoost, Hierarchical hay Ordinal) có thể tạo ra tín hiệu phân tách giả tạo khi dữ liệu sinh lý nền tảng không mang thông tin phân định mạnh.

2. **Điểm nghẽn phân loại lớp Trầm cảm nhẹ ($C_1$):**
   - Trên nhiệm vụ 3 lớp $T_3$, lớp $C_1$ chịu sự chèn ép hai đầu từ lớp đa số tuyệt đối $C_0$ (chiếm 75,2%) và nhóm triệu chứng rõ rệt $C_2^+$ (chiếm 10,7%).
   - Các mô hình tối ưu theo Macro-F1 hoặc Accuracy ($E_0, E_1, E_4, E_5$) gần như "bỏ quên" lớp $C_1$ (Recall $C_1$ chỉ đạt từ 0% đến 14,5%). Khi can thiệp bằng tối ưu ngưỡng hoặc ma trận nhạy chi phí ($E_{6b}$), Recall $C_1$ có thể tăng vọt lên 60,3%, nhưng phải trả giá bằng việc tỷ lệ cảnh báo giả tăng gấp nhiều lần.

---

## 2. RÀO CẢN VÀ ĐÁNH ĐỔI LÂM SÀNG TRONG ỨNG DỤNG THỰC TẾ

1. **Ranh giới giữa Hệ thống sàng lọc nguy cơ và Công cụ chẩn đoán:**
   - Khóa luận đã khẳng định rõ ràng: **DepSev-ML là một khung công nghệ hỗ trợ sàng lọc phân tầng nguy cơ (Risk Stratification & Screening), tuyệt đối không phải là công cụ thay thế chẩn đoán lâm sàng của bác sĩ chuyên khoa tâm thần.**
   - Mức Macro-F1 hiện tại trên nhiệm vụ $T_3$ dao động từ 0.35 đến 0.45 và Balanced Accuracy từ 48% đến 52%. Mức độ này phản ánh đúng bản chất tín hiệu yếu của dữ liệu sinh lý đơn lẻ. Các khẳng định về việc "triển khai tự động hóa hoàn toàn chẩn đoán lâm sàng" là không có cơ sở khoa học và đã được lược bỏ triệt để trong phiên bản báo cáo cuối cùng.

2. **Quy luật đánh đổi bất khả kháng giữa Bỏ sót bệnh và Báo động nhầm:**
   - Kết quả từ Bảng 4.7 và Hình 4.8 minh chứng quy luật bảo toàn đánh đổi:
     - Với mô hình tối ưu độ chính xác ($E_1$): Số ca bỏ sót nhóm nặng $C_2^+ \to C_0$ là 156 ca (chiếm 36,0% tổng số ca nặng), nhưng số ca báo động nhầm $C_0 \to C_2^+$ chỉ là 33 ca.
     - Với mô hình ưu tiên cứu sinh nhạy chi phí ($E_{6b}$): Số ca bỏ sót giảm mạnh xuống chỉ còn 31 ca (giảm 80%), nhưng số ca báo động nhầm từ người khỏe mạnh thành nặng tăng vọt lên 1.340 ca.
   - Trong bối cảnh y tế cộng đồng, việc lựa chọn cấu hình mô hình nào hoàn toàn phụ thuộc vào năng lực chịu tải của cơ sở y tế (chi phí can thiệp một ca báo động giả so với tổn thất khi bỏ sót một ca trầm cảm nặng có ý định tự hại).

---

## 3. KHẢ NĂNG TỔNG QUÁT HÓA THEO THỜI GIAN (TEMPORAL GENERALIZATION)

1. **Sự dịch chuyển phân phối giữa các chu kỳ khảo sát (Distribution Drift):**
   - Đánh giá trên nhánh dữ liệu Branch D (NHANES 2015--2016, $N = 4.888$) cho thấy Macro-F1 của các mô hình giữ được độ ổn định tương đối ($0.42 - 0.44$).
   - Tuy nhiên, một số biến sinh lý (như tỷ lệ sử dụng thuốc chống trầm cảm tự khai, các chỉ số xét nghiệm chuyển hóa) có sự biến thiên theo từng chu kỳ điều tra của CDC. Việc triển khai thực tế đòi hỏi cơ chế cập nhật mô hình định kỳ (Continuous Monitoring) để ứng phó với trôi dạt dữ liệu.

2. **Dữ liệu khuyết thiếu và quy trình nội suy:**
   - Một số chỉ số sinh hóa chuyên sâu trong NHANES có tỷ lệ khuyết thiếu tự nhiên từ 5% đến 12%. Việc sử dụng SimpleImputer (median) trong đường ống scikit-learn hiện tại bảo đảm không rò rỉ dữ liệu qua các Fold, nhưng trong tương lai có thể khảo sát thêm các kỹ thuật nội suy đa biến như MICE hoặc K-NN Imputation.

---

## 4. TÌNH TRẠNG KỸ THUẬT VÀ DÀN TRANG ĐÃ GIẢI QUYẾT TRỌN VẸN

| Hạng mục dàn trang | Trạng thái kỹ thuật | Đánh giá |
| :--- | :---: | :--- |
| **Lỗi tràn lề trang in** | Đã giải quyết 100% | Toàn bộ 77 trang có tọa độ $x_1 \le 538.58$ pt, không còn bất kỳ điểm tràn lề nào. |
| **Trôi lệch hình ảnh** | Đã giải quyết 100% | Hình 3.1 và toàn bộ các hình được cố định vị trí hợp lý sau đoạn văn dẫn nhập. |
| **Mất thông tin bảng biểu** | Đã giải quyết 100% | Bảng 4.7 được tái cấu trúc thành 2 phần đứng, hiển thị đầy đủ 100% chỉ số. |
| **Tính dễ đọc của đồ thị** | Đã giải quyết 100% | Toàn bộ đồ thị được tái tạo ở độ phân giải 300 DPI, nhãn tiếng Việt to rõ, ma trận nhầm lẫn tách riêng cực kỳ sắc nét. |
| **Danh mục TOC/LOT/LOF** | Đã giải quyết 100% | Số trang và tiêu đề bảng biểu/hình vẽ hoàn toàn trùng khớp giữa mục lục và nội dung. |

