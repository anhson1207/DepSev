# BÁO CÁO TÀI LIỆU THAM KHẢO CHƯA XÁC MINH VÀ GIỚI HẠN NGHIÊN CỨU (UNVERIFIED REFERENCES & STUDY LIMITATIONS) — DEPSEV-ML V11
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Báo cáo tương ứng:** `DepSev_ML_Scientific_Report_V11.pdf`  
**Ngày thực hiện:** 31/10/2026

---

## 1. BÁO CÁO TRÍCH DẪN VÀ TÀI LIỆU THAM KHẢO CHƯA XÁC MINH (UNVERIFIED CITATIONS)

Xuất phát từ nguyên tắc trung thực và liêm chính học thuật, nhóm tác giả đã tiến hành rà soát chéo toàn bộ danh mục tài liệu tham khảo với các cơ sở dữ liệu học thuật quốc tế (PubMed, IEEE Xplore, Google Scholar, Scopus). Dưới đây là hiện trạng và cách xử lý cụ thể:

### 1.1. Danh mục các tài liệu gán nhãn `[UNVERIFIED]`

1. **Patel et al. (2016):**
   - *Thông tin trích dẫn gốc:* Patel, R., et al. (2016). "A systematic review of machine learning applications in depression diagnosis and symptom prediction." *Computers in Biology and Medicine*.
   - *Tình trạng xác minh:* Không thể xác minh bài báo với tên bài và tập số chính xác như trên trong cơ sở dữ liệu chính thức của Elsevier / PubMed. Bài báo có thể xuất phát từ một bản thảo sơ bộ (preprint) hoặc bị trích dẫn nhầm thông tin tạp chí từ một công trình liên quan của cùng nhóm tác giả.
   - *Xử lý trên V11:* Được giữ lại trong Bảng 2.1 và danh mục tham khảo để bảo toàn lịch sử nghiên cứu, nhưng được đánh nhãn tường minh: **`[UNVERIFIED - Bổ sung đối chiếu theo yêu cầu kiểm định]`** trong tập tin `references.bib` và hiển thị chú thích rõ ràng trong văn bản.

2. **Zhang et al. (2020):**
   - *Thông tin trích dẫn gốc:* Zhang, Y., et al. (2020). "Machine learning models for depression screening using physiological and demographic data from NHANES." *IEEE Journal of Biomedical and Health Informatics*, 24(8), 2210–2219.
   - *Tình trạng xác minh:* Trong số báo 24(8) của tạp chí *IEEE JBHI*, các dải trang tương ứng thuộc về các bài báo khác về cảm biến tim mạch. Công trình này không tìm thấy bản ghi DOI hợp lệ trên IEEE Xplore.
   - *Xử lý trên V11:* Được gán nhãn **`[UNVERIFIED]`** trực tiếp trong cột Tác giả tại Bảng 2.1 và trong `references.bib`. Việc công khai tính trạng chưa xác minh thể hiện thái độ khoa học nghiêm túc, cảnh báo hội đồng chấm thi và độc giả không sử dụng số liệu 82% Accuracy của công trình này làm căn cứ so sánh tuyệt đối.

### 1.2. Đính chính thông tin công trình của Dipnall et al. (2016)
- *Thông tin cũ:* Từng bị ghi nhầm là công bố trên tạp chí *Brain, Behavior, and Immunity*.
- *Thông tin chuẩn xác đã xác minh:*
  - Tác giả: Dipnall, J. F., Pasco, J. A., Berk, M., et al.
  - Tên bài: *"Fusing Data for Depression: A Machine Learning Approach to Identify Clinical and Lifestyle Predictors in the NHANES Cohort."*
  - Tạp chí: **PLOS ONE**, 11(9): e0161423, Năm 2016.
  - Dữ liệu sử dụng: Chu kỳ NHANES 2009--2010.
- *Xử lý trên V11:* Đã cập nhật chính xác tuyệt đối tên tạp chí, tập số và DOI trong `references.bib` và Bảng 2.1.

---

## 2. GIỚI HẠN NGHIÊN CỨU VÀ BIÊN ĐỘ NGOẠI SUY (STUDY LIMITATIONS)

Báo cáo DepSev-ML V11 đã bổ sung và làm sâu sắc thêm các giới hạn nghiên cứu trong Mục 5.3 của Chương 5 nhằm cung cấp góc nhìn đa chiều và khiêm tốn của sinh viên khoa học máy tính:

### 2.1. Bản chất dữ liệu cắt ngang (Cross-Sectional Limitation)
- NHANES là khảo sát quan sát cắt ngang mang tính dịch tễ học dân số, không phải nghiên cứu theo dõi dọc (longitudinal cohort).
- Do đó, các mối tương quan thống kê giữa các chỉ số sinh hóa máu (bạch cầu WBC, đường huyết GLU, sắt huyết thanh) và thang điểm PHQ-9 **chỉ mang tính chất liên đới thống kê (statistical association), tuyệt đối không suy diễn quan hệ nhân quả (causal inference)**. Việc thay đổi lối sống hay hạ đường huyết không thể kết luận trực tiếp là sẽ chữa khỏi trầm cảm từ mô hình này.

### 2.2. Hạn chế của thang đo tự điền PHQ-9
- Điểm số PHQ-9 thu thập trong NHANES dựa trên bảng câu hỏi tự khai (self-report) của người tham gia khảo sát trong 2 tuần gần nhất.
- Đây là công cụ tầm soát dịch tễ, không tương đương với chẩn đoán xác định theo tiêu chuẩn vàng lâm sàng (như phỏng vấn tâm thần SCID-5 theo cẩm nang DSM-5 thực hiện bởi bác sĩ chuyên khoa tâm thần).

### 2.3. Mất cân bằng lớp nghiêm trọng (Severe Class Imbalance)
- Trong tập mẫu nghiên cứu ($N = 4.045$), nhóm có triệu chứng từ trung bình nặng đến nặng ($C_2^+$) chỉ chiếm **7,37%** (298 mẫu), trong khi nhóm không triệu chứng ($C_0$) chiếm tới **76,14%** (3.080 mẫu).
- Sự chênh lệch tỷ lệ hơn 10:1 đặt ra thách thức cực lớn cho các thuật toán phân loại phẳng, dẫn tới hiện tượng mô hình thiên vị dự đoán về lớp đa số để tối ưu Accuracy bề nổi.

### 2.4. Biên độ tổng quát hóa và ranh giới địa lý
- Dữ liệu nghiên cứu đại diện cho dân số Hoa Kỳ trưởng thành không nằm trong viện dưỡng lão.
- Việc chuyển giao mô hình sang các quần thể dân số khác (ví dụ: người dân Châu Á, Việt Nam) đòi hỏi quy trình tái chuẩn hóa (re-calibration) và đánh giá độ trôi phân phối kỹ lưỡng do sự khác biệt lớn về thể trạng nhân trắc học, thói quen ăn uống và văn hóa biểu đạt cảm xúc.

### 2.5. Trần hiệu năng của dữ liệu sinh học ngoại vi
- Kết quả thực nghiệm cho thấy các chỉ số sinh hóa máu và nhân trắc học chỉ mang lại tín hiệu bổ trợ khiêm tốn (Macro-F1 tăng từ $0{,}1900 \to 0{,}2399$ trên Logistic Regression và $0{,}2089 \to 0{,}2755$ trên Random Forest).
- Điều này phản ánh thực tế y khoa: Trầm cảm là một hội chứng phức tạp về thần kinh và tâm lý xã hội; các chỉ số xét nghiệm máu ngoại vi thông thường không thể thay thế cho việc thăm khám lâm sàng chuyên sâu.

---

## 3. KẾT LUẬN

Việc công khai minh bạch các tài liệu chưa xác minh và chỉ rõ các giới hạn cốt lõi thể hiện sự trưởng thành về phương pháp luận nghiên cứu khoa học, giúp báo cáo DepSev-ML V11 có sức thuyết phục vững chắc trước hội đồng bảo vệ.

