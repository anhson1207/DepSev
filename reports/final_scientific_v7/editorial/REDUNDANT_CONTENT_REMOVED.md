# DANH SÁCH NỘI DUNG VÀ CÔNG THỨC ĐÃ CẮT BỎ TRONG BÁO CÁO DEPSEV-ML V7
*(Redundant Content and Formula Pruning Report)*

Trong phiên bản V7, nhằm loại bỏ hoàn toàn cảm giác lý thuyết hàn lâm giả tạo, sáo rỗng hoặc dàn trải như đồ án môn học, các nội dung sau đã được rà soát và cắt bỏ triệt để.

---

## 1. CÁC CÔNG THỨC TOÁN HỌC ĐÃ CẮT BỎ

1. **Công thức giải tích tỷ lệ mẫu Out-of-Bag (OOB $\approx 36.8\%$)**:
   - *Công thức cũ*:
     $$P(\text{không được chọn}) = \left(1 - \frac{1}{N}\right)^N \xrightarrow{N \to \infty} \frac{1}{e} \approx 0.3679$$
   - *Lý do cắt bỏ*: Đây là công thức toán cơ bản của giải tích đại học, không liên quan trực tiếp đến đóng góp khoa học của DepSev-ML. Thay vào đó, báo cáo chỉ tóm lược nguyên lý: *"ước tính có khoảng 36.8% số mẫu không xuất hiện trong tập con huấn luyện của cây (gọi là mẫu Out-of-Bag), đóng vai trò như một tập kiểm tra độc lập tự nhiên"*.

2. **Công thức toán học suy diễn phương sai Ensemble tổng quát**:
   - *Công thức cũ*:
     $$\mathrm{Var}\left(\frac{1}{B}\sum_{b=1}^B T_b(\mathbf{x})\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
   - *Lý do cắt bỏ*: Việc biến đổi giới hạn phương sai khi $B \to \infty$ mang tính kinh điển trong sách giáo khoa, không làm tăng thêm giá trị thực nghiệm của đề tài. V7 giải thích bằng lời trực quan: việc lấy mẫu ngẫu nhiên không gian đặc trưng làm giảm độ tương quan giữa các cây ($\rho$), qua đó làm giảm phương sai dự đoán của toàn bộ rừng mà không làm tăng độ chệch.

3. **Các công thức toán học chi tiết của các mô hình cơ sở không trọng tâm**:
   - Công thức hàm mất mát của XGBoost (Taylor cấp 2), công thức cây quyết định CART tổng quát đa tầng, công thức hồi quy Logistic nhiều lớp tổng quát đều được giản lược. Thay vào đó, tập trung trình bày mạch lạc và sâu sắc về cấu trúc Random Forest và ma trận rủi ro tối thiểu Bayes.

---

## 2. CÁC ĐOẠN VĂN, MỤC VÀ HÌNH VẼ ĐÃ CẮT BỎ HOẶC ĐIỀU CHỈNH

1. **Cắt bỏ các đoạn tóm tắt mở đầu/kết thúc chương mang tính lặp lại (Chapter Recaps & Roadmaps)**:
   - Các đoạn dài kiểu *"Trong chương này, chúng tôi sẽ lần lượt trình bày phần A rồi đến phần B rồi đến phần C..."* và các đoạn tóm tắt chương lê thê ở cuối mỗi chương đã được rút gọn thành 1--2 câu xúc tích, chuyển tiếp tự nhiên giữa các chương.

2. **Cắt bỏ các lời tự khen và tuyên bố thương mại/phóng đại**:
   - Cắt bỏ các cụm từ: *"đột phá toàn diện"*, *"hiệu năng xuất sắc vượt trội"*, *"bảo vệ tính mạng người bệnh tuyệt đối"*, *"khung làm việc tối tân nhất"*, *"bằng chứng không thể chối cãi"*.
   - Thay thế bằng các nhận xét khách quan, khoa học, thừa nhận giới hạn thực tế của dữ liệu bảng y tế.

3. **Cắt bỏ hình ảnh trùng lặp (Duplicate Figure 3.6)**:
   - Trong V6, file `figures/two_stage_data_flow.png` bị khai báo 2 lần liên tiếp dưới dạng 2 môi trường `figure` riêng biệt tại Mục 3.3. Trong V7, một hình trùng lặp đã bị xóa bỏ hoàn toàn.

4. **Sửa đổi hình ảnh hiển thị sai nội dung (Hình 4.8)**:
   - Trong V6, Hình 4.8 tại Mục 4.5 ("Phân tích sự suy giảm hiệu năng theo độ chi tiết nhãn") lại hiển thị file `multicycle_soft_comparison.png` (vốn so sánh các mô hình đa chu kỳ theo thời gian).
   - Trong V7, hình này được thay thế bằng `figures/multicycle_cohort_comparison.png` (thể hiện sự suy giảm hiệu năng từ $T_2$ đến $T_5$ qua các nhánh A, B, D). File `multicycle_soft_comparison.png` được chuyển về đúng vị trí tại Mục 4.8 (Kiểm định ngoại suy theo dòng thời gian).

