# BẢNG ĐỐI CHIẾU VĂN PHONG TRƯỚC VÀ SAU BIÊN TẬP (V6 VS V7)
*(Before & After Editorial Comparison Examples)*

Dưới đây là 16 ví dụ đối chiếu trực tiếp giữa các đoạn văn trong phiên bản V6 (văn phong máy móc, dịch thô, khoa trương) và phiên bản V7 (văn phong khoa học, chuẩn mực Computer Science).

---

### Ví dụ 1: Mục tiêu nghiên cứu và giải thích chuyển đổi $T_5 \to T_3$
- **Vị trí**: `chapters/chapter1.tex` (Mục 1.2, Mục tiêu nghiên cứu)
- **Trước (V6)**:  
  *"Đề tài nhằm mục đích chứng minh tính ưu việt tuyệt đối của kiến trúc phân cấp hai giai đoạn, cứu vãn sự sụp đổ thảm hại của các mô hình phân loại phẳng 5 mức triệu chứng cổ điển, mở ra kỷ nguyên mới cho trí tuệ nhân tạo y tế."*
- **Sau (V7)**:  
  *"Nghiên cứu được thiết kế qua hai giai đoạn rõ ràng: Giai đoạn 1 khảo sát bài toán phân loại phẳng 5 mức triệu chứng ($T_5$) nhằm thiết lập mốc đối chuẩn và nhận diện các giới hạn phân tách của dữ liệu sinh lý ngoại vi; Giai đoạn 2 mở rộng sang bài toán phân loại phân cấp hai bước ($T_3$) để phân tích cơ chế lan truyền sai số và thử nghiệm các chiến lược ra quyết định nhạy chi phí."*

---

### Ví dụ 2: Giải thích thuật toán Random Forest
- **Vị trí**: `chapters/chapter2.tex` (Mục 2.3, Giới thiệu Random Forest)
- **Trước (V6)**:  
  *"Thuật toán Rừng Ngẫu Nhiên là một vũ khí học máy tối tân kết hợp hàng trăm Cây Quyết Định tinh nhuệ. Nhờ phép lấy mẫu khởi động tự thân (Bootstrap) kỳ diệu, nó khắc phục hoàn toàn nhược điểm quá khớp của cây đơn lẻ."*
- **Sau (V7)**:  
  *"Random Forest (RF) là một thuật toán học máy kết hợp (ensemble learning) dựa trên kỹ thuật Bagging (Bootstrap Aggregating). Bằng cách kết hợp nhiều Decision Tree được huấn luyện độc lập trên các tập con dữ liệu lấy mẫu có hoàn lại (Bootstrap Sampling) và ngẫu nhiên hóa không gian đặc trưng tại mỗi điểm phân nhánh, Random Forest giúp giảm đáng kể phương sai của mô hình dự đoán mà không làm tăng độ chệch."*

---

### Ví dụ 3: Giản lược công thức lý thuyết OOB
- **Vị trí**: `chapters/chapter2.tex` (Mục 2.3, Mẫu Out-of-Bag)
- **Trước (V6)**:  
  *"Ta chứng minh xác suất một mẫu không được chọn vào túi huấn luyện qua giới hạn giải tích:  
  $P(\text{không chọn}) = \lim_{N \to \infty} (1 - 1/N)^N = 1/e \approx 0.3679$. Đây là bằng chứng toán học bất biến khẳng định 36.8% dữ liệu luôn nằm ngoài túi để thẩm định mô hình."*
- **Sau (V7)**:  
  *"Trong quá trình lấy mẫu Bootstrap với cỡ mẫu $N$, ước tính có khoảng 36.8% số lượng quan sát không được lựa chọn vào tập con huấn luyện của một cây cụ thể. Những mẫu này được gọi là mẫu Out-of-Bag (OOB) và đóng vai trò như một tập kiểm định ngoài mẫu độc lập tự nhiên để đánh giá cây mà không cần phân chia thêm dữ liệu."*

---

### Ví dụ 4: Giản lược công thức suy diễn phương sai Ensemble
- **Vị trí**: `chapters/chapter2.tex` (Mục 2.3, Cơ chế giảm phương sai)
- **Trước (V6)**:  
  *"Công thức phương sai ensemble tổng quát được khai triển:  
  $\mathrm{Var}(\bar{T}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$. Khi số cây $B \to \infty$, số hạng thứ hai triệt tiêu, phương sai của rừng hội tụ về $\rho \sigma^2$, chứng minh sự vượt trội vĩnh cửu của rừng so với cây đơn lẻ."*
- **Sau (V7)**:  
  *"Về mặt thống kê, phương sai của mô hình kết hợp phụ thuộc vào số lượng cây và hệ số tương quan giữa các cây. Việc ngẫu nhiên hóa tập con đặc trưng tại mỗi nút giúp giảm độ tương quan giữa các cây đơn lẻ, qua đó làm giảm phương sai dự đoán chung của toàn bộ tập hợp khi kích thước rừng đủ lớn."*

---

### Ví dụ 5: Quy ước Ma trận chi phí Bayes
- **Vị trí**: `chapters/chapter2.tex` (Mục 2.4, Lý thuyết quyết định nhạy chi phí)
- **Trước (V6)**:  
  *"Hàm chi phí $C_{i,j}$ phạt nhầm lẫn giữa các ca bệnh. Ký hiệu lúc thì $C(\text{pred}, \text{true})$, lúc lại $C(\text{true}, \text{pred})$ gây bối rối cho người đọc."*
- **Sau (V7)**:  
  *"Nghiên cứu thống nhất quy ước ma trận chi phí $\mathbf{C} \in \mathbb{R}^{K \times K}$, trong đó phần tử $\mathbf{C}(i, j)$ xác định chi phí phạt khi mẫu có nhãn thực tế là lớp $i$ nhưng bị mô hình dự đoán nhầm thành lớp $j$. Khi đó, rủi ro kỳ vọng Bayes của quyết định chọn lớp $j$ được tính bởi:  
  $R(j \mid \mathbf{x}) = \sum_{i=0}^{K-1} P(Y = i \mid \mathbf{x}) \mathbf{C}(i, j)$."*

---

### Ví dụ 6: Phân biệt quy mô huấn luyện in-fold và toàn bộ dữ liệu
- **Vị trí**: `chapters/chapter3.tex` (Mục 3.3, Thiết kế phân loại phân cấp)
- **Trước (V6)**:  
  *"Stage 1 được huấn luyện trên 4.045 mẫu, còn Stage 2 được huấn luyện trên 1.029 mẫu có nhãn PHQ-9 $\ge 5$." (Gây hiểu nhầm là huấn luyện trên toàn bộ dữ liệu rồi kiểm tra, vi phạm rò rỉ dữ liệu)*
- **Sau (V7)**:  
  *"Trong mỗi fold huấn luyện của quy trình Nested Cross-Validation, Stage 1 được huấn luyện trên toàn bộ phân vùng huấn luyện nội bộ của fold đó ($N_{\text{train}} \approx 3.236$ mẫu). Trong khi đó, Stage 2 được huấn luyện nghiêm ngặt chỉ trên các mẫu có nhãn thực tế $\text{PHQ-9} \ge 5$ nằm trong chính phân vùng huấn luyện nội bộ này ($N_{\text{train, S2}} \approx 823$ mẫu). Các con số $N = 4.045$ và $N = 1.029$ là quy mô tổng thể của toàn bộ tập Legacy Development."*

---

### Ví dụ 7: Đánh giá mô hình phẳng 5 lớp $T_5$
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.2, Kết quả mô hình cơ sở $T_5$)
- **Trước (V6)**:  
  *"Kết quả Macro-F1 = 0.2755 của Random Forest là một thành tựu đáng kinh ngạc, vượt mặt hoàn toàn các đối thủ sừng sỏ khác trong ngành."*
- **Sau (V7)**:  
  *"Trong số các cấu hình được đánh giá trên bài toán 5 lớp phẳng $T_5$, Random Forest đạt Macro-F1 = $0.2755 \pm 0.0270$ và Weighted-F1 = $0.6437 \pm 0.0087$. Mặc dù dẫn đầu về Macro-F1, điểm số này cho thấy việc phân tách chi tiết cả 5 mức độ PHQ-9 trực tiếp từ dữ liệu sinh lý ngoại vi là một nhiệm vụ rất khó khăn do sự mất cân bằng lớp và độ đặc hiệu hạn chế của các biến số dạng bảng."*

---

### Ví dụ 8: Phân tích phép chiếu PCA 2D
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.4, Phân tích phân bố dữ liệu)
- **Trước (V6)**:  
  *"Hình chiếu PCA 2D là bằng chứng đanh thép chứng minh các lớp dữ liệu bị trộn lẫn vào nhau một cách hỗn loạn, khẳng định dữ liệu NHANES đã chạm trần thông tin tuyệt đối."*
- **Sau (V7)**:  
  *"Cần lưu ý rằng PCA chỉ là một phép chiếu tuyến tính giảm chiều xuống không gian 2 chiều, phản ánh các hướng biến thiên lớn nhất của dữ liệu chứ không thể hiện đầy đủ toàn bộ ranh giới phi tuyến trong không gian đa chiều gốc. Dù vậy, mức độ chồng lấn cao của các điểm dữ liệu trên mặt phẳng chiếu cung cấp một góc nhìn trực quan về nguyên nhân gây khó khăn cho các mô hình phân loại phẳng."*

---

### Ví dụ 9: Diễn giải mối tương quan thống kê
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.4, Ma trận tương quan đặc trưng)
- **Trước (V6)**:  
  *"Tương quan dương giữa chỉ số BMI, huyết áp và điểm PHQ-9 chứng minh rằng béo phì và huyết áp cao là nguyên nhân trực tiếp dẫn tới rối loạn trầm cảm nặng."*
- **Sau (V7)**:  
  *"Hệ số tương quan Spearman cho thấy mối liên hệ tuyến tính yếu đến vừa giữa một số chỉ số sinh lý (như BMI, số lượng bệnh mạn tính) và điểm PHQ-9. Tuy nhiên, mối tương quan thống kê này không đồng nghĩa với quan hệ nhân quả y sinh, mà chỉ phản ánh sự đồng xuất hiện của các rối loạn chuyển hóa ở nhóm đối tượng có triệu chứng tự báo cáo."*

---

### Ví dụ 10: Thống nhất số ca bỏ sót của mô hình $E_2$
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.5, Bảng 4.4 và đoạn phân tích)
- **Trước (V6)**:  
  *(Trong cùng một báo cáo: Bảng 4.4 ghi 95 ca, đoạn văn ghi 98 ca, Bảng 4.6 ghi 119 ca)*
- **Sau (V7)**:  
  *"Thống nhất chính xác con số **98 ca** trên toàn bộ văn bản và các bảng biểu, đúng theo ma trận nhầm lẫn out-of-fold lưu trữ tại `results/tables/E2_Hierarchical_RF_OptThresh_confusion_matrix.csv`."*

---

### Ví dụ 11: Đánh giá sự đánh đổi sai số trong mô hình nhạy chi phí $E_{6b}$
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.6, Đánh đổi sai số)
- **Trước (V6)**:  
  *"Mô hình chi phí Bayes $E_{6b}$ đã tiêu diệt hoàn toàn mối nguy hiểm bỏ sót bệnh nhân nặng, mang lại sự an toàn tuyệt đối cho người bệnh."*
- **Sau (V7)**:  
  *"Khi áp dụng ma trận chi phí Bayes $M_1$, số ca nặng bị bỏ sót thành bình thường ($C_2^+ \to C_0$) giảm từ 120 ca ($E_1$) xuống còn 62 ca (giảm 48.33\%), và tiếp tục giảm xuống 19 ca với ma trận phạt cao $M_2$. Đổi lại, số lượng đối tượng bình thường bị xếp nhầm vào nhóm có triệu chứng tăng lên tương ứng. Đây là sự đánh đổi tự nhiên trong bài toán sàng lọc cộng đồng."*

---

### Ví dụ 12: Đánh giá mô hình Ordinal Logistic Regression ($E_{7a}$)
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.5, Phân tích kết quả $E_7$)
- **Trước (V6)**:  
  *"Mô hình OLR chứng minh trí tuệ nhân tạo thứ bậc là giải pháp vạn năng, đè bẹp tất cả các kiến trúc phân cấp phức tạp khác."*
- **Sau (V7)**:  
  *"Mô hình Ordinal Logistic Regression cân bằng ($E_{7a}$) đạt Balanced Accuracy cao nhất (0.5151), phân bổ độ nhạy tương đối đồng đều trên cả 3 mức độ ($C_0: 61.2\%$, $C_1: 32.9\%$, $C_2^+: 60.4\%$) và duy trì độ ổn định tốt qua kiểm định ngoại suy theo thời gian (0.5073). Kết quả này cho thấy tính ổn định của các ranh giới siêu phẳng song song trong bài toán thứ bậc."*

---

### Ví dụ 13: Mô tả hiện tượng vi phạm tính đơn điệu trong CumBin
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.5, Mô hình Cumulative Binary)
- **Trước (V6)**:  
  *"Mô hình CumBin bị vỡ vụn toán học do các bộ phân loại nhị phân đánh nhau, tạo ra các xác suất âm vô nghĩa."*
- **Sau (V7)**:  
  *"Do hai bộ phân loại nhị phân $P(Y \ge 1)$ và $P(Y \ge 2)$ được huấn luyện độc lập, hiện tượng vi phạm tính đơn điệu xác suất $P(Y \ge 2) > P(Y \ge 1)$ xuất hiện ở 19.53% số dự đoán ở Logistic Regression và 5.32% ở Random Forest. Khi áp dụng giải thuật hiệu chỉnh ngưỡng (Isotonic Probability Clipping), tính đơn điệu được bảo đảm nhưng dẫn tới hiện tượng triệt tiêu xác suất lớp trung gian $C_1$ tại các điểm vi phạm."*

---

### Ví dụ 14: Nhận định về thang đo PHQ-9
- **Vị trí**: `chapters/chapter5.tex` (Mục 5.4, Hạn chế của nghiên cứu)
- **Trước (V6)**:  
  *"Các đối tượng trong tập dữ liệu đều là bệnh nhân trầm cảm nặng đã được ghi nhận vào hồ sơ y tế."*
- **Sau (V7)**:  
  *"Thang đo PHQ-9 là công cụ tầm soát dựa trên bảng câu hỏi tự điền, do đó điểm số thu được phản ánh mức độ triệu chứng tự báo cáo trong 2 tuần gần nhất chứ không thay thế cho chẩn đoán lâm sàng chính thức của bác sĩ tâm thần học."*

---

### Ví dụ 15: Định hướng nghiên cứu tương lai
- **Vị trí**: `chapters/chapter5.tex` (Mục 5.5, Hướng phát triển)
- **Trước (V6)**:  
  *"Trong tương lai gần, hệ thống sẽ được triển khai trực tiếp vào các bệnh viện lớn để thay thế hoàn toàn bác sĩ tâm thần."*
- **Sau (V7)**:  
  *"Các hướng phát triển tiếp theo có thể tập trung vào việc tích hợp dữ liệu đa phương thức (kết hợp bảng số liệu với giọng nói hoặc văn bản), thu thập tín hiệu sinh lý liên tục từ thiết bị đeo (HRV, giấc ngủ) và thử nghiệm các kiến trúc học sâu thứ bậc nhạy chi phí."*

---

### Ví dụ 16: Diễn đạt đóng góp của các nhóm đặc trưng (Ablation Study)
- **Vị trí**: `chapters/chapter4.tex` (Mục 4.1, Nghiên cứu loại bỏ đặc trưng)
- **Trước (V6)**:  
  *"Nhóm xét nghiệm máu $F_1$ hoàn toàn vô dụng và là gánh nặng cho mô hình, trong khi nhóm bệnh mạn tính $F_2$ là vị cứu tinh thần thánh."*
- **Sau (V7)**:  
  *"Các xét nghiệm máu và chỉ số thể chất đơn thuần ($F_1$) chỉ mang lại hiệu năng phân tách khiêm tốn (Macro-F1 $\approx 0.19$). Sự bổ sung nhóm tiền sử bệnh mạn tính ($F_2$) đóng vai trò quan trọng nhất trong việc cải thiện hiệu năng (+12.6% đến +23.1% Macro-F1), phản ánh mối liên hệ giữa các bệnh lý mạn tính kéo dài và sức khỏe tinh thần."*

