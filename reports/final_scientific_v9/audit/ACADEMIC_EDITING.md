# NHẬT KÝ BIÊN TẬP HỌC THUẬT VÀ CHUẨN HÓA VĂN PHONG (ACADEMIC EDITING LOG) — DEPSEV-ML V8

**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản biên tập:** Báo cáo khoa học V8 (`DepSev_ML_Scientific_Report_V8.pdf`)  
**Ngày thực hiện:** 30/10/2026  
**Định vị văn phong:** Báo cáo nghiên cứu khoa học của sinh viên ngành Khoa học Máy tính (Computer Science), xưng "em", văn phong khách quan, khiêm tốn, trung thực, dựa trên bằng chứng thực nghiệm (evidence-based).

---

## 1. Mục tiêu và Nguyên tắc biên tập

1. **Loại bỏ triệt để văn phong máy móc (AI-generated tone):**
   - Cắt bỏ các cụm từ khoa trương, sáo rỗng, tự khen ngợi quá đà hoặc kịch tính hóa kết quả thực nghiệm.
   - Chuyển đổi các cấu trúc câu bị động, dịch thô từ tiếng Anh sang câu chủ động, mạch lạc, tự nhiên theo văn phong học thuật tiếng Việt.
2. **Chuẩn hóa thuật ngữ Machine Learning theo chuẩn ngành Computer Science:**
   - Tên các thuật toán và kỹ thuật học máy giữ nguyên danh pháp tiếng Anh chuẩn mực: `Random Forest (RF)`, `Decision Tree (DT)`, `Logistic Regression (LR)`, `XGBoost`, `Nested Cross-Validation`, `Stratified 5-Fold CV`, `Bootstrap Sampling`, `Soft Voting`, `Stacking`, `Ordinal Logistic Regression (OLR)`, `Cumulative Binary Classification (CBC)`.
   - Tuyệt đối không dịch gượng ép thành "Rừng ngẫu nhiên", "Cây quyết định", "Hồi quy Logistic".
   - Các khái niệm dữ liệu và thực nghiệm sử dụng thuật ngữ tiếng Việt chuẩn mực: tập huấn luyện, tập kiểm tra/kiểm định, đặc trưng đầu vào, nhãn mục tiêu, tiền xử lý dữ liệu in-fold, phân bố lớp, ma trận nhầm lẫn, dự đoán ngoài mẫu (out-of-fold).
3. **Tinh gọn cấu trúc và loại bỏ lặp lại:**
   - Cắt giảm các đoạn văn "bật mí kết quả trước" (result dumping) tại Chương 1.
   - Thay thế các đoạn văn tóm tắt lặp lại ở đầu các mục con bằng phần dẫn dắt ngắn gọn, đi thẳng vào bản chất kỹ thuật.

---

## 2. Bảng đối chiếu cụm từ loại bỏ và thay thế

| Cụm từ máy móc / Khoa trương (V7) | Vấn đề học thuật | Cụm từ chuẩn hóa (V8) | Vị trí áp dụng |
| :--- | :--- | :--- | :--- |
| **"Trường phái mô hình hóa"** | Quá hàn lâm, khoa trương, không phải thuật ngữ Computer Science tự nhiên | **"Hướng tiếp cận mô hình hóa" / "Phương pháp phân loại"** | Mục 2.4, Hình 2.5, Mục 3.1, Hình 4.9, Hình 4.13, Bảng 4.6 |
| **"Nguyên tắc cách ly bảo toàn tuyệt đối Zero Data Leakage"** | Từ ngữ phóng đại kiểu marketing, lặp đi lặp lại nhiều lần | **"Quy trình tiền xử lý in-fold độc lập để kiểm soát rò rỉ dữ liệu"** | Mục 1.2, Mục 3.1, Mục 3.2.3, Hình 3.2, Mục 3.4, Mục 5.1 |
| **"Kiến trúc toàn diện 7 module"** | Gây cảm giác hệ thống phần mềm thương mại thay vì quy trình nghiên cứu thực nghiệm | **"Quy trình nghiên cứu thực nghiệm gồm 7 bước" (Bước 1 $\to$ Bước 7)** | Mục 3.1, Hình 3.1 |
| **"Rào cản thông tin nội tại"** | Diễn đạt mơ hồ, thiếu chính xác về mặt toán học | **"Hiện tượng chồng lấn phân phối đặc trưng giữa các lớp"** | Mục 3.4, Mục 4.7, Tóm tắt |
| **"Suy thoái nghiêm trọng / sụp đổ khả năng nhận diện"** | Kịch tính hóa quá mức, thiếu tính định lượng khách quan | **"Suy giảm hiệu năng / khó khăn khi nhận diện lớp thiểu số"** | Mục 1.1, Mục 4.3, Mục 4.4, Bảng 4.3 |
| **"Triệt tiêu phần lớn phương sai"** | Sai về mặt toán học (Random Forest chỉ giảm phương sai theo tỷ lệ $\frac{1}{B} + \rho$, không thể "triệt tiêu") | **"Làm giảm đáng kể phương sai ngẫu nhiên"** | Mục 2.3.2 |
| **"Năng lực khái quát hóa vượt trội / Bằng chứng khoa học không thể phủ nhận"** | Giọng điệu AI tự khen, khẳng định chủ quan | **"Hiệu năng ổn định / Kết quả thực nghiệm ghi nhận"** | Mục 4.2, Mục 4.8 |
| **"Đối chuẩn chuẩn tắc"** | Dịch thô từ "canonical benchmark", tối nghĩa | **"Mô hình đối chuẩn / Đường cơ sở (Baselines)"** | Mục 2.3.1, Mục 3.1 |
| **"Chỉ số an toàn lâm sàng"** | Đề tài nghiên cứu dữ liệu dạng bảng dịch tễ, không phải thử nghiệm lâm sàng (clinical trial) | **"Số ca nặng bị phân loại nhầm thành bình thường ($C_2^+ \to C_0$)"** | Mục 3.1, Mục 3.6, Bảng 4.6, Bảng 4.7 |
| **"Cơ chế cách ly bảo toàn tuyệt đối"** | Lạm dụng từ "tuyệt đối", câu chữ sáo rỗng | **"Phân hoạch độc lập / Tiền xử lý độc lập từng fold"** | Mục 1.2, Mục 3.1 |

---

## 3. Nhật ký cắt tỉa nội dung và tinh gọn văn bản

### 3.1. Chương 1: Giới thiệu
- **Mục 1.2 (Phát biểu bài toán):** Thay thế đoạn nhấn mạnh "nguyên tắc cách ly bảo toàn tuyệt đối Zero Data Leakage" thành đoạn văn chuẩn mực giải thích quy trình tiền xử lý độc lập trên từng fold kiểm định chéo (In-Fold Preprocessing).
- **Mục 1.5 (Đóng góp nghiên cứu):** Loại bỏ toàn bộ việc "xả số liệu chi tiết" (result dumping) như việc trích dẫn trước các chỉ số cụ thể $E_1 \to E_7$ (như 120 ca, 62 ca, 0.4550...). Thay vào đó, viết lại thành 4 đóng góp khoa học mang tính phương pháp luận và phát hiện thực nghiệm cốt lõi.

### 3.2. Chương 2: Cơ sở lý thuyết và nghiên cứu liên quan
- **Mục 2.3 (Thuật toán Random Forest):** 
  - Đính chính phát biểu toán học về việc giảm phương sai: từ "triệt tiêu phương sai" thành "làm giảm đáng kể phương sai ngẫu nhiên của các cây đơn lẻ".
  - Giữ vững cấu trúc giải thích Random Forest dễ hiểu, trực quan, có sơ đồ minh họa Bootstrap Sampling và Feature Subspace Sampling.
- **Mục 2.4 (Các hướng tiếp cận mô hình hóa):** Đổi tên tiêu đề từ "Các trường phái mô hình hóa" thành "Các hướng tiếp cận mô hình hóa"; các tiểu mục thành "Phương pháp phân loại phẳng", "Phương pháp phân loại phân cấp", "Phương pháp phân loại thứ bậc".

### 3.3. Chương 3: Dữ liệu và phương pháp nghiên cứu
- **Mục 3.1 (Quy trình thực nghiệm):** Chuyển đổi toàn bộ khái niệm "Module 1 $\to$ Module 7" thành "Bước 1 $\to$ Bước 7" trong quy trình thực nghiệm khoa học, phản ánh đúng bản chất một đề tài nghiên cứu học máy.
- **Mục 3.2 và 3.4:** Chuẩn hóa thuật ngữ "kiểm soát rò rỉ dữ liệu in-fold" thay cho cụm từ "Zero Data Leakage".

### 3.4. Chương 4: Thực nghiệm và thảo luận
- **Mục 4.4.1 (Phân tích tương quan):** Viết lại hoàn toàn nội dung để bám sát biểu đồ 2 panel mới (Hình 4.6): Panel (a) Spearman rank correlation $\rho$ của các biến liên tục với PHQ-9 (Albumin: -0.164, PIR: -0.159, Sleep: -0.139, BMI: +0.114, Waist: +0.095, Pulse: +0.085); Panel (b) Cramer's V của các biến phân loại (HUQ010: 0.195, Gender: 0.144, PAQ665: 0.125, BPQ020: 0.109, DIQ010: 0.100).
- **Mục 4.4.2 (Biểu đồ hộp và PCA):** Bổ sung số liệu định lượng cho 6 biến liên tục tiêu biểu trong biểu đồ hộp 2x3 (Hình 4.7); phân tích chi tiết tỷ lệ 21.1% phương sai giải thích của PCA 2D (PC1: 12.3%, PC2: 8.8%) và giải thích hiện tượng chồng lấn phân phối của lớp nhẹ $C_1$ trong đường bao mật độ của lớp tối thiểu $C_0$ (Hình 4.8).
- **Mục 4.5 $\to$ 4.8:** Thống nhất dứt điểm số ca nặng bị bỏ sót ở cấu hình $E_2$ là 98 ca (loại bỏ hoàn toàn các vị trí nhầm thành 119 ca).

### 3.5. Chương 5: Kết luận và hướng phát triển
- Tinh gọn phần kết luận, bỏ các khẳng định tuyệt đối hóa; nêu bật 4 mô hình tiêu biểu đại diện cho 4 tiêu chí thực tiễn (Random Forest Tuned, $E_2$ Hard Tuned, $E_{7a}$ Balanced OLR, $E_{6b}$ Soft Bayes Risk).
- Bổ sung đánh giá khách quan về những giới hạn nội tại của dữ liệu cắt ngang NHANES và tính gián tiếp của các chỉ số sinh lý ngoại vi.

