# BÁO CÁO NỘI DUNG ĐÃ TINH GIẢN VÀ LOẠI BỎ (DEPSEV-ML V10)
**Tập tin:** `reports/final_scientific_v10/audit/CONTENT_REMOVED_OR_SIMPLIFIED.md`  
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Ngày thực hiện:** 31/10/2026  
**Người thực hiện:** Antigravity Research Team  

---

## 1. TỔNG QUAN

Để nâng cao tính cô đọng học thuật, tránh lặp lại công thức toán học và loại bỏ các thành phần rườm rà không phục vụ trực tiếp cho mục tiêu nghiên cứu Machine Learning, bản báo cáo V10 đã tiến hành rà soát và tinh giản có hệ thống trên toàn bộ 5 chương. Tài liệu này ghi nhận chi tiết các nội dung đã được loại bỏ, rút gọn hoặc tái cấu trúc.

---

## 2. TINH GIẢN CÔNG THỨC TOÁN HỌC VÀ TRÁNH LẶP LẠI GIỮA CÁC CHƯƠNG

### 2.1. Hiện tượng trùng lặp trong V9:
Trong phiên bản V9, một số công thức toán học xuất hiện tới hai lần trong cả Chương 2 (Tổng quan nghiên cứu) và Chương 3 (Phương pháp đề xuất):
- **Công thức định tuyến xác suất mềm (Soft Hierarchical Routing):**
  $$P(y = c \mid \mathbf{x}) = \sum_{s \in \mathcal{S}} P(y = c \mid s, \mathbf{x}) \cdot P(s \mid \mathbf{x})$$
- **Quy tắc quyết định rủi ro Bayes (Bayes Minimum Risk Decision Rule):**
  $$\hat{y} = \arg\min_{c \in \mathcal{C}} \sum_{k \in \mathcal{C}} L(c, k) \cdot P(y = k \mid \mathbf{x})$$

Việc lặp lại này làm loãng cấu trúc chương, khiến Chương 2 mang tính chất mô tả phương pháp của tác giả thay vì khảo cứu y văn độc lập.

### 2.2. Xử lý trong V10:
- **Trong Chương 2 (Cơ sở lý thuyết & Tổng quan):** Rút gọn thành đoạn phân tích khái niệm (conceptual review), dẫn nguồn tài liệu tham khảo quốc tế về lý thuyết phân loại phân cấp mềm và tối ưu chi phí Bayes trong y tế. Không hiển thị các khối phương trình toán học chi tiết.
- **Trong Chương 3 (Phương pháp nghiên cứu):** Đặt trọn vẹn và chuẩn hóa các khối công thức toán học với đầy đủ định nghĩa không gian trạng thái, ký hiệu chỉ số, và ma trận tổn thất $L(c, k)$.

---

## 3. LOẠI BỎ CÁC YẾU TỐ KHÔNG PHÙ HỢP TRONG HÌNH ẢNH

### 3.1. Hình 2.1 (`fig2_phq9_tiers.png`):
- **Vấn đề ở V9:** Hình chứa các hộp văn bản hướng dẫn y khoa/điều trị mang tính chất tờ rơi bệnh viện (ví dụ: *"Theo dõi định kỳ"*, *"Khám bác sĩ chuyên khoa"*, *"Trị liệu tâm lý"*, *"Cân nhắc dùng thuốc"*). Các dòng chữ này bị dính chặt vào nhau, che khuất các ngưỡng điểm và làm mất đi bản chất một nghiên cứu Khoa học Máy tính về tái cấu trúc không gian nhãn.
- **Cải tiến ở V10:**
  - Loại bỏ $100\%$ các chỉ dẫn điều trị lâm sàng mang tính chủ quan.
  - Tách thành 2 phần rõ rệt:
    - **Phần A (Part A - 5 Classes $T_5$):** Thể hiện 5 mức phân loại nguyên bản ($C_0$: Không có triệu chứng, $C_1$: Nhẹ, $C_2$: Vừa, $C_3$: Trung bình nặng, $C_4$: Nặng) cùng khoảng điểm PHQ-9, số lượng mẫu thực tế ($n$) và tỷ lệ phần trăm trong tập dữ liệu NHANES ($N = 4{,}880$).
    - **Phần B (Part B - 3 Clinical Tiers $T_3$):** Thể hiện trực quan cách gom cụm $C_2 \cup C_3 \cup C_4$ thành mức $C_2^+$ (Cần can thiệp lâm sàng) bằng dấu ngoặc gộp rõ ràng, giải thích tính cấp thiết trong sàng lọc y tế cộng đồng.

### 3.2. Hình 3.4 (`fig3_4_hard_hierarchical.png`) và Hình 3.5 (`fig3_5_soft_cost_sensitive.png`):
- **Vấn đề ở V9:** Xuất hiện sự nhập nhằng giữa ký hiệu "M1/M2" vừa chỉ mô hình phân cấp giai đoạn 1/giai đoạn 2, vừa trùng với ký hiệu ma trận tổn thất Bayes $M_0, M_1, M_2, M_3$. Nhiều khối hình vẽ có quá nhiều chữ dẫn tới tràn viền và khó đọc khi in đen trắng.
- **Cải tiến ở V10:**
  - Chuẩn hóa tên gọi thành **Giai đoạn 1 (Stage 1 Classifier)** và **Giai đoạn 2 (Stage 2 Classifier)**.
  - Dành riêng ký hiệu $M_0 \dots M_3$ cho các ma trận tổn thất rủi ro Bayes trong Bảng 3.3.
  - Rút gọn nội dung trong các khối sơ đồ thành từ khóa kỹ thuật súc tích, nâng kích thước chữ lên $\ge 10$ pt.

---

## 4. TINH GIẢN NỘI DUNG LÝ THUYẾT THUẬT TOÁN CƠ BẢN

### 4.1. Cắt giảm các khối lý thuyết giáo trình:
- Các thuật toán học máy cổ điển như **Logistic Regression**, **Decision Tree** và **XGBoost** là những thuật toán tiêu chuẩn đã được cộng đồng khoa học biết rõ. Trong các bản thảo ban đầu, việc dành nhiều trang để trình bày lại hàm mất mát cross-entropy, tiêu chí phân chia Gini/Entropy hay khai triển chuỗi Taylor của Gradient Boosting đã làm tăng độ dày không cần thiết cho báo cáo.
- Trong V10, các tiểu mục này được rút gọn tối đa: chỉ trình bày lý do lựa chọn thuật toán, đặc điểm xử lý dữ liệu bảng (tabular data), cơ chế kiểm soát overfitting (regularization, early stopping), và bảng siêu tham số cụ thể đã dùng trong nghiên cứu.

---

## 5. BỎ CÁC ĐOẠN VĂN TỔNG KẾT RƯỜM RÀ VÀ THỪA THÃI

- **Loại bỏ các đoạn "recap" lặp lại nguyên văn bảng số liệu:** Trong V9, sau mỗi bảng thực nghiệm (ví dụ Bảng 4.1 hay Bảng 4.4), tác giả thường viết 2--3 đoạn văn chỉ để liệt kê lại từng con số đã có sẵn trên dòng của bảng.
- **Trong V10:** Thay thế bằng các đoạn thảo luận phân tích chuyên sâu (analytical insights): tập trung vào *ý nghĩa khoa học* của sự thay đổi chỉ số, giải thích tại sao Macro-F1 tăng trong khi Balanced Accuracy giảm, và bàn luận về ý nghĩa thực tiễn đối với hệ thống y tế dự phòng.

