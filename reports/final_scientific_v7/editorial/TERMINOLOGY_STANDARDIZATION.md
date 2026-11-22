# CHUẨN HÓA THUẬT NGỮ TRONG BÁO CÁO DEPSEV-ML V7
*(Terminology Standardization Document)*

Tài liệu này xác lập quy chuẩn thuật ngữ khoa học Machine Learning và Y tế cho toàn bộ báo cáo `DepSev-ML` phiên bản V7, tuân thủ chặt chẽ phong cách học thuật của sinh viên và nhà nghiên cứu Khoa học Máy tính (Computer Science).

---

## 1. TÊN THUẬT TOÁN HỌC MÁY (BẮT BUỘC GIỮ NGUYÊN TIẾNG ANH)

| Thuật ngữ Tiếng Anh chuẩn | Tên viết tắt | Dạng dịch sai/máy móc (ĐÃ LOẠI BỎ) | Ghi chú quy chuẩn |
| :--- | :---: | :--- | :--- |
| **Random Forest** | RF | Rừng ngẫu nhiên, rừng cây quyết định | Giữ nguyên tiếng Anh trong mọi văn cảnh. |
| **Decision Tree** | DT | Cây quyết định | Giữ nguyên tiếng Anh hoặc "cây quyết định đơn lẻ". |
| **Logistic Regression** | LR | Hồi quy Logistic, mô hình hồi quy | Giữ nguyên tiếng Anh. |
| **XGBoost** | XGB | Tăng cường độ dốc cực đại, Extreme Gradient Boosting | Giữ nguyên tiếng Anh. |
| **Ordinal Logistic Regression** | OLR | Hồi quy Logistic thứ bậc | Giữ nguyên tiếng Anh hoặc viết tắt OLR. |
| **Cumulative Binary Classification** | CumBin | Phân loại nhị phân lũy tích/tích lũy | Giữ nguyên tiếng Anh CumBin. |

---

## 2. CÁC THUẬT NGỮ KỸ THUẬT MACHINE LEARNING

| Thuật ngữ Tiếng Anh | Cách dùng chuẩn trong V7 | Cách diễn đạt sai/sáo rỗng (ĐÃ LOẠI BỎ) |
| :--- | :--- | :--- |
| **Nested Cross-Validation** | Nested Cross-Validation (hoặc kiểm định chéo lồng nhau) | Thẩm định chéo lồng ghép tầng bậc |
| **Stratified 5-Fold Cross-Validation** | Phân tầng 5-fold (hoặc 5-fold phân tầng) | Kiểm thử tầng hóa 5 nếp gấp |
| **Bootstrap Sampling** | Bootstrap Sampling | Lấy mẫu khởi động, rút mẫu hoàn lại tự thân |
| **Out-of-Bag (OOB)** | Out-of-Bag (hoặc mẫu ngoài túi) | Mẫu ngoài bao bì |
| **Out-of-Fold (OOF)** | Out-of-Fold (hoặc ngoài fold) | Mẫu ngoài nếp gấp |
| **Feature Selection** | Lựa chọn đặc trưng | Sàng lọc thuộc tính tinh nhuệ |
| **Feature Ablation** | Loại bỏ từng nhóm đặc trưng (Feature Ablation) | Cắt bỏ đặc trưng triệt để |
| **Baseline** | Mô hình cơ sở | Đường cơ sở vàng, mốc chuẩn tuyệt đối |
| **Hyperparameter Tuning** | Tối ưu hóa siêu tham số | Tinh chỉnh tham số siêu việt |
| **Soft Voting / Hard Voting** | Soft Voting / Hard Voting (bỏ phiếu mềm / cứng) | Biểu quyết mềm dẻo / biểu quyết thô |
| **Cost-Sensitive Classification** | Phân loại nhạy chi phí (Cost-Sensitive) | Phân loại nhạy cảm tổn thất |
| **Bayes Minimum Risk** | Rủi ro tối thiểu Bayes (Bayes Minimum Risk) | Cực tiểu hóa rủi ro Bayes thần kỳ |
| **Hard / Soft Hierarchical** | Phân cấp cứng / Phân cấp mềm | Phân tầng cứng / phân tầng linh hoạt |

---

## 3. THUẬT NGỮ DỮ LIỆU VÀ TẬP MẪU (DÙNG TIẾNG VIỆT HỌC THUẬT TỰ NHIÊN)

| Thuật ngữ Chuẩn | Thuật ngữ Tiếng Anh tương ứng | Từ ngữ cấm kỵ / máy móc |
| :--- | :--- | :--- |
| **Tập huấn luyện** | Training set | Tập xe lửa, tập rèn luyện |
| **Tập kiểm tra** | Test set | Tập khảo nghiệm |
| **Tập kiểm định ngoài mẫu** | Out-of-sample validation set | Tập thẩm định cô lập |
| **Dữ liệu đầu vào** | Input data | Đầu vào dữ liệu thô |
| **Nhãn dự đoán** | Predicted label | Nhãn tiên tri, nhãn phán đoán |
| **Nhãn thực tế** | Ground-truth label | Chân lý mặt đất, nhãn vàng chân lý |
| **Mức độ triệu chứng** | Severity level | Cấp bậc suy sụp |
| **Phân bố lớp** | Class distribution | Sự phân tán nhãn lớp |
| **Số mẫu dự đoán sai** | Misclassified samples / Misses | Số ca tử thương, số lỗi thảm họa |
| **Tiền xử lý dữ liệu** | Data preprocessing | Sơ chế dữ liệu |

---

## 4. QUY CHUẨN THUẬT NGỮ Y TẾ VÀ LÂM SÀNG

1. **Nhóm đối tượng nghiên cứu**:
   - Sử dụng: **"người tham gia khảo sát"**, **"mẫu dữ liệu"**, **"đối tượng"**.
   - Tuyệt đối tránh: *"bệnh nhân nặng"*, *"người hoàn toàn khỏe mạnh"* (vì dữ liệu NHANES là khảo sát dịch tễ học cộng đồng, không phải hồ sơ bệnh án nội trú).

2. **Quy ước nhóm nhãn**:
   - $C_0$: Triệu chứng tối thiểu hoặc không có triệu chứng ($\text{PHQ-9} \in [0, 4]$).
   - $C_1$: Triệu chứng mức độ nhẹ ($\text{PHQ-9} \in [5, 9]$).
   - $C_2^+$: **"Nhóm Nặng theo quy ước nghiên cứu"** ($\text{PHQ-9} \ge 10$, bao gồm các mức Vừa, Nặng vừa và Nặng).
   - Không được gọi $C_2^+$ là *"bệnh nhân trầm cảm lâm sàng chính thức"* vì PHQ-9 chỉ là thang điểm sàng lọc tự báo cáo, không thay thế chẩn đoán tâm thần học của bác sĩ chuyên khoa.

3. **Sai số phân loại và đánh đổi rủi ro**:
   - Lỗi $C_2^+ \to C_0$: Gọi là **"sai số bỏ sót ca nặng"** (hoặc lỗi nhảy 2 bước $C_2^+ \to C_0$).
   - Lỗi $C_0 \to C_2^+$: Gọi là **"sai số báo động nhầm"** (False Alarm).
   - Tuyệt đối tránh các cụm từ bi kịch hóa: *"hậu quả chết người"*, *"thảm họa y tế"*, *"gây nguy hiểm tính mạng"*. Dùng cách diễn đạt chừng mực: *"gây nguy cơ trì hoãn can thiệp hỗ trợ tâm lý"*, *"làm gia tăng gánh nặng kiểm tra lâm sàng không cần thiết"*.

