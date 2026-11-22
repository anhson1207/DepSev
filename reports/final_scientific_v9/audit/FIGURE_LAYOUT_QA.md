# BÁO CÁO KIỂM THỬ GIAO DIỆN HÌNH ẢNH VÀ DÀN TRANG (FIGURE & LAYOUT QA) — DEPSEV-ML V8

**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản:** Báo cáo khoa học V8 (`DepSev_ML_Scientific_Report_V8.pdf`)  
**Ngày thực hiện:** 30/10/2026  

---

## 1. Mục tiêu kiểm định giao diện và hình ảnh

1. **Thiết kế lại các hình ảnh kém chất lượng:** Sửa dứt điểm tình trạng chữ bé, chữ đè lên nhau, biểu đồ quá tải dữ liệu hoặc không khớp nội dung phân tích.
2. **Loại bỏ hiện tượng tràn lề (Overfull \hbox) và lỗi tham chiếu gãy (Broken references `??`):** Bảo đảm toàn bộ bảng biểu, biểu đồ và văn bản nằm trọn vẹn trong khung lề in chuẩn (539.0 pt).
3. **Đạt chuẩn xuất bản khoa học:** Toàn bộ 19 hình ảnh trong báo cáo được tái tạo ở độ phân giải cao 300 DPI bằng script Python tự động (`generate_perfect_figures.py`).

---

## 2. Danh mục các hình ảnh và bảng biểu được nâng cấp trong V8

| Tên hình / Bảng | Vị trí trong PDF V8 | Nội dung cũ (V7) | Nâng cấp trong V8 | Dữ liệu gốc trích xuất | Kết quả QA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bảng 2.1** | Trang 35 | Bảng 5 cột chật chội, chứa Kaggle Benchmark phi học thuật | Thiết kế lại 4 cột tinh gọn (`p{3.6cm} p{2.4cm} p{3.2cm} X`), loại bỏ Kaggle, tập trung vào 5 công trình học thuật | Nghiên cứu y văn chính thức | **PASS** (Không tràn lề, chữ nét, thoáng đãng) |
| **Hình 4.6** | Trang 57 | Ma trận tương quan đặc trưng nội bộ $34 \times 34$ không có biến mục tiêu PHQ-9 | Thiết kế lại 2 panel: **(a) Spearman $\rho$ với PHQ-9** cho các biến liên tục; **(b) Cramer's V với PHQ-9** cho các biến phân loại | `results/tables/overlap_spearman_correlations.csv` & `overlap_categorical_associations.csv` | **PASS** (Khớp 100% nội dung phân tích trong Chương 4) |
| **Hình 4.7** | Trang 58 | 15 boxplot nhỏ li ti, chữ nhãn bị cắt, khó đọc | Rút gọn thành **lưới 2x3 gồm 6 biến liên tục tiêu biểu** (BMI, PIR, Giờ ngủ, Albumin, Nhịp tim, Thời gian ngồi tĩnh tại), font chữ lớn, hiển thị rõ đơn vị đo | `results/tables/overlap_numerical_summary.csv` | **PASS** (Trực quan, dễ so sánh trung vị và khoảng IQR) |
| **Hình 4.8** | Trang 59 | 6 biểu đồ con PCA nhỏ chật, thiếu chú thích | Thiết kế thành **2 panel lớn**: **(a) 5 mức độ $T_5$**; **(b) 3 nhóm $T_3$** làm nổi bật đường bao mật độ và sự lọt trọn vẹn của lớp $C_1$ vào $C_0$ | `results/tables/overlap_pca_coordinates.csv` | **PASS** (Ghi rõ tỷ lệ phương sai PC1: 12.3%, PC2: 8.8%, Tổng: 21.1%) |
| **Hình 1.1** | Trang 18 | Sơ đồ luồng bài toán nhiều chữ, hộp dính sát nhau | Giản lược câu chữ, tăng khoảng cách padding giữa các khối, bo góc mượt mà | Sơ đồ phương pháp luận | **PASS** |
| **Hình 1.2** | Trang 21 | Sơ đồ tiến trình nghiên cứu $T_5 \to T_3$ rối mắt | Tái thiết kế với 2 khối chính rõ ràng (Stage 1 và Stage 2), mũi tên chỉ hướng rành mạch | Tiến trình hai giai đoạn | **PASS** |
| **Hình 2.4** | Trang 29 | Sơ đồ Decision Tree chữ nhỏ | Chuẩn hóa font chữ 10--12pt, làm nổi bật tiêu chuẩn Gini Impurity | Lý thuyết Decision Tree | **PASS** |
| **Hình 2.5** | Trang 30 | Sơ đồ kiến trúc Random Forest rườm rà | Tinh gọn 3 giai đoạn: Bootstrap $\to$ Cây con $\to$ Soft Voting | Lý thuyết Random Forest | **PASS** |
| **Hình 3.1** | Trang 38 | Sơ đồ "7 module" kiểu phần mềm | Đổi thành sơ đồ quy trình 7 bước nghiên cứu thực nghiệm | Quy trình DepSev-ML | **PASS** |
| **Hình 3.2** | Trang 41 | Sơ đồ tiền xử lý in-fold nhiều chữ | Tinh gọn quy trình cô lập tham số scaler/imputer trên tập huấn luyện | Quy trình In-Fold CV | **PASS** |
| **Hình 3.4** | Trang 46 | Sơ đồ phân cấp cứng dính nhãn | Làm nổi bật điểm nghẽn lan truyền sai số một chiều ($C_2^+ \to C_0$) | Phân cấp cứng $E_1 \to E_3$ | **PASS** |
| **Hình 3.5** | Trang 48 | Sơ đồ phân cấp mềm kết hợp chi phí Bayes phức tạp | Tinh gọn công thức xác suất toàn phần và ma trận chi phí bất đối xứng | Phân cấp mềm $E_4 \to E_6$ | **PASS** |
| **Bảng 4.6 / 4.7** | Trang 65--66 | Bảng đối sánh tổng thể bị lỗi số liệu $E_0$ Recall và dùng từ "Trường phái" | Cập nhật số liệu chuẩn xác 100%, đổi tên cột thành "Nhóm mô hình", bổ sung ghi chú giải thích độ chênh giữa 5-fold mean và pooled OOF matrix | `results/tables/master_benchmark_t3.csv` | **PASS** (Bố cục 2 phần A và B liền mạch, chuyên nghiệp) |

---

## 3. Kết quả kiểm tra biên dịch và dàn trang tự động

Script kiểm thử bằng PyMuPDF (`fitz`) thực hiện trên tệp PDF hoàn chỉnh:
- **Tổng số trang:** **87 trang** (Gọn hơn bản V7 89 trang nhờ tinh gọn văn bản và cấu trúc lại bảng biểu).
- **Lỗi tham chiếu gãy (`??`):** **0 lỗi** (Đạt tỷ lệ hoàn hảo 100%).
- **Lỗi tràn lề (Bouding Box $x_1 > 542.0$ pt):** **0 lỗi**. Toàn bộ các dòng chữ và bảng biểu đều nằm an toàn bên trong lề trang.
- **Thư mục ảnh kiểm thử trực quan (Visual Previews):** Đã xuất 11 ảnh trang mẫu tại `reports/final_scientific_v8/audit/previews/` (gồm các trang 35, 56, 57, 58, 59, 65, 66, 83, 85, 86, 87) để phục vụ kiểm tra ngoại quan.

