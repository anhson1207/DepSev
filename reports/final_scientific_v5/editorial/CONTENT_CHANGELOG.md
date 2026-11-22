# NHẬT KÝ THAY ĐỔI VÀ TỐI ƯU HÓA NỘI DUNG (CONTENT CHANGELOG V3 -> V4)

## 1. TỔNG QUAN CÁC THAY ĐỔI CỐT LÕI

| Hạng mục | Phiên bản V3 (Cũ) | Phiên bản V4 (Mới - Tinh tuyển) | Tác động học thuật |
| :--- | :--- | :--- | :--- |
| **Loại tài liệu** | "Khóa luận tốt nghiệp đại học hệ chính quy" | **"BÁO CÁO DỰ ÁN KHOA HỌC"** | Chuẩn hóa chính xác tính chất nghiên cứu khoa học chuyên ngành IT/ML. |
| **Dung lượng trang** | 70 trang (nhiều bảng lớn và diễn giải dài dòng) | **~45–50 trang nội dung chính** (+ Phụ lục kỹ thuật) | Đạt độ cô đọng cao, tăng mật độ thông tin khoa học, loại bỏ hoàn toàn phần thừa. |
| **Số lượng công thức** | >35 công thức (nhiều biến đổi toán sơ cấp, Taylor, Gini) | **11 công thức trọng tâm** trong thân bài | Loại bỏ lý thuyết giáo trình; tập trung vào cơ chế toán học của mô hình đề xuất. |
| **Hệ thống hình ảnh** | 24 hình (một số biểu đồ phụ trợ trùng lặp thông điệp) | **16 hình vẽ chất lượng cao** (vector / 300 DPI) | Mỗi hình ảnh đóng vai trò minh chứng trực tiếp cho một luận điểm thực nghiệm. |
| **Hệ thống bảng biểu** | 10 bảng trong thân bài (bảng 34 biến chiếm 2 trang) | **7 bảng cốt lõi** trong thân bài, chuyển chi tiết sang Phụ lục | Bảng thân bài tinh gọn, dễ đối chiếu số liệu; không gây ngắt quãng dòng đọc. |
| **Văn phong học thuật** | Còn tồn tại một số câu từ khuôn mẫu ("trong bối cảnh", "bước chuyển mình") | **Học thuật phản biện khách quan, tự nhiên, khiêm tốn** | Loại bỏ hoàn toàn cảm giác văn bản do AI tạo tự động; câu văn ngắn gọn, sắc sảo. |

---

## 2. CHI TIẾT THAY ĐỔI THEO TỪNG CHƯƠNG

### Chương 1: Giới thiệu (Đặt vấn đề)
- **Cắt bỏ:** Các đoạn văn mô tả chung chung về chuyển đổi số y tế và các số liệu dịch tễ không cần thiết; giảm 40% độ dài phần mở đầu.
- **Bổ sung / Làm sắc nét:**
  - Định vị rõ bài toán: Học máy trên dữ liệu sinh lý và cận lâm sàng dạng bảng (Tabular Health Data) từ khảo sát cộng đồng NHANES.
  - Trình bày mạch lạc 5 câu hỏi nghiên cứu cụ thể (RQ1–RQ5).
  - Tuyên bố minh bạch nguyên tắc kỹ thuật bắt buộc: **Zero Data Leakage** (không rò rỉ bất kỳ câu hỏi nào của DPQ vào ma trận đặc trưng đầu vào $\mathbf{X}$; cô lập hoàn toàn khâu điền khuyết và chuẩn hóa in-fold).
- **Hình ảnh:** Tích hợp Hình 1.1 (`fig1_problem_formulation.png`) làm sơ đồ khái quát toàn bộ bài toán.

### Chương 2: Cơ sở lý thuyết và các nghiên cứu liên quan
- **Cắt bỏ triệt để:**
  - Toàn bộ các định nghĩa giáo trình đại cương về Hồi quy Logistic, Cây quyết định, Rừng ngẫu nhiên và XGBoost (xóa bỏ công thức khai triển Taylor bậc hai, gradient, Hessian, entropy, Gini, bagging).
  - Các công thức mất mát Cross-Entropy và đạo hàm phổ thông.
- **Tập trung chuyên sâu:**
  - Thang đo chuẩn PHQ-9 (0–27 điểm) và 5 mức độ lâm sàng (Hình 2.1).
  - Cơ chế sinh lý học kết nối 4 hệ thống chỉ số ngoại vi với triệu chứng trầm cảm (Hình 2.2).
  - Bản chất hình học của hai rào cản cốt lõi: Mất cân bằng lớp cực đoan ($\text{IR} = 68.5:1$) và hiện tượng chồng lấn phân phối sinh lý (Hình 2.3).
  - Bản chất cơ chế 3 trường phái: Phẳng (Flat), Phân cấp (Hierarchical: Hard vs Soft & Bayes Cost Matrix), Thứ bậc (Ordinal: OLR & Cumulative Binary) (Hình 2.4).
  - Bảng tổng hợp đối sánh 5 nhóm nghiên cứu liên quan tiêu biểu và chỉ rõ khoảng trống nghiên cứu (Bảng 2.1).

### Chương 3: Dữ liệu và phương pháp nghiên cứu
- **Tái cấu trúc luồng xử lý:**
  - Mô tả cô đọng kiến trúc 7 module khoa học của DepSev-ML (Hình 3.1).
  - Tách bạch rõ ràng giữa hai tập dữ liệu: Legacy Cohort (NHANES 2013–2014, $N=5.054$) và Multi-cycle Cohort (NHANES 2011–2016, $N=14.581$).
  - Trực quan hóa quy trình tiền xử lý in-fold và nguyên tắc bảo toàn Zero Data Leakage (Hình 3.2).
  - Rút gọn phần đặc trưng: Trong thân bài chỉ trình bày sơ đồ phân tầng $F_1 \to F_4$ (Hình 3.3) và bảng tóm tắt 4 nhóm (Bảng 3.3); toàn bộ danh mục chi tiết 34 mã biến CDC được chuyển sang **Phụ lục A**.
  - Trực quan hóa chi tiết cơ chế phân cấp cứng và điểm nghẽn lan truyền sai số (Hình 3.4), cơ chế phân cấp mềm kết hợp tối ưu chi phí Bayes (Hình 3.5), và mô hình thứ bậc OLR / Cumulative Binary (Hình 3.6).
  - Sơ đồ hóa quy trình kiểm định hai tầng: Nested 5-Fold Stratified CV và Temporal Validation (Hình 3.7).

### Chương 4: Thực nghiệm và Thảo luận (Trọng tâm)
- **Chuẩn hóa lập luận thực nghiệm:**
  - Không so sánh Macro-F1 giữa $T_3$ và $T_5$ để tuyên bố phương pháp vượt trội; làm rõ $T_3$ là một bài toán phân tầng rủi ro lâm sàng độc lập.
  - Phân tích sâu sắc bước suy thoái hiệu năng lớn nhất từ $T_2$ sang $T_3$ (Macro-F1 giảm từ 0.6639 xuống 0.4354) do lớp nhẹ $C_1$ bị tách khỏi $C_0$ trong khi phân phối sinh lý học hoàn toàn chồng lấn (minh chứng qua PCA 2D Biplot Hình 4.3).
  - Phân tích đóng góp thực tế của việc tăng mẫu đa chu kỳ ($N=14.581$): chỉ ổn định nhẹ Balanced Accuracy nhưng không phá vỡ được trần hiệu năng, khẳng định rào cản thông tin nội tại của dữ liệu sinh lý ngoại vi.
  - Vạch trần điểm nghẽn thực nghiệm của phân cấp cứng ($E_1$): loại bỏ nhầm 120 ca nặng quy ước $C_2^+$ ngay tại Giai đoạn 1 do sai số một chiều.
  - Làm nổi bật đóng góp của phân cấp mềm và ma trận chi phí Bayes ($E_{6b}$): giảm 74.2% số ca bỏ sót nặng (từ 120 ca xuống còn 31 ca), đồng thời lượng hóa chính xác sự đánh đổi với số ca báo động giả (Hình 4.4).
  - Trình bày bảng tổng hợp Master Comparison 2 phần cân đối (Bảng 4.3) và hệ thống ma trận nhầm lẫn rõ nét (Hình 4.5 & 4.6).
  - Đánh giá kiểm định ngoại suy theo dòng thời gian (Temporal Validation trên NHANES 2015–2016, $N=4.888$, Bảng 4.4), làm nổi bật độ bền vững của mô hình Hồi quy Logistic thứ bậc cân bằng ($E_{7a}$).

### Chương 5: Kết luận và Hướng phát triển
- **Tổng kết sắc bén:**
  - Trả lời trực diện, gãy gọn 5 câu hỏi nghiên cứu RQ1–RQ5 bằng số liệu thực nghiệm đã kiểm chứng.
  - Nhìn nhận thẳng thắn 3 hạn chế thực tế của dữ liệu (bản chất cắt ngang, tín hiệu ngoại vi gián tiếp, nhãn PHQ-9 tự báo cáo).
  - Đề xuất 3 hướng phát triển khả thi cao trong tương lai (dữ liệu đa phương thức, cảm biến đeo chuỗi thời gian liên tục, mạng học thứ bậc sâu).

### Phụ lục kỹ thuật (Mới bổ sung)
- **Phụ lục A:** Danh mục đầy đủ 34 biến $F_4$ và 31 biến $F_{4\_\text{common31}}$ kèm mô tả y tế.
- **Phụ lục B:** Bảng thông số siêu tham số chi tiết của các mô hình $E_0 \to E_7$.
- **Phụ lục C:** Bảng chi tiết kết quả từng fold của quy trình kiểm định Nested 5-Fold Stratified CV.
- **Phụ lục D:** Chi tiết 4 ma trận chi phí Bayes ($M_0, M_1, M_2, M_3$) và giải thuật hiệu chỉnh đơn điệu trong CBC.

