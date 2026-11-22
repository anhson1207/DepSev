# ĐỀ CƯƠNG CHI TIẾT BÁO CÁO DỰ ÁN KHOA HỌC DEPSEV-ML V4

## 1. CẤU TRÚC TỔNG THỂ VÀ PHÂN BỔ DUNG LƯỢNG

- **Trang bìa & Bìa phụ (3 trang):** Bìa chính, Bìa phụ Tiếng Việt, Bìa phụ Tiếng Anh (Chuẩn VNU-UET, danh xưng **BÁO CÁO DỰ ÁN KHOA HỌC**, GVHD: TS. Vũ Thị Hồng Nhạn, đầy đủ biểu trưng Logo UET trên cả 3 trang).
- **Phần mở đầu (Frontmatter, ~8 trang):**
  - Lời cảm ơn (chân thành, học thuật).
  - Lời cam đoan (liêm chính nghiên cứu).
  - Tuyên bố về sự hỗ trợ của AI (theo mẫu chuẩn VNU-UET 2026).
  - Tóm tắt (Tiếng Việt, ~350 từ).
  - Abstract (Tiếng Anh, ~350 từ).
  - Mục lục (Table of Contents).
  - Danh sách hình vẽ (List of Figures).
  - Danh sách bảng (List of Tables).
  - Danh mục ký hiệu và chữ cái viết tắt (Table of Abbreviations).
- **Phần nội dung chính (5 Chương, dự kiến 42–48 trang):**
  - Chương 1: Giới thiệu (4–5 trang).
  - Chương 2: Cơ sở lý thuyết và các nghiên cứu liên quan (6–7 trang).
  - Chương 3: Dữ liệu và phương pháp nghiên cứu (10–12 trang).
  - Chương 4: Thực nghiệm và thảo luận (16–18 trang).
  - Chương 5: Kết luận và hướng phát triển (3–4 trang).
- **Tài liệu tham khảo (~3 trang):** Chọn lọc 25–30 tài liệu y sinh học và Machine Learning cốt lõi.
- **Phụ lục kỹ thuật (~6–8 trang):**
  - Phụ lục A: Danh mục chi tiết 34 biến đầu vào và mã định danh CDC NHANES.
  - Phụ lục B: Không gian siêu tham số và cấu hình đóng băng mô hình (Frozen Specs).
  - Phụ lục C: Bảng kết quả chi tiết theo fold của kiểm định chéo Nested 5-Fold CV.
  - Phụ lục D: Ma trận nhầm lẫn mở rộng và các biến thể ma trận chi phí Bayes ($M_0 \to M_3$).

---

## 2. ĐỀ CƯƠNG CHI TIẾT TỪNG CHƯƠNG VÀ MỤC TIÊU NỘI DUNG

### CHƯƠNG 1. GIỚI THIỆU (4–5 trang)
- **1.1. Bối cảnh và vấn đề nghiên cứu:**
  - Thực trạng gánh nặng của rối loạn trầm cảm và sự thiếu hụt công cụ sàng lọc khách quan.
  - Thang đo tự báo cáo PHQ-9: ưu thế lâm sàng và rào cản thiên kiến chủ quan/tâm lý xã hội.
  - Tiềm năng của việc khai phá dữ liệu khám sức khỏe và xét nghiệm định kỳ từ NHANES.
  - *Mục tiêu:* Dẫn dắt người đọc trực tiếp vào bài toán, giải thích tại sao cần tiếp cận bằng học máy trên dữ liệu sinh lý dạng bảng.
- **1.2. Phát biểu bài toán toán học:**
  - Không gian đặc trưng đầu vào $\mathbf{x} \in \mathbb{R}^d$ và điểm số liên tục PHQ-9 ($s \in [0, 27]$).
  - Ánh xạ sang các không gian mục tiêu rời rạc: $T_2$ (nhị phân), $T_3$ (3 nhóm gộp), $T_4$ (4 mức), $T_5$ (5 mức chuẩn).
  - Các hàm mục tiêu đánh giá chính: Macro-F1 và Balanced Accuracy (lý do không dùng Accuracy).
  - Ràng buộc bắt buộc: **Zero Data Leakage** (tuyệt đối không đưa câu hỏi DPQ vào ma trận đầu vào; cô lập 100% khâu điền khuyết và chuẩn hóa in-fold).
- **1.3. Mục tiêu và câu hỏi nghiên cứu:**
  - Mục tiêu tổng quát của framework DepSev-ML.
  - 5 câu hỏi nghiên cứu cụ thể (RQ1: Đóng góp của các nhóm đặc trưng $F_1 \to F_4$; RQ2: Quy luật suy thoái theo độ chi tiết nhãn $T_2 \to T_5$; RQ3: Tác động của mở rộng dữ liệu đa chu kỳ Pooled NHANES; RQ4: Hiệu quả và điểm nghẽn của phân loại phân cấp cứng; RQ5: Đánh đổi giữa phân cấp mềm, tối ưu chi phí Bayes và phân loại thứ bậc).
- **1.4. Phạm vi và giới hạn nghiên cứu:**
  - Giới hạn trên dữ liệu cắt ngang dạng bảng NHANES 2011–2016, người trưởng thành $\ge 20$ tuổi.
  - Không sử dụng tín hiệu chuỗi thời gian liên tục (ECG, EEG).
  - Mục đích là phân tầng rủi ro dịch tễ học, không thay thế quy trình chẩn đoán tâm thần chuyên khoa.
- **1.5. Phương pháp tiếp cận và đóng góp của đề tài:**
  - Tóm tắt pipeline nghiên cứu 5 giai đoạn và 4 đóng góp thực chất có bằng chứng.
- **1.6. Cấu trúc báo cáo:**
  - Tóm tắt ngắn gọn nội dung của 5 chương.

---

### CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ CÁC NGHIÊN CỨU LIÊN QUAN (6–7 trang)
- **2.1. Thang đo PHQ-9 và cơ sở sinh lý học của triệu chứng trầm cảm:**
  - Cấu trúc 9 câu hỏi PHQ-9 và 5 mức độ nghiêm trọng lâm sàng chuẩn (Hình 2.1).
  - Các cơ chế y sinh học liên kết sinh lý ngoại vi với trầm cảm (Hình 2.2): phản ứng viêm mạn tính (Albumin, HbA1c), chuyển hóa lipid (BMI, vòng eo, Cholesterol, HDL), huyết động học và thần kinh tự chủ (huyết áp, nhịp tim), và nhịp sinh học (giấc ngủ, vận động).
- **2.2. Đặc thù và thách thức của dữ liệu sức khỏe dạng bảng:**
  - Mất cân bằng lớp cực đoan ($\text{IR} = 68.5:1$) và rủi ro tối ưu hóa sai lệch trên dữ liệu cộng đồng (Hình 2.3a).
  - Chồng lấn phân phối sinh lý học (High Feature Overlap) và diện tích lỗi Bayes tối thiểu (Hình 2.3b).
- **2.3. Các trường phái mô hình hóa trong bài toán phân loại đa lớp có thứ bậc:**
  - Sơ đồ so sánh tổng thể 3 trường phái (Hình 2.4).
  - *Phân loại phẳng (Flat):* Bỏ qua quan hệ khoảng cách giữa các lớp; dễ bị đẩy ép ranh giới.
  - *Phân loại phân cấp (Hierarchical):* Cơ chế 2 giai đoạn (Sàng lọc sơ bộ $\to$ Phân loại chi tiết); phân tích sự khác biệt giữa Định tuyến cứng (nguy cơ lan truyền sai số) và Định tuyến mềm (bảo toàn phân phối xác suất toàn phần).
  - *Lý thuyết quyết định nhạy chi phí Bayes:* Tối thiểu hóa rủi ro kỳ vọng theo ma trận chi phí phạt lỗi nặng.
  - *Phân loại thứ bậc (Ordinal):* Hồi quy Logistic thứ bậc (Proportional Odds) và Phân loại nhị phân tích lũy (Cumulative Binary); phân tích nguy cơ vi phạm tính đơn điệu xác suất.
- **2.4. Tổng quan các nghiên cứu liên quan:**
  - Khảo sát các nghiên cứu tiền nhiệm trên NHANES (Bảng 2.1).
  - Xác lập khoảng trống học thuật: Đa số chỉ làm bài toán nhị phân $T_2$; chưa có nghiên cứu nào khảo sát có hệ thống về phân loại phân cấp mềm nhạy chi phí và kiểm định trôi phân phối theo thời gian.
- **2.5. Tóm tắt chương:**
  - Chuyển tiếp tự nhiên sang thiết kế phương pháp ở Chương 3.

---

### CHƯƠNG 3. DỮ LIỆU VÀ PHƯƠNG PHÁP NGHIÊN CỨU (10–12 trang)
- **3.1. Tổng quan quy trình nghiên cứu DepSev-ML:**
  - Kiến trúc 7 module khoa học khép kín (Hình 3.1).
- **3.2. Bộ dữ liệu NHANES và Quy trình tiền xử lý in-fold:**
  - Tập dữ liệu phát triển Legacy (NHANES 2013–2014, $N=5.054$: $4.045$ Development, $1.009$ Held-out lịch sử, Bảng 3.1).
  - Tập dữ liệu đa chu kỳ chính thức (NHANES 2011–2016, $N=14.581$, Bảng 3.2).
  - Quy trình tiền xử lý, hài hòa hóa và cô lập in-fold bảo toàn nguyên tắc Zero Data Leakage (Hình 3.2).
- **3.3. Xây dựng không gian đặc trưng lũy tiến ($F_1 \to F_4$):**
  - Cấu trúc 4 nhóm đặc trưng lũy tiến: $F_1$ (10 biến sinh tồn/xét nghiệm máu) $\to$ $F_2$ (19 biến + bệnh mạn tính) $\to$ $F_3$ (29 biến + lối sống/giấc ngủ) $\to$ $F_4$ (34 biến + nhân khẩu/kinh tế) (Hình 3.3).
  - Bảng tổng hợp các nhóm đặc trưng (Bảng 3.3; chi tiết 34 mã biến đưa vào Phụ lục A).
- **3.4. Xây dựng nhãn PHQ-9 và các không gian mục tiêu ($T_2 \to T_5$):**
  - Định nghĩa ánh xạ toán học và phân bố mẫu trên từng không gian.
  - Phân biệt bản chất: $C_2^+$ là "nhóm Nặng theo quy ước nghiên cứu" ($s \ge 10$), không đồng nhất với severe lâm sàng ($s \ge 20$).
- **3.5. Thiết kế các kiến trúc phân loại ($E_0 \to E_7$):**
  - Mô hình phẳng cơ sở ($E_0$: Flat Random Forest).
  - Kiến trúc phân cấp cứng ($E_1 \to E_3$) và sơ đồ lan truyền sai số một chiều (Hình 3.4).
  - Kiến trúc phân cấp mềm và lý thuyết quyết định nhạy chi phí Bayes ($E_4 \to E_6$) (Hình 3.5).
  - Mô hình phân loại thứ bậc ($E_{7a}$ OLR và $E_{7b}$ Cumulative Binary) (Hình 3.6).
- **3.6. Thiết kế giao thức kiểm thử thực nghiệm:**
  - Nested 5-Fold Stratified Cross-Validation trên tập Legacy Development ($N=4.045$) (Hình 3.7a).
  - Kiểm định ngoại suy độc lập theo dòng thời gian (Temporal Validation trên NHANES 2015–2016, $N=4.888$) (Hình 3.7b).
- **3.7. Hệ thống độ đo đánh giá đa chiều:**
  - Macro-F1, Balanced Accuracy, Accuracy, Class MAE, Quadratic Weighted Kappa (QWK), và chỉ số an toàn lâm sàng (số ca bỏ sót nặng $C_2^+ \to C_0$).
- **3.8. Tóm tắt chương.**

---

### CHƯƠNG 4. THỰC NGHIỆM VÀ THẢO LUẬN (16–18 trang)
- **4.1. Thiết lập thực nghiệm và kiểm soát tính khả lập:**
  - Cấu hình máy trạm, môi trường phần mềm, cố định hạt giống ngẫu nhiên `random_state = 42`.
- **4.2. Baseline và Feature Ablation ($F_1 \to F_4$ trên $T_5$):**
  - *Động cơ:* Trả lời RQ1 – khảo sát đóng góp thực tế của từng nhóm biến số.
  - *Kết quả:* Bảng 4.1 và Hình 4.1.
  - *Phân tích:* Nhóm tiền sử bệnh lý ($F_2$) tạo bước tăng vọt lớn nhất (+12.6% đến +23.1% Macro-F1); $F_4$ nâng XGBoost lên Macro-F1 = 0.2707.
- **4.3. Tinh chỉnh siêu tham số và khảo sát mô hình kết hợp (Ensemble):**
  - Kết quả Nested CV Outer Macro-F1 (RF/F4 dẫn đầu với 0.2755 $\pm$ 0.0270).
  - Khóa cấu hình Frozen Random Forest Spec.
  - So sánh Ensemble (Soft Voting và Stacking Classifier không cải thiện có ý nghĩa so với mô hình RF đơn nhất, Hình 4.2).
- **4.4. Ảnh hưởng của độ chi tiết nhãn mục tiêu ($T_2 \to T_5$):**
  - *Động cơ:* Trả lời RQ2 – lượng hóa quy luật suy giảm hiệu năng khi tăng số lớp.
  - *Kết quả:* Bảng 4.2, Hình 4.3 và Hình 4.4.
  - *Phát hiện then chốt:* Bước sụt giảm lớn nhất xảy ra ngay từ $T_2$ sang $T_3$ (Macro-F1 giảm từ 0.6639 xuống 0.4354, -34.4%) do lớp nhẹ $C_1$ bị tách khỏi $C_0$ và bị nhận diện kém (Recall $C_1 \approx 16.3\%$).
- **4.5. Phân tích nguyên nhân suy thoái: Chồng lấn phân phối sinh lý:**
  - Ma trận tương quan đặc trưng (Hình 4.5): Không có biến nào có $|r| > 0.25$ với điểm PHQ-9.
  - Biểu đồ hộp Boxplots (Hình 4.6): Phân bố sinh lý giữa $C_0$ và $C_1$ trùng khít nhau.
  - Chiếu không gian 2D PCA Biplot (Hình 4.7): Các mẫu $C_0$ và $C_1$ hoàn toàn bao phủ lẫn nhau trong cùng một vùng không gian mật độ cao, khẳng định rào cản thông tin nội tại của dữ liệu.
- **4.6. Thực nghiệm mở rộng dữ liệu đa chu kỳ NHANES:**
  - *Động cơ:* Trả lời RQ3 – kiểm tra liệu tăng cỡ mẫu gấp 3.6 lần ($N=14.581$) có phá vỡ được trần hiệu năng hay không.
  - *Kết quả & Thảo luận:* Hình 4.8. Cỡ mẫu lớn chỉ ổn định nhẹ Balanced Accuracy (0.2841 lên 0.3076) nhưng không cải thiện Macro-F1 nhất quán, chứng minh giới hạn trần thông tin của các chỉ số sinh lý ngoại vi.
- **4.7. Nghiên cứu mở rộng trên không gian 3 nhóm ($T_3$): Khảo sát $E_0 \to E_7$:**
  - *Động cơ:* Trả lời RQ4 & RQ5 – đánh giá các kiến trúc phân cấp và thứ bậc trên nhiệm vụ 3 nhóm lâm sàng thực tế.
  - *Phân cấp cứng ($E_1 \to E_3$):* Bảng 4.3; vạch trần điểm nghẽn lan truyền sai số một chiều khiến $E_1$ bỏ sót 120 ca $C_2^+$ ngay tại Stage 1.
  - *Phân cấp mềm & Nhạy chi phí ($E_4 \to E_6$):* Bảng 4.4; cơ chế mềm giúp giải tỏa điểm nghẽn và cho phép dịch chuyển ngưỡng theo ma trận phạt.
  - *Phân loại thứ bậc ($E_{7a}, E_{7b}$):* Bảng 4.5; Balanced OLR ($E_{7a}$) đạt Balanced Accuracy cao nhất toàn nghiên cứu (0.5151); CBC ($E_{7b}$) phát hiện 0.32% trường hợp vi phạm đơn điệu xác suất (Hình 4.9).
  - *Bảng tổng hợp Master Comparison:* Bảng 4.6 (Phần A: Hiệu năng tổng thể; Phần B: An toàn lâm sàng).
  - *Trực quan hóa đối sánh:* Hình 4.10 (E0--E6), Hình 4.11 (Macro-F1 vs Class MAE), Hình 4.12 (Lỗi nhảy 2 bước), Hình 4.13 & 4.14 (Ma trận nhầm lẫn 6 mô hình tiêu biểu).
- **4.8. Phân tích sự đánh đổi lâm sàng và ma trận chi phí:**
  - Đường cong đánh đổi giữa Bỏ sót ca nặng và Báo động nhầm (Hình 4.15).
  - Định lượng: Giảm 74.2% số ca bỏ sót nặng (từ 120 ca xuống 31 ca) đổi lại bằng việc tăng báo động giả từ 111 ca lên 1.343 ca.
- **4.9. Kiểm định độc lập theo dòng thời gian (Temporal Validation):**
  - Đánh giá trên tập tương lai NHANES 2015–2016 ($N=4.888$, Bảng 4.7).
  - Khẳng định độ bền vững vượt trội của Balanced OLR ($E_{7a}$ đạt BalAcc = 0.5073, suy giảm chỉ -1.5%).
- **4.10. Thảo luận tổng hợp các phát hiện chính:**
  - Trả lời tổng hợp 5 câu hỏi nghiên cứu RQ1–RQ5.
- **4.11. Tóm tắt chương.**

---

### CHƯƠNG 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (3–4 trang)
- **5.1. Kết luận khoa học:**
  - Tóm tắt các kết quả thực nghiệm then chốt trả lời trực diện RQ1–RQ5.
- **5.2. Đóng góp của đề tài:**
  - Về phương pháp luận: Quy trình chuẩn mực, Zero Data Leakage, Nested CV.
  - Về thực nghiệm: Bằng chứng về trần hiệu năng và sự chồng lấn sinh lý học.
  - Về ứng dụng: Lượng hóa sự đánh đổi lâm sàng theo ma trận chi phí Bayes.
- **5.3. Những hạn chế của nghiên cứu:**
  - Dữ liệu khảo sát cắt ngang (thiếu chiều thời gian dọc).
  - Tín hiệu sinh lý ngoại vi gián tiếp (không phản ánh trực tiếp hoạt động não bộ).
  - Tính chủ quan của thang đo tự điền PHQ-9.
- **5.4. Hướng phát triển trong tương lai:**
  - Dữ liệu đa phương thức (giọng nói, video, văn bản).
  - Dữ liệu cảm biến đeo liên tục (PPG, HRV, giấc ngủ).
  - Mạng nơ-ron học thứ bậc sâu và công cụ giải thích quyết định XAI.

