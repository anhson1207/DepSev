# PHẢN BIỆN TOÀN DIỆN BÁO CÁO KHOA HỌC DEPSEV-ML V9
## Báo cáo thẩm định độc lập từ Giảng viên phản biện chuyên ngành Machine Learning

**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Tài liệu thẩm định:** `DepSev_ML_Scientific_Report_V9.pdf` (91 trang)  
**Thời điểm thực hiện:** 31/10/2026  
**Đơn vị thực hiện:** Nhóm nghiên cứu & Biên tập học thuật DepSev-ML  

---

## I. TỔNG QUAN ĐÁNH GIÁ VỀ BẢN BÁO CÁO V9

Bản báo cáo V9 đã có những bước tiến đáng kể so với các phiên bản tiền nhiệm (V1–V8), đặc biệt là việc chuyển đổi thành công khung trần thuật từ "khắc phục thất bại" sang **"khảo sát đa phương pháp luận có hệ thống"** (gồm hai hướng nghiên cứu song song: Khảo sát phân loại đa mức $T_5$ và Khảo sát phân loại phân cấp/thứ bậc $T_3$). Cấu trúc 5 chương đã bám sát chuẩn mực của một Báo cáo Dự án Khoa học (Scientific Project Report) ngành Khoa học Máy tính.

Tuy nhiên, dưới góc độ phản biện chuyên môn độc lập của giảng viên và nhà nghiên cứu Machine Learning, bản V9 vẫn tồn tại nhiều hạt sạn kỹ thuật, lỗi định dạng, chồng chữ và một số điểm chưa chuẩn xác về mặt khoa học. Những hạn chế này làm suy giảm tính thuyết phục học thuật của công trình và cần được khắc phục triệt để trước khi nghiệm thu chính thức.

### Điểm đánh giá tổng quan V9: **7.5 / 10.0**
- **Ưu điểm:** Bố cục 5 chương rõ ràng; hai hướng nghiên cứu $T_5$ và $T_3$ được định hình mạch lạc; số liệu chính của các mô hình cơ sở và mô hình phân cấp đã được đối soát tương đối tốt với kết quả thực nghiệm out-of-fold.
- **Hạn chế lớn:**
  1. *Font chữ thiếu đồng nhất:* Vẫn tồn tại font monospace `LMMono12-Regular` rải rác trên 12 trang PDF, đặc biệt xuất hiện ở các tên siêu tham số ngay trong văn bản thông thường và tiêu đề bảng.
  2. *Lỗi trực quan nghiêm trọng:* Hình 2.1 (trang PDF 28) bị chồng chữ nặng nề giữa nhãn các lớp $T_3$ và chú thích can thiệp y tế; font trong hình là sans-serif lệch chuẩn so với thân bài.
  3. *Lỗi dàn trang và thứ tự float:* Bảng 4.1 (trang PDF 54) bị trôi lên trước tiêu đề Mục 4.2; Bảng 2.1 (trang PDF 39) bị đẩy sang trang riêng với khoảng trắng lớn phía trên và các cột bị bóp méo; Hình 3.4 chứa quá nhiều chữ trong các khối hình chữ nhật.
  4. *Số liệu và diễn giải khoa học:* Cách gọi "Random Forest (Frozen)" tại Bảng 4.2 chưa chính xác về protocol Nested CV; so sánh Feature Ablation trong đoạn văn lấy số liệu của hai thuật toán khác nhau ($0.1900$ của LR và $0.2707$ của XGBoost); nhầm lẫn ký hiệu $M_1/M_2$ giữa mô hình phân cấp và ma trận chi phí Bayes; trùng lặp công thức toán giữa Chương 2 và Chương 3.

---

## II. DANH MỤC CHI TIẾT CÁC VẤN ĐỀ THEO 12 TIÊU CHÍ PHẢN BIỆN

### Tiêu chí 1: Tính đầy đủ của nội dung (Content Completeness)
- **PDF Trang 27–38 (Chương 2, Mục 2.3):**
  - *Vấn đề:* Trình bày về Random Forest đã bao quát nguyên lý, nhưng phần giải thích cơ chế giảm phương sai (variance reduction) của bagging chưa nhấn mạnh vai trò trực tiếp đối với dữ liệu sinh lý tabular có tương quan cao.
  - *Tại sao chưa tốt:* Khiến người đọc có cảm giác phần lý thuyết mang tính sách giáo khoa phổ thông, chưa gắn chặt với đặc thù dữ liệu NHANES.
  - *Cách sửa:* Bổ sung phân tích ngắn gọn: việc lấy mẫu ngẫu nhiên không gian con đặc trưng (`max_features`) giúp làm giảm tương quan giữa các cây khi có nhiều biến số nhân trắc học/sinh hóa cùng phản ánh một trục bệnh lý (như BMI và vòng eo).
  - *Mức độ:* **MAJOR**.

### Tiêu chí 2: Logic giữa các mục (Section Logic)
- **PDF Trang 53–54 (Chương 4, Mục 4.1 & 4.2):**
  - *Vấn đề:* Tiêu đề Mục 4.2 "Đánh giá đóng góp của các nhóm đặc trưng (Feature Ablation)" nằm ở nửa dưới trang 54, trong khi Bảng 4.1 (bảng kết quả của mục này) lại nằm ngay đầu trang 54, xếp ngay sau phần kết thúc của Mục 4.1.
  - *Tại sao chưa tốt:* Vi phạm nghiêm trọng trật tự trình bày khoa học: kết quả thực nghiệm xuất hiện trước khi người đọc được giới thiệu về bối cảnh thí nghiệm và mục tiêu nghiên cứu.
  - *Cách sửa:* Chèn lệnh `\FloatBarrier` trước `\section{Đánh giá đóng góp của các nhóm đặc trưng...}` để buộc Bảng 4.1 và hình minh họa chỉ được xuất hiện sau tiêu đề mục và đoạn văn mở đầu.
  - *Mức độ:* **CRITICAL**.

### Tiêu chí 3: Sự liên kết giữa các chương (Coherence Across Chapters)
- **PDF Trang 25 (Chương 1) và Trang 79 (Chương 5):**
  - *Vấn đề:* Chương 1 và Chương 5 có đoạn nhắc lại các chỉ số cụ thể của từng mô hình $E_0 \to E_7$ một cách chi tiết, làm giảm tính khái quát cấp cao của chương Giới thiệu và chương Kết luận.
  - *Tại sao chưa tốt:* Chương 1 nên tập trung vào bài toán, câu hỏi nghiên cứu (RQ1–RQ5) và đóng góp cốt lõi; Chương 5 nên tổng kết bài học kinh nghiệm và hướng mở, không nên chép lại bảng số liệu của Chương 4.
  - *Cách sửa:* Tinh gọn phần đóng góp ở Chương 1 thành 3 trụ cột phương pháp; rút gọn phần tổng kết ở Chương 5 tập trung vào 4 phát hiện mang tính quy luật (trade-offs) thay vì liệt kê vụn vặt các số thập phân.
  - *Mức độ:* **MAJOR**.

### Tiêu chí 4: Chất lượng giải thích phương pháp (Quality of Method Explanation)
- **PDF Trang 46–47 (Chương 3, Mục 3.4.1):**
  - *Vấn đề:* Mô tả kiến trúc phân cấp cứng (Hard Hierarchical) dùng thuật ngữ $M_1$ và $M_2$ để chỉ mô hình Giai đoạn 1 và Giai đoạn 2.
  - *Tại sao chưa tốt:* Gây xung đột và nhầm lẫn trực tiếp với các ma trận chi phí Bayes $M_0, M_1, M_2, M_3$ được giới thiệu ở Mục 3.4.3 và Phụ lục B.
  - *Cách sửa:* Đổi tên định danh mô hình thành `Stage 1` (hoặc $M_{\text{Stage 1}}$) và `Stage 2` (hoặc $M_{\text{Stage 2}}$); chỉ dành riêng ký hiệu $M_0, M_1, M_2, M_3$ cho ma trận chi phí phạt lỗi.
  - *Mức độ:* **CRITICAL**.

### Tiêu chí 5: Tính chính xác khoa học (Scientific Precision)
- **PDF Trang 54, 63, 78 (Chương 4):**
  - *Vấn đề 1 (Feature Ablation):* Đoạn văn so sánh Macro-F1 tăng từ $0.1900$ trên $F_1$ lên $0.2707$ trên $F_4$. Thực tế, $0.1900$ là kết quả của Logistic Regression trên $F_1$, còn $0.2707$ là kết quả của XGBoost trên $F_4$.
  - *Tại sao chưa tốt:* So sánh giữa hai thuật toán khác nhau rồi quy toàn bộ mức tăng cho việc bổ sung đặc trưng là ngụy biện khoa học.
  - *Cách sửa:* Sửa lại so sánh trong cùng một thuật toán: Với XGBoost, hiệu năng tăng từ $0.2173$ ($F_1$) lên $0.2707$ ($F_4$); với Random Forest tăng từ $0.2052$ lên $0.2232$; với Logistic Regression tăng từ $0.1900$ lên $0.2152$.
  - *Mức độ:* **CRITICAL**.
  - *Vấn đề 2 (Khẳng định về PCA):* Báo cáo khẳng định 2 chiều PCA chỉ giải thích 21.1% phương sai là "bằng chứng chứng minh giới hạn hiệu năng không thể vượt qua của dữ liệu sinh lý".
  - *Tại sao chưa tốt:* PCA là phép chiếu tuyến tính 2 chiều, không đại diện cho toàn bộ thông tin phi tuyến trong không gian 34 chiều; không thể lấy PCA 2D để tuyên bố trần hiệu năng tuyệt đối của bài toán.
  - *Cách sửa:* Diễn đạt thận trọng: Phép chiếu PCA 2 chiều phản ánh sự chồng lấn hình học đáng kể giữa các phân phối mức độ trầm cảm trên các trục biến thiên chính, phần nào giải thích lý do các thuật toán tuyến tính phẳng gặp khó khăn trong việc tách biệt ranh giới lớp.
  - *Mức độ:* **MAJOR**.

### Tiêu chí 6: Tính nhất quán với source code (Consistency with Source Code)
- **PDF Trang 56 (Chương 4, Bảng 4.2):**
  - *Vấn đề:* Bảng 4.2 định danh dòng kết quả Macro-F1 = $0.2755 \pm 0.0270$ là `Random Forest (Frozen)`.
  - *Tại sao chưa tốt:* Khi đối soát file `results/tables/nested_cv_baseline_comparison.csv`, cấu hình này thực chất được tối ưu hóa siêu tham số độc lập trong từng outer fold (Fold 4 chọn `max_features = 0.5`, các fold khác chọn `sqrt`). Do đó, gọi là "Frozen" là sai quy trình Nested CV.
  - *Cách sửa:* Đổi tên nhãn trong Bảng 4.2 thành `Random Forest (Tuned)`. Giải thích rõ: đây là kết quả của mô hình Random Forest tinh chỉnh qua Nested CV; từ kết quả này, bộ siêu tham số phổ biến nhất mới được cố định ("Frozen") làm cấu hình đối chuẩn cho các nghiên cứu Ensemble và Target Granularity tiếp theo.
  - *Mức độ:* **CRITICAL**.

### Tiêu chí 7: Mức độ thuyết phục của kết quả & Sự đánh đổi (Convincingness & Trade-offs)
- **PDF Trang 68–71 (Chương 4, Mục 4.8):**
  - *Vấn đề:* Khi phân tích các ma trận chi phí phạt cao ($M_2, M_3$), văn bản gọi việc số ca báo động nhầm ($C_0 \to \{C_1, C_2^+\}$) tăng vọt từ 538 ca lên 1.906 ca và 2.294 ca là "sự gia tăng có kiểm soát".
  - *Tại sao chưa tốt:* Tăng từ 538 lên 2.294 ca báo động nhầm trên tổng số 3.016 người bình thường (tỷ lệ báo động sai > 76%) là sự đánh đổi rất nặng nề, không thể dùng từ nhẹ nhàng "có kiểm soát" nếu không có ngữ cảnh chấp nhận lâm sàng cụ thể.
  - *Cách sửa:* Diễn đạt khách quan: Đây là sự đánh đổi trực tiếp và rõ ràng giữa độ nhạy bảo vệ ca nặng và tỷ lệ dương tính giả (false positive surge). Mô hình $M_1$ mang lại điểm cân bằng thực tiễn hợp lý nhất khi giảm gần một nửa ca bỏ sót ($48.3\%$) với mức tăng báo động nhầm vừa phải, trong khi $M_2$ và $M_3$ đẩy hệ thống về trạng thái cực đoan ưu tiên tầm soát.
  - *Mức độ:* **MAJOR**.

### Tiêu chí 8: Văn phong tự nhiên và khiêm tốn (Academic Tone & Modesty)
- **Toàn bộ báo cáo:**
  - *Vấn đề:* Một số đoạn văn còn sử dụng các tính từ khoa trương kiểu AI ("vượt trội một cách toàn diện", "giải pháp tối hậu", "khẳng định đanh thép", "cứu nguy cho mô hình phẳng").
  - *Tại sao chưa tốt:* Văn phong thiếu tính khiêm tốn khoa học, dễ gây ác cảm với các thầy cô hội đồng chấm khóa luận / dự án.
  - *Cách sửa:* Viết lại với giọng văn bình tĩnh, khách quan của một sinh viên CS ("em"), diễn đạt theo dữ liệu thực chứng: "kết quả thực nghiệm cho thấy...", "trong phạm vi khảo sát của đề tài...", "sự cải thiện này đi kèm với sự đánh đổi về...".
  - *Mức độ:* **MAJOR**.

### Tiêu chí 9: Độ cần thiết và trùng lặp của công thức (Formula Redundancy)
- **PDF Trang 37–38 (Chương 2) và Trang 48–49 (Chương 3):**
  - *Vấn đề:* Hai khối công thức toán học bị lặp lại gần như nguyên vẹn:
    1. Công thức xác suất toàn phần định tuyến mềm Soft Hierarchical $\hat{P}(C_0), \hat{P}(C_1), \hat{P}(C_2^+)$ (xuất hiện ở cả Mục 2.3.4 và Mục 3.4.2).
    2. Công thức hàm rủi ro Bayes $R(j \mid \mathbf{x}) = \sum \hat{P}(C_i) \mathbf{C}(i, j)$ và luật quyết định tối thiểu rủi ro (xuất hiện ở cả Mục 2.3.5 và Mục 3.4.3).
  - *Tại sao chưa tốt:* Làm loãng nội dung, lãng phí không gian trang giấy, tạo cảm giác cắt dán cơ học giữa các chương.
  - *Cách sửa:* Tại Chương 2, chỉ giải thích nguyên lý toán học tổng quát một cách súc tích. Dành toàn bộ việc trình bày công thức tường minh và ký hiệu thực nghiệm cho Chương 3 (nơi thiết kế kiến trúc hệ thống), đồng thời dẫn chiếu chéo giữa hai chương.
  - *Mức độ:* **MAJOR**.

### Tiêu chí 10: Chất lượng hình và bảng (Figure & Table Quality)
- **PDF Trang 28 (Hình 2.1):**
  - *Vấn đề:* Nhãn $T_3$ cho $C_0$ và $C_1$ ở phía trên trục bị chồng lên nhau; các chú thích can thiệp y tế ("Tư vấn lối sống", "Đánh giá chuyên khoa", v.v.) bên dưới trục điểm số bị dính sát vào nhau và chồng chữ khó đọc; font chữ trong hình là font sans-serif không đồng nhất với tài liệu.
  - *Tại sao chưa tốt:* Đây là hình minh họa phương pháp cốt lõi của đề tài ngay đầu Chương 2. Lỗi chồng chữ tại đây tạo ấn tượng rất xấu về chất lượng biên tập.
  - *Cách sửa:* Thiết kế lại Hình 2.1 từ đầu theo cấu trúc hai phần riêng biệt: Phần A biểu diễn 5 phân đoạn $T_5$ ($C_0 \to C_4$); Phần B biểu diễn phép ánh xạ sang 3 nhóm $T_3$ ($C_0, C_1, C_2^+$) bằng các ngoặc khối thoáng đãng; loại bỏ hoàn toàn các câu khuyến nghị can thiệp y tế thừa; chuyển toàn bộ font chữ sang Times New Roman.
  - *Mức độ:* **CRITICAL**.
- **PDF Trang 39 (Bảng 2.1):**
  - *Vấn đề:* Bảng nghiên cứu liên quan bị đẩy sang trang 39 đứng trơ trọi một mình, phía dưới còn khoảng trống mênh mông; các cột nội dung bị bóp hẹp khiến chữ ngắt dòng liên tục, khó đọc.
  - *Tại sao chưa tốt:* Phân bố layout xấu, bảng thiếu tính chuyên nghiệp.
  - *Cách sửa:* Tinh gọn nội dung các ô trong bảng theo 4 cột chuẩn học thuật: Nghiên cứu, Dữ liệu, Phương pháp, Kết quả & Giới hạn; căn chỉnh lại độ rộng cột và kiểm soát float để bảng gắn kết tự nhiên với văn bản của Mục 2.4.
  - *Mức độ:* **MAJOR**.
- **PDF Trang 47 (Hình 3.4):**
  - *Vấn đề:* Khối sơ đồ định tuyến cứng nhồi nhét quá nhiều chữ giải thích (hyperparameters, mô tả tập train fold, ký hiệu M1/M2); mũi tên và chữ bị sát nhau.
  - *Tại sao chưa tốt:* Sơ đồ kiến trúc cần trực quan, súc tích; không nên biến hình vẽ thành đoạn văn thu nhỏ.
  - *Cách sửa:* Vẽ lại sơ đồ theo luồng tinh giản: Đối tượng đầu vào $\to$ Giai đoạn 1 (Sàng lọc triệu chứng) $\to$ Rẽ nhánh [$C_0$ dừng / Chuyển Giai đoạn 2] $\to$ Giai đoạn 2 (Phân tầng nguy cơ) $\to$ Xuất nhãn [$C_1$ / $C_2^+$]. Chi tiết kỹ thuật đưa vào thân bài.
  - *Mức độ:* **MAJOR**.

### Tiêu chí 11: Thống nhất Font chữ và loại bỏ Monospace (Font Consistency)
- **PDF Trang 31, 34, 35, 43, 46, 51, 53, 56, 57, 59, 86, 87:**
  - *Vấn đề:* Xuất hiện font `LMMono12-Regular` (monospace) trong văn bản thông thường cho các siêu tham số như `\texttt{n_estimators = 150}`, `\texttt{max_depth = 8}`, `\texttt{max_features = 0.5}`, tên thư viện `\texttt{scikit-learn}`, `\texttt{xgboost}` và trong các bảng số liệu.
  - *Tại sao chưa tốt:* Tạo sự chắp vá về mặt thị giác, phá vỡ tính trang trọng của font Times New Roman trong báo cáo học thuật.
  - *Cách sửa:* Thay thế toàn bộ `\texttt{...}` bằng font chính `\textnormal{...}` (hoặc in đậm `\textbf{...}` khi cần nhấn mạnh tên siêu tham số) kèm theo việc escape dấu gạch dưới đúng cú pháp (`n\_estimators = 150`, `max\_depth = 8`, `max\_features = 0.5`). Giữ Times New Roman đồng nhất 100% trong toàn bộ tài liệu.
  - *Mức độ:* **CRITICAL**.

### Tiêu chí 12: Khả năng bảo vệ trước Hội đồng chấm (Defensibility)
- **PDF Trang 24–25 (Chương 1) và Trang 80–82 (Chương 5):**
  - *Vấn đề:* Một số đoạn kết luận có xu hướng ngụ ý rằng mô hình DepSev-ML "sẵn sàng triển khai ứng dụng thực tế trong sàng lọc y tế cộng đồng".
  - *Tại sao chưa tốt:* Giảng viên phản biện chuyên ngành y - tin học sẽ chất vấn gay gắt vì mô hình chỉ được thử nghiệm trên dữ liệu khảo sát hồi cứu NHANES, chưa qua thử nghiệm lâm sàng tiền cứu (prospective clinical validation), và các chỉ số Macro-F1 còn khiêm tốn ($0.45$--$0.51$).
  - *Cách sửa:* Tuyên bố giới hạn một cách minh bạch: DepSev-ML là một nghiên cứu thực nghiệm học máy nhằm khám phá không gian giải pháp và đặc tính đánh đổi sai số trên dữ liệu bảng sinh lý; đề tài mang ý nghĩa phương pháp luận và tạo tiền đề nghiên cứu, không phải một sản phẩm chẩn đoán y khoa sẵn sàng sử dụng lâm sàng.
  - *Mức độ:* **MAJOR**.

---

## III. BẢNG PHÂN LOẠI MỨC ĐỘ VÀ KẾ HOẠCH HÀNH ĐỘNG CHO BẢN V10

| STT | Vị trí (Trang PDF V9 / Mục) | Nội dung vấn đề | Mức độ nghiêm trọng | Hành động sửa đổi trong V10 | Trạng thái thực hiện |
| :---: | :--- | :--- | :---: | :--- | :---: |
| 1 | **Trang 28 (Hình 2.1)** | Chồng chữ nhãn $T_3$ $C_0, C_1$; chồng chữ khuyến nghị can thiệp; font sans-serif | **CRITICAL** | Thiết kế lại Hình 2.1 từ đầu thành 2 phần A & B; bỏ khuyến nghị y tế; đổi sang Times New Roman | Sẵn sàng sửa |
| 2 | **Trang 31, 34, 35, 43, 46, 51, 53, 56, 57, 86, 87** | Font monospace `LMMono` xuất hiện trong văn bản thường cho tên tham số | **CRITICAL** | Xóa bỏ `\texttt`, dùng `\textnormal` / `\textbf` với font Times New Roman thống nhất | Sẵn sàng sửa |
| 3 | **Trang 54 (Bảng 4.1)** | Bảng 4.1 bị đảo thứ tự, xuất hiện trước tiêu đề Mục 4.2 | **CRITICAL** | Đặt `\FloatBarrier` trước Mục 4.2, bố trí tiêu đề và lời mở đầu trước bảng | Sẵn sàng sửa |
| 4 | **Trang 56 (Bảng 4.2)** | Nhãn `Random Forest (Frozen)` sai bản chất Nested CV tuning | **CRITICAL** | Sửa thành `Random Forest (Tuned)`, giải thích rõ protocol và frozen spec | Sẵn sàng sửa |
| 5 | **Trang 54, 63, 78 (Chương 4)** | So sánh Feature Ablation lấy số liệu khác mô hình (LR $0.1900$ vs XGB $0.2707$) | **CRITICAL** | Sửa lại so sánh nội bộ cùng thuật toán (XGB: $0.2173 \to 0.2707$; RF: $0.2052 \to 0.2232$) | Sẵn sàng sửa |
| 6 | **Trang 46–47, 51, 70 (Chương 3, 4)** | Trùng ký hiệu $M_1/M_2$ cho cả mô hình phân cấp và ma trận chi phí Bayes | **CRITICAL** | Dùng `Stage 1 / Stage 2` cho mô hình, dành riêng $M_0 \dots M_3$ cho ma trận chi phí | Sẵn sàng sửa |
| 7 | **Trang 39 (Bảng 2.1)** | Bảng đứng trơ trọi trang riêng, cột chật, chữ ngắt liên tục | **MAJOR** | Thiết kế lại theo chuẩn học thuật 4 cột gọn gàng, kiểm soát float | Sẵn sàng sửa |
| 8 | **Trang 47 (Hình 3.4)** | Sơ đồ phân cấp nhiều chữ, chật chội, nhồi nhét tham số | **MAJOR** | Tinh gọn sơ đồ thành các khối từ khóa trực quan, loại bỏ chữ thừa | Sẵn sàng sửa |
| 9 | **Trang 37–38 vs 48–49** | Trùng lặp nguyên văn công thức Soft Hierarchical và Bayes Risk ở Chương 2 & 3 | **MAJOR** | Tinh gọn công thức ở Chương 2, giữ công thức chi tiết tại Chương 3 và dẫn chiếu | Sẵn sàng sửa |
| 10 | **Trang 65, 78 (Chương 4)** | Khẳng định PCA 2D chứng minh giới hạn hiệu năng tuyệt đối của bài toán | **MAJOR** | Giảm tông khẳng định: PCA 2D phản ánh sự chồng lấn hình học, không phải trần lý thuyết | Sẵn sàng sửa |
| 11 | **Trang 68–71 (Chương 4)** | Gọi sự tăng vọt báo động nhầm ở $M_2, M_3$ là "có kiểm soát" | **MAJOR** | Phân tích khách quan sự đánh đổi; nhấn mạnh $M_1$ là điểm cân bằng thực tiễn hợp lý nhất | Sẵn sàng sửa |
| 12 | **Toàn bộ báo cáo** | Câu chữ khuôn mẫu, tính từ khoa trương kiểu văn bản AI | **MAJOR** | Viết lại bằng văn phong sinh viên CS khiêm tốn ("em"), lập luận dựa trên bằng chứng | Sẵn sàng sửa |
| 13 | **Trang 24, 82 (Chương 1, 5)** | Tuyên bố khả năng ứng dụng lâm sàng vượt quá phạm vi dữ liệu hồi cứu | **MAJOR** | Bổ sung giới hạn nghiên cứu: mô hình là nguyên mẫu thực nghiệm, chưa thẩm định lâm sàng | Sẵn sàng sửa |
| 14 | **Tất cả các hình vẽ còn lại** | Font chữ sans-serif trong các biểu đồ matplotlib | **MINOR** | Chuẩn hóa font trong script sinh hình sang font Times / Serif đồng bộ | Sẵn sàng sửa |
| 15 | **Trang 83–85 (Tài liệu tham khảo)** | Định dạng trích dẫn và tính xác thực của các bài báo liên quan | **MINOR** | Kiểm tra toàn bộ bibtex keys (Pratt, Patel, Dipnall, Zhang), đảm bảo trích dẫn chính xác | Sẵn sàng sửa |

---

## IV. TỔNG KẾT SỐ LƯỢNG LỖI VÀ CAM KẾT HOÀN THIỆN V10

- **Tổng số lỗi phát hiện:** **15 vấn đề trọng yếu**
  - **Lỗi CRITICAL:** **6 lỗi** (Font monospace trong text; Chữ chồng trong Hình 2.1; Thứ tự float Bảng 4.1; Nhãn Frozen RF Bảng 4.2; So sánh sai mô hình trong Feature Ablation; Trùng ký hiệu $M_1/M_2$).
  - **Lỗi MAJOR:** **7 lỗi** (Layout Bảng 2.1; Sơ đồ Hình 3.4 rườm rà; Trùng lặp công thức Chương 2 & 3; Khẳng định quá mức về PCA; Đánh đổi báo động nhầm chi phí Bayes; Văn phong khoa trương; Khẳng định sẵn sàng lâm sàng).
  - **Lỗi MINOR:** **2 lỗi** (Font biểu đồ matplotlib; Định dạng thư mục tài liệu tham khảo).
- **Mục tiêu phiên bản V10:**
  - Giải quyết dứt điểm **100% các lỗi CRITICAL, MAJOR và MINOR** nêu trên.
  - Bản báo cáo V10 phải đạt điểm tuyệt đối về tính thống nhất hình thức (Times New Roman toàn diện, zero overlapping text, trật tự float chuẩn xác) và tính chính xác khoa học (số liệu minh bạch, lập luận khiêm tốn, bảo toàn nguyên vẹn mã nguồn và dữ liệu thực nghiệm).

