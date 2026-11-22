# PHÂN TÍCH ĐỐI CHIẾU PHONG CÁCH TÀI LIỆU THAM KHẢO VÀ ĐỊNH VỊ DEPSEV-ML V4

## 1. TỔNG QUAN CÁC TÀI LIỆU THAM CHIẾU

Nghiên cứu khảo sát 8 tài liệu báo cáo tốt nghiệp/khóa luận đại học ngành Công nghệ Thông tin - Khoa học Máy tính tại Trường Đại học Công nghệ – ĐHQGHN (VNU-UET) cùng 1 báo cáo nghiên cứu khoa học chuyên sâu về Machine Learning y tế:

1. **Lê Ngọc Nhật Tân (2026, 82 trang):** Quản lý phòng tập gym Xala tích hợp AI (Vision/Pose estimation).
2. **Đoàn Trung Kiên (2026, 107 trang):** Website học tiếng Anh thông minh tích hợp AI (FSRS Spaced Repetition, LLM RAG).
3. **Nguyễn Chí Huy (2026, 137 trang):** Hệ thống nhà thuốc trực tuyến tích hợp AI SmartPharma (Chatbot, Recommendation).
4. **Nguyễn Đức Kiên (2026, 113 trang):** Chợ công nghệ trực tuyến TechXchange tích hợp AI (Search, Recommendation).
5. **Trương Thị Xuân Sang (2026, 137 trang):** Kiểm thử tự động hệ thống ERPNext (Robot Framework, Selenium).
6. **Nguyễn Thu Hà (2026, 119 trang):** Kiểm thử hệ thống sàn nông sản Freso (Test design, Test execution).
7. **Nguyễn Thị Cát Tường (2026, 110 trang):** Hệ thống chứng chỉ số dựa trên NFT và tích hợp AI.
8. **Hoàng Quốc Việt (2026, 107 trang):** Hệ thống giao dịch lợi suất trên Sui Blockchain (Smart contracts, Yield trading).
9. **Tài liệu tham chiếu bổ sung đặc biệt — Nguyễn Đức Tân (2024, 55 trang):** Nghiên cứu phương pháp học liên tục cho theo dõi bệnh tim mạch vành trong hoạt động hàng ngày (Graph Neural Networks, Continuous Learning trên dữ liệu y tế chuỗi thời gian).

---

## 2. BẢNG PHÂN TÍCH SO SÁNH PHONG CÁCH VÀ MỨC ĐỘ PHÙ HỢP VỚI DEPSEV-ML

| Tiêu chí | 8 Khóa luận Công nghệ / Phần mềm (2026) | Nghiên cứu Học máy Tim mạch (Nguyễn Đức Tân 2024) | Định vị Chuẩn tắc cho DepSev-ML V4 |
| :--- | :--- | :--- | :--- |
| **Bản chất đề tài** | Dự án Kỹ thuật Công nghệ / Xây dựng sản phẩm phần mềm (Website, Mobile App, Testing, Blockchain). | Nghiên cứu Khoa học thuần túy (Scientific ML Research) giải quyết bài toán lâm sàng. | **Dự án Khoa học thực nghiệm Machine Learning** (Đánh giá trần hiệu năng, phân cấp và thứ bậc trên dữ liệu y tế). |
| **Độ dài báo cáo** | 82 – 137 trang (rất dày do chứa đặc tả Use Case, thiết kế CSDL, giao diện, bảng ca kiểm thử). | **55 trang** (cô đọng, súc tích, tập trung 100% vào phương pháp và kết quả thực nghiệm). | **~45 – 55 trang** nội dung chính + phụ lục (tinh gọn, loại bỏ mọi râu ria kéo dài trang). |
| **Bố cục chương** | 4–5 chương thiên về SDLC: Giới thiệu $\to$ Cơ sở công nghệ $\to$ Phân tích thiết kế $\to$ Cài đặt thử nghiệm $\to$ Kết luận. | 4–5 chương khoa học: Đặt vấn đề $\to$ Tổng quan lý thuyết $\to$ Phương pháp đề xuất $\to$ Thực nghiệm & Đánh giá $\to$ Kết luận. | **5 chương khoa học chuẩn mực:** 1. Giới thiệu $\to$ 2. Cơ sở lý thuyết $\to$ 3. Dữ liệu & Phương pháp $\to$ 4. Thực nghiệm & Thảo luận $\to$ 5. Kết luận. |
| **Mật độ công thức** | Rất ít (chỉ 2–5 công thức cơ bản cho thuật toán phụ trợ). | Vừa phải (~12–15 công thức trọng tâm, có chú giải biến số rõ ràng). | **10–14 công thức cốt lõi**; loại bỏ triệt để các đạo hàm trung gian, khai triển Taylor, Gini/Entropy phổ thông. |
| **Mật độ hình ảnh** | 25 – 40 hình (chủ yếu là biểu đồ ca sử dụng UML, sơ đồ ERD, ảnh chụp màn hình UI, biểu đồ kiểm thử). | 15 – 20 hình (sơ đồ khối luồng dữ liệu, kiến trúc mạng, biểu đồ đường cong ROC, biểu đồ phân tích lỗi). | **14 – 16 hình vẽ chất lượng cao:** Sơ đồ pipeline, kiến trúc phân cấp/thứ bậc, biểu đồ phân tích lỗi và ma trận nhầm lẫn. |
| **Trang bìa & Danh xưng** | "KHÓA LUẬN TỐT NGHIỆP" | "KHOÁ LUẬN TỐT NGHIỆP CHÍNH QUY" | **BÁO CÁO DỰ ÁN KHOA HỌC** (Chuẩn hóa chính xác loại tài liệu, không dùng danh xưng khóa luận). |
| **Văn phong học thuật** | Tường thuật kỹ thuật triển khai phần mềm ("hệ thống cho phép", "người dùng thực hiện"). | Lập luận khoa học khách quan ("kết quả chỉ ra", "nghiên cứu khảo sát", "đánh đổi giữa độ nhạy và đặc hiệu"). | **Học thuật phản biện khách quan, khiêm tốn khoa học:** Phân biệt rõ quan sát thực nghiệm và giả thuyết y sinh; không dùng từ ngữ phóng đại. |

---

## 3. NHỮNG ĐIỂM ĐÁNG HỌC HỎI TỪ CÁC TÀI LIỆU MẪU

1. **Tính thẩm mỹ và chuẩn mực Typography của VNU-UET:**
   - Căn lề chuẩn A4: Lề trái 3.0 cm (đóng gáy), lề phải 2.0 cm, trên 2.0–2.5 cm, dưới 2.0–2.5 cm.
   - Font chữ chuẩn (Times New Roman / Computer Modern Roman serif), kích thước 12pt, giãn dòng 1.2–1.3x cân đối, không để trang bị quá thưa hoặc quá nghẹt chữ.
   - Đánh số hình và bảng theo chương (Hình 3.1, Bảng 4.2), vị trí Caption chuẩn mực (Caption của Bảng đặt trên, Caption của Hình đặt dưới).

2. **Cách trình bày phần mở đầu (Frontmatter) trang nhã:**
   - Trang bìa có logo vector sắc nét của Trường Đại học Công nghệ.
   - Lời cảm ơn và Lời cam đoan chân thành, nghiêm túc.
   - Mục "Tuyên bố về sự hỗ trợ của AI" (học hỏi từ khóa luận của Nguyễn Chí Huy) giúp minh bạch phương pháp hỗ trợ định dạng/ngôn ngữ trong khi bảo toàn 100% tính nguyên bản của nghiên cứu.
   - Danh mục từ viết tắt được tổ chức thành bảng 3 cột gọn gàng, có gióng hàng thẳng hàng.

3. **Mô hình hóa quy trình xử lý trực quan:**
   - Các khóa luận xuất sắc luôn có ít nhất 1 sơ đồ pipeline tổng quan ở Chương 3 giúp người đọc nắm bắt toàn bộ chu trình xử lý mà không cần đọc hàng trang chữ.
   - Các khối hộp phân biệt màu sắc nhã nhặn, mũi tên luồng dữ liệu rõ ràng, có nhãn diễn giải cụ thể.

4. **Trình bày bảng biểu khoa học (Booktabs styling):**
   - Không sử dụng các đường kẻ dọc (`|`) gây rối mắt.
   - Chỉ sử dụng 3 đường kẻ ngang chính (`\toprule`, `\midrule`, `\bottomrule`) tạo cảm giác thanh thoát, chuyên nghiệp theo chuẩn quốc tế IEEE/ACM.

---

## 4. NHỮNG ĐIỂM CẦN TRÁNH VÀ TUYỆT ĐỐI KHÔNG SAO CHÉP

1. **Không sao chép cấu trúc "Dự án Công nghệ / Phần mềm":**
   - Tuyệt đối không đưa vào báo cáo các nội dung: Đặc tả ca sử dụng (Use Case Specs), Biểu đồ ca sử dụng tổng quát/chi tiết, Biểu đồ lớp (Class Diagram), Biểu đồ quan hệ thực thể (ERD), Thiết kế cơ sở dữ liệu vật lý (MySQL/PostgreSQL tables), Thiết kế giao diện (UI/UX Wireframes/Screenshots), Kịch bản kiểm thử (Test Cases), Kết quả kiểm thử tự động (Selenium/Robot Framework).
   - DepSev-ML là **nghiên cứu thực nghiệm Machine Learning**, do đó toàn bộ nội dung phải xoay quanh: Bài toán $\to$ Đặc trưng y sinh $\to$ Kiến trúc mô hình $\to$ Giao thức kiểm định $\to$ Đánh giá đa chiều $\to$ Đánh đổi lâm sàng.

2. **Tránh bẫy "Viết dài để lấy số trang" (Padding):**
   - Các khóa luận phần mềm thường kéo dài trên 100 trang bằng cách chụp nhiều màn hình và copy code/bảng database.
   - DepSev-ML V4 hướng tới sự sắc sảo, cô đọng: mỗi trang viết đều phải mang hàm lượng thông tin khoa học thực sự.

3. **Tránh văn phong tiếp thị sản phẩm:**
   - Không dùng từ ngữ mang tính thương mại hoặc quảng bá sản phẩm như "hệ thống mang lại trải nghiệm đột phá", "giao diện trực quan thân thiện", "tính năng ưu việt hoàn hảo".
   - Thay vào đó là ngôn ngữ định lượng chính xác: "giảm 74.2% số ca bỏ sót nặng", "đạt Balanced Accuracy = 0.5151", "Macro-F1 duy trì ở mức 0.2630 do chồng lấn phân phối".

