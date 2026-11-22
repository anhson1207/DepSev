# BÁO CÁO KIỂM ĐỊNH VĂN PHONG HỌC THUẬT (V7)
*(Academic Writing and Tone Audit)*

**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản:** V7 (Hoàn thiện biên tập văn phong, tinh gọn nội dung và kiểm định khoa học)  
**Thời gian thực hiện:** Tháng 10/2026  

---

## I. TỔNG QUAN VỀ ĐỢT BIÊN TẬP VĂN PHONG V7

Phiên bản V6 tuy đã cấu trúc hóa được tiến trình nghiên cứu từ bài toán 5 mức ($T_5$) sang bài toán phân cấp 3 mức ($T_3$), nhưng vẫn còn nhiều đoạn văn mang văn phong máy móc, sáo rỗng, lạm dụng từ ngữ đao to búa lớn (như *"cứu cánh y tế"*, *"đột phá toàn diện"*, *"bảo vệ tính mạng người bệnh"*), các cách dịch thuật ngữ thô kệch (*"Rừng ngẫu nhiên"*, *"Cây quyết định"*, *"Hồi quy Logistic"*, *"bệnh nhân nặng"*), và các đoạn tóm tắt chương lặp đi lặp lại.

Trong V7, toàn bộ báo cáo đã được biên tập lại với **giọng văn của một sinh viên / nhóm nghiên cứu Khoa học Máy tính (Computer Science)**:
- Khiêm tốn, khách quan, chính xác, dựa hoàn toàn trên bằng chứng số liệu thực nghiệm.
- Bỏ hết các khẩu hiệu, lời tự khen, từ ngữ thương mại hóa hoặc bi kịch hóa sai số y tế.
- Giữ nguyên tên thuật ngữ tiếng Anh chuẩn của các thuật toán ML (`Random Forest`, `Decision Tree`, `Logistic Regression`, `XGBoost`).
- Sử dụng tiếng Việt tự nhiên cho các khái niệm dữ liệu (`tập huấn luyện`, `tập kiểm tra`, `tập kiểm định ngoài mẫu`, `nhãn dự đoán`, `đối tượng khảo sát`).

---

## II. DANH SÁCH CÁC MỤC ĐÃ ĐƯỢC VIẾT LẠI TOÀN BỘ

1. **Lời cam đoan, Lời cảm ơn và Tóm tắt đề tài (Preamble & Abstract)**:
   - Loại bỏ các câu chữ sáo rỗng, viết lại tóm tắt cô đọng, nêu đúng vấn đề thực tế: dữ liệu sinh lý ngoại vi dạng bảng có độ đặc hiệu thấp, dẫn đến sự chồng lấn phân phối giữa các mức độ nhẹ và bình thường.
2. **Chương 1 (Mở đầu)**:
   - Viết lại toàn bộ mục 1.1 (Bối cảnh và Động lực), 1.2 (Mục tiêu nghiên cứu) và 1.3 (Phát biểu bài toán).
   - Làm rõ ngay từ đầu hai giai đoạn nghiên cứu: Giai đoạn 1 ($T_5$) và Giai đoạn 2 ($T_3$).
   - Nhấn mạnh $T_3$ là một nhiệm vụ được tái cấu trúc từ quan sát thực nghiệm của $T_5$, không phải là kết quả cải thiện trực tiếp trên cùng không gian 5 lớp.
3. **Chương 2 (Cơ sở lý thuyết và Thuật toán)**:
   - Cắt bỏ các tiểu mục lý thuyết dài dòng của Logistic Regression, Decision Tree và XGBoost. Thay vào đó chỉ giới thiệu ngắn gọn rằng đây là các thuật toán cơ sở để đối chuẩn.
   - Tập trung đi sâu vào giải thích thuật toán **Random Forest**: từ nguyên lý Decision Tree đơn lẻ, Bootstrap Sampling, ngẫu nhiên hóa không gian đặc trưng, cơ chế Soft Voting đến việc giảm phương sai mà không làm tăng độ chệch.
   - Thống nhất ký hiệu ma trận chi phí $\mathbf{C}(i, j)$ với $i$ là nhãn thực tế và $j$ là nhãn dự đoán.
   - Bổ sung trích dẫn đầy đủ cho Bảng 2.1 các công trình nghiên cứu liên quan.
4. **Chương 3 (Phương pháp nghiên cứu và Thiết kế thực nghiệm)**:
   - Làm rõ quy trình huấn luyện in-fold: Stage 1 huấn luyện trên toàn bộ tập train của fold ($N_{\text{train}} \approx 3.236$), Stage 2 huấn luyện nghiêm ngặt trên các mẫu có nhãn thật $\text{PHQ-9} \ge 5$ trong tập train ($N_{\text{train, S2}} \approx 823$); các con số $N=4.045$ và $N=1.029$ là quy mô toàn bộ tập dữ liệu (Total Cohort).
   - Xóa bỏ hình ảnh bị trùng lặp ở Mục 3.3.
5. **Chương 4 (Kết quả thực nghiệm và Thảo luận)**:
   - Bổ sung biểu đồ nhiệt Confusion Matrix 5x5 của $T_5$ (Hình 4.5) để minh họa trực quan sự nhầm lẫn giữa các mức độ.
   - Sửa đổi Hình 4.8 tại Mục 4.5: thay bằng biểu đồ so sánh các nhánh dữ liệu $T_2 \to T_5$ (`multicycle_cohort_comparison.png`), chuyển biểu đồ kiểm định thời gian về đúng Mục 4.8.
   - Chuẩn hóa toàn bộ số liệu: số ca nặng $C_2^+ \to C_0$ của $E_2$ là **98 ca**, sửa số liệu cột $C_0 \to C_2^+$ trong Bảng 4.6 Phần B cho khớp chính xác với Confusion Matrix gộp.
   - Tiết chế các kết luận: làm rõ PCA chỉ là phép chiếu tuyến tính 2D không gian đặc trưng, correlation không đồng nghĩa với causation.
6. **Chương 5 (Kết luận và Hướng phát triển)**:
   - Rút gọn từ một bản kết luận dài dòng xuống 3 trang súc tích.
   - Trả lời trực diện 5 câu hỏi nghiên cứu, tổng kết 4 mô hình tiêu biểu, chỉ ra 3 hạn chế khách quan của dữ liệu và 3 hướng mở rộng thực tế.
7. **Phụ lục C (Appendix C)**:
   - Sửa dứt điểm lỗi float của Bảng C.1 bằng việc cấu hình `\suppressfloats[t]` trong lớp tài liệu, đưa bảng về đúng vị trí dưới tiêu đề chương.

---

## III. BẢNG TỔNG HỢP CÁC CỤM TỪ ĐÃ LOẠI BỎ VÀ THAY THẾ

| STT | Cụm từ máy móc / Phóng đại (V6) | Cụm từ khoa học thay thế (V7) | Văn cảnh sử dụng |
| :---: | :--- | :--- | :--- |
| 1 | Cứu cánh y tế / bảo vệ tính mạng tuyệt đối | Hạn chế rủi ro bỏ sót ca nặng trong sàng lọc | Mục 4.6, Đánh đổi chi phí Bayes |
| 2 | Đột phá toàn diện chưa từng có | Cải thiện hiệu năng nhận diện / tối ưu hóa | Mục 4.5, Kết quả mô hình $E_2$ |
| 3 | Chân lý mặt đất (Ground truth) | Nhãn thực tế / Nhãn tham chiếu | Xuyên suốt các chương |
| 4 | Bệnh nhân nặng lâm sàng | Nhóm Nặng theo quy ước nghiên cứu ($C_2^+$) | Chương 3, 4, 5 |
| 5 | Người hoàn toàn khỏe mạnh | Nhóm triệu chứng tối thiểu ($C_0$) | Chương 3, 4, 5 |
| 6 | Thảm họa y tế khó lường | Gia tăng gánh nặng kiểm tra lâm sàng không cần thiết | Mục 4.6, Phân tích sai số |
| 7 | Bằng chứng đanh thép không thể chối cãi | Kết quả thực nghiệm cho thấy xu hướng rõ rệt | Mục 4.4, Phân tích PCA |
| 8 | Thuật toán rừng ngẫu nhiên kỳ diệu | Thuật toán Random Forest (RF) | Chương 2, 4 |
| 9 | Lấy mẫu hoàn lại tự thân | Bootstrap Sampling | Mục 2.3, Cấu trúc RF |
| 10 | Báo cáo hoàn hảo điểm 10/10 | Báo cáo nghiên cứu học thuật hoàn chỉnh | Toàn văn báo cáo |

