# BẢNG SO SÁNH ĐỐI CHIẾU CHI TIẾT GIỮA V9 VÀ V10 (BEFORE / AFTER COMPARISON)
**Tập tin:** `reports/final_scientific_v10/validation/BEFORE_AFTER_COMPARISON.md`  
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Ngày thực hiện:** 31/10/2026  
**Người thực hiện:** Antigravity Research Team  

---

## 1. TỔNG HỢP SO SÁNH TỔNG QUAN

| Tiêu chí đánh giá | Bản báo cáo V9 (Trước chỉnh sửa) | Bản báo cáo V10 (Sau khi hoàn thiện) | Minh chứng / Tệp ảnh kiểm tra |
| :--- | :--- | :--- | :--- |
| **Tổng số trang** | 91 trang | 92 trang | Phân bổ trang thoáng, các bảng lớn không bị tràn lề |
| **Font chữ văn bản** | Chắp vá; sử dụng font `\texttt` (Latin Modern Monospace) trên 12 trang văn bản | $100\%$ đồng nhất **Times New Roman** (Regular, Bold, Italic, Bold-Italic). Triệt tiêu $100\%$ font Monospace | `reports/final_scientific_v10/validation/FONT_REPORT.md` |
| **Font chữ hình ảnh** | Font sans-serif mặc định, ký hiệu toán học khác kiểu chữ với tài liệu | $100\%$ đồ thị dùng font **Times New Roman** và bộ ký hiệu toán **STIX**, tương thích tuyệt đối với TeX | Toàn bộ 19 tệp hình trong `source/figures/` |
| **Hình 2.1 (Phân loại PHQ-9)** | Chữ dính vào nhau; chứa các ô văn bản chỉ định thuốc, khám bệnh mang tính chất tờ rơi y tế | Tách 2 phần rõ rệt (Part A: 5 lớp $T_5$ với cỡ mẫu $n$ và $\%$; Part B: 3 mức lâm sàng $T_3$ có ngoặc gộp). Bỏ $100\%$ chỉ định thuốc | `previews/page_28_fig2_phq9_tiers.png` |
| **Hình 3.4 & 3.5 (Sơ đồ phân cấp)** | Trùng lặp ký hiệu "M1/M2" với ma trận tổn thất Bayes $M_0 \dots M_3$; nhiều chữ rườm rà | Đổi thành "Giai đoạn 1 (Stage 1)" và "Giai đoạn 2 (Stage 2)". Nội dung trong hộp súc tích, chữ $\ge 10\text{ pt}$ | `previews/page_47_fig3_4_hard_hierarchical.png` |
| **Bảng 2.1 (Ánh xạ nhãn)** | Chật chội 6 cột, nhồi nhét nội dung điều trị y khoa | Rút gọn thành **4 cột chuẩn mực**, định dạng `booktabs` cao cấp, thoáng đãng | `previews/page_38_table2_1.png` |
| **Bảng 4.1 (Feature Ablation)** | **Lỗi đảo vị trí nổi (Float Inversion):** Bảng trôi lên đỉnh đầu trang 54, nằm trước cả tiêu đề Mục 4.2 | **Khắc phục triệt để:** Tiêu đề Mục 4.2 và Tiểu mục 4.2.1 xuất hiện ở đầu trang, Bảng 4.1 xuất hiện phía dưới | `previews/page_54_table4_1_and_heading4_2.png` |
| **Bảng 4.2 (Nested CV Baseline)** | Dòng 14 ghi `Random Forest (Frozen) = 0.2755`, mâu thuẫn với bản chất Nested CV | Dòng 14 đổi thành `Random Forest (Tuned) = 0.2755` kèm chú thích chân trang số 2 phân biệt với Frozen Spec ($0.2741$) | `previews/page_56_table4_2_nested_cv.png` |
| **Phân tích Feature Ablation** | So sánh chéo khập khiễng (so sánh LR ở $F_1$ với XGBoost ở $F_4$) | Đánh giá mức tăng trưởng nghiêm ngặt *trong cùng một họ mô hình* (within-model analysis) | Mục 4.2, trang 54--55 |
| **Số liệu ca nặng bỏ sót** | Nêu số liệu không kiểm chứng ("giảm 74.2%, còn 31 ca") | Đối chiếu chính xác ma trận nhầm lẫn Out-of-Fold: $E_1$ sót 120 ca, $E_2$ sót 98 ca, $E_{6b}$ còn 62 ca (giảm 48.33%) | Mục 4.5.3, trang 64--65 |
| **Trùng lặp công thức toán** | Xuất hiện hai lần công thức Soft Routing và Bayes Risk ở cả Chương 2 và Chương 3 | Chương 2 chỉ phân tích khái niệm; Chương 3 trình bày khối phương trình toán học chi tiết | Chương 2 và Chương 3 |
| **Lỗi tham chiếu chéo** | Xuất hiện liên kết hỏng `??` tại Trang 37 | Khắc phục liên kết `\ref{subsec:soft-cost-hierarchical}`. Số lỗi `??` là 0 | Quét toàn bộ 92 trang PDF |
| **Văn phong học thuật** | Lạm dụng từ ngữ AI khoa trương ("đột phá", "chiến thắng áp đảo"); kể câu chuyện "chữa lỗi" tiêu cực | Văn phong học thuật khách quan, khiêm tốn của sinh viên nghiên cứu ("kết quả cho thấy"); so sánh đa chiều các kiến trúc | Toàn bộ 5 chương |

---

## 2. ĐỐI CHIẾU CHI TIẾT TRỰC QUAN QUA CÁC HÌNH ẢNH PREVIEW

Trong thư mục `reports/final_scientific_v10/validation/previews/`, các tệp hình ảnh trích xuất trực tiếp từ các trang quan trọng của tài liệu PDF hoàn thiện chứng minh cụ thể những cải tiến:

1. **`previews/page_28_fig2_phq9_tiers.png`:**
   - Minh chứng cho Hình 2.1 được đặt trang nhã ngay dưới đoạn văn dẫn nhập của Mục 2.3.
   - Phần A (Part A) hiển thị 5 cột màu sắc pastel hài hòa với các khoảng điểm 0--4, 5--9, 10--14, 15--19, 20--27 kèm tỷ lệ mẫu chuẩn xác từ NHANES ($77.54\%$, $13.81\%$, $5.23\%$, $2.38\%$, $1.05\%$).
   - Phần B (Part B) hiển thị 3 nhánh lâm sàng rõ ràng, không còn bất kỳ dấu vết nào của các văn bản hướng dẫn y khoa ngoài chuyên môn.

2. **`previews/page_38_table2_1.png`:**
   - Bảng 2.1 kết thúc Chương 2 với định dạng 4 cột chuẩn xác, các đường kẻ `booktabs` tạo độ thoáng đãng cho trang in, khoảng cách giữa các hàng chữ rộng rãi, dễ đọc.

3. **`previews/page_47_fig3_4_hard_hierarchical.png`:**
   - Sơ đồ phân loại phân cấp cứng 2 bước trong Chương 3.
   - Thể hiện rõ Giai đoạn 1 (Bộ phân loại nhị phân sàng lọc $C_0$ vs $C_1 \cup C_2^+$) và Giai đoạn 2 (Bộ phân loại phân định mức độ nghiêm trọng $C_1$ vs $C_2^+$).
   - Tên gọi Stage 1 và Stage 2 hoàn toàn tách biệt với các ký hiệu ma trận chi phí.

4. **`previews/page_54_table4_1_and_heading4_2.png`:**
   - Tiêu đề mục lớn: `4.2. Đánh giá đóng góp của các nhóm đặc trưng (Feature Ablation)` đặt trang trọng ở đầu trang.
   - Tiểu mục `4.2.1. Thiết lập thực nghiệm` và đoạn văn bản phân tích bài toán RQ1 nằm tiếp nối tự nhiên.
   - Tiểu mục `4.2.2. Kết quả thực nghiệm` và câu dẫn nhập.
   - Bảng 4.1 hiển thị gọn gàng, thanh thoát ở nửa dưới trang giấy, giải quyết triệt để lỗi dàn trang nghiêm trọng của V9.

5. **`previews/page_56_table4_2_nested_cv.png`:**
   - Bảng 4.2 tổng kết 14 mô hình cơ sở qua Nested Cross-Validation 10x5 folds.
   - Dòng cuối cùng được in đậm: `Random Forest (Tuned)` với Macro-F1 = $0.2755 \pm 0.0270$, Balanced Acc = $0.2983 \pm 0.0325$, Weighted-F1 = $0.6437 \pm 0.0087$.
   - Chú thích chân trang số 2 giải thích rõ ràng sự độc lập của phiên bản tối ưu siêu tham số trong từng fold so với phiên bản đóng băng tham số cho các mô hình phân cấp.

---

## 3. KẾT LUẬN

Bản báo cáo V10 là một bước hoàn thiện toàn diện, khắc phục triệt để tất cả các điểm yếu về mặt **khoa học**, **số liệu**, **văn phong**, **bố cục hình ảnh** và **kỹ thuật dàn trang LaTeX** so với bản V9. Tài liệu đáp ứng đầy đủ các tiêu chuẩn học thuật cao nhất của một Đồ án / Khóa luận tốt nghiệp chuyên ngành Khoa học Máy tính.

