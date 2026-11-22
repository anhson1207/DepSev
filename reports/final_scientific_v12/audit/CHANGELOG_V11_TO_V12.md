# NHẬT KÝ THAY ĐỔI TỪ V11 SANG V12 (CHANGELOG V11 -> V12)
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Bản báo cáo hoàn thiện:** `DepSev_ML_Scientific_Report_V12.pdf`  
**Ngày phát hành:** 22/11/2026

---

## 1. MỤC TIÊU VÀ ĐỊNH HƯỚNG CẢI TỔ

Bản V12 được thực hiện theo định hướng trực tiếp từ giảng viên phản biện:
1. **Tinh gọn độ dài:** Giảm từ 91 trang xuống **56 trang** ("tầm 50 gì đó"), loại bỏ các phần độn lý thuyết giáo khoa và bảng biểu trùng lặp.
2. **Khử triệt để định dạng AI:**
   - Cắt giảm **63% số lượng chữ in đậm** (`\textbf`), triệt tiêu việc bôi đậm tràn lan ở đầu mỗi câu và đầu gạch đầu dòng.
   - Thay thế toàn bộ khuôn mẫu **RQ1, RQ2, RQ3...** bằng lối diễn đạt câu hỏi khoa học và mục tiêu nghiên cứu tự nhiên của con người.
3. **Cá nhân hóa đồ họa (Custom Visual Taste):**
   - Thay thế hoàn toàn bảng màu mặc định kiểu Claude/Tailwind (xanh lơ, cam, đỏ tươi) bằng bảng màu kỹ thuật hàn lâm (Slate Navy `#2C3E50`, Muted Sage `#4A6572`, Muted Ochre `#B7950B`, Terracotta `#A93226`).
   - Đường viền thanh mảnh (`lw=1.1`), góc bo vi mô (`rounding_size=0.8`), thiết kế tối giản như bản vẽ kỹ thuật.
4. **Bảo tồn toàn vẹn tính chính xác khoa học:**
   - Giữ nguyên 100% các giá trị thực nghiệm đã đối soát: LR F4 = 0,2399 $\pm$ 0,0148; các SD của $E_2, E_{3b}, E_{6b}, E_{7a}$; giải thích chính xác 3 biến bị loại của $F_{4\_\text{common31}}$; phân định rạch ròi mức giảm 70,62% (nhân quả cô lập Bayes) và 48,33% (hệ thống so với baseline).

---

## 2. CHI TIẾT CÁC CẢI TIẾN THEO CHƯƠNG

### 2.1. Phần mở đầu (Front Matter, Trang 1 -- 17)
- Rút gọn phần Tóm tắt và Abstract, bỏ các thẻ bôi đậm khuôn sáo.
- Bảng từ viết tắt và mục lục được tinh chỉnh độ giãn dòng, giúp luồng chuyển sang Chương 1 mượt mà.

### 2.2. Chương 1: Giới thiệu (Trang 18 -- 22, 5 trang)
- Thay thế danh sách liệt kê `RQ1--RQ5` bằng đoạn văn dẫn dắt học thuật tự nhiên về 5 mục tiêu nghiên cứu trọng tâm.
- Loại bỏ toàn bộ việc in đậm ở các gạch đầu dòng thách thức kỹ thuật và đóng góp.
- Sơ đồ phát biểu bài toán (Hình 1.1) và tiến trình thực nghiệm (Hình 1.2) được vẽ lại với tone màu Slate/Sage/Terracotta nhã nhặn.

### 2.3. Chương 2: Cơ sở lý thuyết và Nghiên cứu liên quan (Trang 23 -- 27, 5 trang)
- **Cắt bỏ hoàn toàn lý thuyết sách giáo khoa:** Loại bỏ tiểu mục phân tích sâu Decision Tree (công thức Gini, nút gốc, nút lá) và Random Forest cơ bản, giúp giảm 8 trang độn không cần thiết.
- Tập trung vào phân tích 4 trường phái mô hình hóa (Phẳng, Phân cấp cứng, Phân cấp mềm Bayes, Thứ bậc) và Bảng 2.1 so sánh các nghiên cứu liên quan có nhãn `[UNVERIFIED]`.
- Giữ lại 3 hình ảnh cốt lõi: Hình 2.1 (Phân tầng PHQ-9), Hình 2.2 (Mất cân bằng và chồng lấn phân phối), Hình 2.3 (4 trường phái mô hình hóa).

### 2.4. Chương 3: Dữ liệu và Phương pháp nghiên cứu (Trang 28 -- 34, 7 trang)
- Trình bày trực diện quy trình 7 bước (Hình 3.1) và quy trình tiền xử lý in-fold bảo đảm Zero Data Leakage (Hình 3.2).
- Trình bày ngắn gọn, tường minh định nghĩa tập $F_{4\_\text{common31}}$ (loại 3 biến `SLD010H`, `PAQ605`, `PAD680`, biến `LBXTR` chưa từng vào F4).
- Tinh giản sơ đồ khối phân cấp cứng (Hình 3.3) và phân cấp mềm Bayes (Hình 3.4).

### 2.5. Chương 4: Thực nghiệm và Thảo luận (Trang 35 -- 47, 13 trang)
- Trình bày tập trung các bảng số liệu then chốt:
  - Bảng 4.1: Feature Ablation ($F_1 \to F_4$, khẳng định LR F4 = 0,2399 $\pm$ 0,0148).
  - Bảng 4.2: Đối chuẩn Nested CV trên $T_5$ (chọn RF làm baseline).
  - Bảng 4.3 & Hình 4.3: Suy giảm hiệu năng từ $T_2$ sang $T_5$.
  - Bảng 4.4: Bảng Master tổng hợp so sánh $E_0 \to E_7$ trên $T_3$.
  - Bảng 4.5: Kiểm định độc lập theo dòng thời gian trên NHANES 2015--2016.
  - Bảng 4.6: Ma trận quyết định ứng dụng y tế.
- Phân tích rạch ròi cơ chế giảm sai số ca nặng: giảm **70,62%** trên cùng nền tảng mô hình mềm ($E_5 \to E_{6b}$ từ 211 xuống 62 ca) và giảm **48,33%** so với baseline phân tầng cứng ($E_1 \to E_{6b}$ từ 120 xuống 62 ca).
- Loại bỏ các biểu đồ phụ lặp ý, giữ lại các đồ thị phân tích sai số sắc nét (Hình 4.4, 4.5, 4.6, 4.7).

### 2.6. Chương 5: Kết luận và Hướng phát triển (Trang 48 -- 50, 3 trang)
- Diễn đạt súc tích, tự nhiên, khiêm tốn.
- Chỉ rõ 3 giới hạn khách quan cốt lõi (dữ liệu cắt ngang, tính gián tiếp của chỉ số sinh lý, thang đo tự khai PHQ-9).

### 2.7. Phụ lục và Tài liệu tham khảo (Trang 51 -- 56, 6 trang)
- Gom toàn bộ 4 phụ lục thành 1 chương duy nhất "Phụ lục tra cứu kỹ thuật", loại bỏ 3 trang ngắt chương trắng thừa thãi.
- Danh mục tài liệu tham khảo đồng bộ chuẩn mực IEEE với nhãn `[UNVERIFIED]`.

---

## 3. KẾT QUẢ KIỂM ĐỊNH KỸ THUẬT

- **Số trang chính thức:** **56 trang** (giảm 38,5% so với V11).
- **Lỗi liên kết gãy (`??`):** **0 lỗi**.
- **Tỷ lệ font Monospace:** **0,00%**.
- **Tỷ lệ ký tự Bold:** **Giảm từ 12,85% xuống 8,50%** (khử hoàn toàn bold trong thân bài).
