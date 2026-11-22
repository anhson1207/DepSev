# NHẬT KÝ THAY ĐỔI CHI TIẾT TỪ V10 SANG V11 (CHANGELOG V10 -> V11)
**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản trước:** `DepSev_ML_Scientific_Report_V10.pdf`  
**Phiên bản hiện tại:** `DepSev_ML_Scientific_Report_V11.pdf`  
**Ngày phát hành:** 31/10/2026

---

## 1. TỔNG HỢP CÁC SỬA ĐỔI SỐ LIỆU THỰC NGHIỆM

| Hạng mục | Phiên bản V10 | Phiên bản V11 (Chuẩn hóa) | Tệp nguồn & Căn cứ đối soát |
| :--- | :---: | :---: | :--- |
| **LR Macro-F1 trên tập F4** | Ghi nhầm $0{,}2152$ tại 3 vị trí văn xuôi | Sửa thành $\mathbf{0{,}2399 \pm 0{,}0148}$ | Khớp 100% với Bảng 4.1 và `feature_ablation_results_final_protocol.csv`. |
| **Mức tăng hiệu năng LR F1 $\to$ F4** | Ghi nhầm tăng $+13{,}3\%$ | Sửa thành tăng $\mathbf{+26{,}3\%}$ ($0{,}1900 \to 0{,}2399$) | Tương đương mức tăng tuyệt đối $+0{,}0499$ điểm Macro-F1. |
| **Độ lệch chuẩn $E_2$ (Hard RF--RF)** | Macro-F1 SD = $0{,}0124$ | Sửa thành Macro-F1 SD = $\mathbf{0{,}0116}$ | Khớp Bảng 4.4, Bảng 4.7 và `experiments/hierarchical_t3/outputs/model_comparison.csv`. |
| **Độ lệch chuẩn $E_{3b}$ (Hard RF--ET)** | Macro-F1 SD = $0{,}0182$ | Sửa thành Macro-F1 SD = $\mathbf{0{,}0090}$ | Khớp Bảng 4.4, Bảng 4.7 và `experiments/hierarchical_t3/outputs/model_comparison.csv`. |
| **Độ lệch chuẩn $E_{6b}$ (Soft Bayes Risk)** | BalAcc SD = $0{,}0163$ | Sửa thành BalAcc SD = $\mathbf{0{,}0218}$ | Khớp Bảng 4.7 và `experiments/soft_hierarchical_t3/outputs/model_comparison.csv`. |
| **Độ lệch chuẩn $E_{7a}$ (Balanced OLR)** | BalAcc SD = $0{,}0152$ | Sửa thành BalAcc SD = $\mathbf{0{,}0276}$ | Khớp Bảng 4.6, Bảng 4.7 và `experiments/ordinal_t3/outputs/model_comparison.csv`. |
| **Nguồn gốc tập $F_{4\_\text{common31}}$** | Diễn giải mơ hồ về việc loại Triglycerides | Làm rõ: Loại 3 biến `SLD010H`, `PAQ605`, `PAD680`; `LBXTR` chưa từng vào F4 | Khớp `src/common31_features.py` và `artifacts/final_feature_schema.json`. |
| **Tỷ lệ giảm sai số ca nặng** | Ghi lẫn lộn 70,62% và 48,33% | Phân định rạch ròi: $E_5 \to E_{6b}$ giảm $\mathbf{70{,}62\%}$; $E_1 \to E_{6b}$ giảm $\mathbf{48{,}33\%}$ | Thống nhất giữa phân tích cơ chế nhân quả cô lập và so sánh hệ thống. |

---

## 2. NÂNG CẤP VÀ TÁI TẠO ĐỒ HỌA (FIGURES)

Toàn bộ 20 hình ảnh minh họa trong thư mục `reports/final_scientific_v11/source/figures/` đã được tái sinh tự động bằng mã nguồn Python `generate_perfect_figures.py` đạt chuẩn xuất bản:
- **Độ phân giải:** 300 DPI, nét căng, tỷ lệ tương phản cao, nền trắng tuyệt đối.
- **Typography:** Đồng bộ 100% font serif (Times New Roman), loại bỏ triệt để font sans-serif và font monospace trong biểu đồ.
- **Cải tiến cụ thể từng đồ thị trọng điểm:**
  1. **Hình 2.1 (`fig2_phq9_tiers.png` - Trang 27):**
     - Loại bỏ các khối văn bản giải thích dài dòng bị nén nhỏ trong các ô.
     - Tách bạch 2 phần: (A) Bài toán 5 lớp gốc $T_5$ (kèm số mẫu $N$ và tỷ lệ %); (B) Bài toán 3 tầng lâm sàng $T_3$ (kèm ý nghĩa định hướng sàng lọc y tế).
  2. **Hình 3.4 (`fig3_4_hard_hierarchical.png` - Trang 47):**
     - Rút gọn văn bản trong các khối Stage 1 (Tách $C_0$) và Stage 2 (Tách $C_1$ vs $C_2^+$).
     - Tăng kích thước chữ hiển thị lên $\ge 9$pt, rõ ràng dễ đọc.
     - Đóng khung nét đứt màu đỏ nhạt làm nổi bật nút thắt sai số tích lũy (Cumulative Error Bottleneck).
  3. **Hình 3.5 (`fig3_5_soft_cost_sensitive.png` - Trang 48):**
     - Tinh giản nội dung các khối Soft Fusion và Bayes Risk Rule, tập trung vào bản chất toán học: Vector xác suất hậu nghiệm $P(C_k|x)$ kết hợp ma trận phạt tổn thất $c(y, \hat{y})$ để tìm nhãn tối ưu $\arg\min R(\hat{y}|x)$.
  4. **Hình 4.18 (`cost_tradeoff_curve.png` - Trang 74):**
     - Viết lại hàm `make_cost_tradeoff_curve()` đọc trực tiếp dữ liệu từ `experiments/soft_hierarchical_t3/outputs/cost_sensitivity.csv`.
     - Đổi tiêu đề đồ thị thành: *"Trade-off between C2+ Recall and C0 False Alarm Rate across Cost Multipliers"*, xóa bỏ thuật ngữ không chính xác *"Pareto Frontier"*.
     - Trực quan hóa rõ ràng điểm vận hành tối ưu tại $w=10$ ($E_{6b}$).

---

## 3. BIÊN TẬP VĂN PHONG, NỘI DUNG VÀ TÀI LIỆU THAM KHẢO

1. **Loại bỏ trích dẫn sinh học không chuẩn xác (Chương 2):**
   - Loại bỏ việc gán ghép bài báo dịch tễ học thuần túy của Pratt et al. (2014) cho các cơ chế trục dưới đồi - tuyến yên - tuyến thượng thận (HPA axis) và viêm thần kinh.
   - Giữ Pratt et al. (2014) đúng vai trò thống kê tỷ lệ hiện mắc trầm cảm trong dân số Hoa Kỳ.
2. **Thiết kế lại Bảng 2.1 (Tổng hợp nghiên cứu liên quan - Trang 38):**
   - Cân chỉnh lại tỷ lệ độ rộng cột: `p{3.0cm} p{2.2cm} p{3.4cm} X`.
   - Khắc phục hoàn toàn lỗi rớt dòng chữ cụt của tác giả Zhang et al. và dòng mô tả của DepSev-ML.
   - Bổ sung nhãn `[UNVERIFIED]` cho Zhang et al. (2020) và Patel et al. (2016) theo nguyên tắc liêm chính học thuật.
   - Đính chính bài báo của Dipnall et al. (2016) công bố trên tạp chí *PLOS ONE* (sử dụng chu kỳ NHANES 2009--2010), thay vì thông tin nhầm lẫn trước đây.
3. **Phân biệt rạch ròi RF Tuned và Frozen RF:**
   - Làm rõ: **RF Tuned** là mô hình tốt nhất thu được từ Nested CV trên bài toán 5 lớp phẳng ($T_5$).
   - **Frozen RF**: Là việc đóng băng trọng số của RF Tuned để làm bộ trích xuất đặc trưng (hoặc phân loại nhị phân $C_0$ vs $C_1^+$) trong Giai đoạn 1 của các cấu trúc phân tầng ($T_3$).
4. **Sửa đổi chú thích và văn bản tham chiếu Chương 4:**
   - Chú thích Hình 4.15: Đổi thành phân tích **cả hai loại sai số 2 bước** (bỏ sót ca nặng $C_2^+ \to C_0$ và báo động giả $C_0 \to C_2^+$).
   - Chú thích Hình 4.17: Cập nhật đầy đủ 3 mô hình Ordinal gồm $E_{7a}$ (Balanced OLR), $E_{7b}$ (CumBin LR), $E_{7b}$ (CumBin RF).
   - Đồng bộ số lượng ma trận nhầm lẫn tham chiếu trong văn bản thành 7 mô hình đại diện (4 mô hình tại Hình 4.16 và 3 mô hình tại Hình 4.17).

---

## 4. TỐI ƯU HÓA DÀN TRANG VÀ TRIỆT TIÊU FONT MONOSPACE

1. **Triệt tiêu hoàn toàn font Monospace (0,00%):**
   - Rà soát toàn bộ mã nguồn LaTeX, loại bỏ mọi lệnh `\texttt{...}`.
   - Kết quả trích xuất tự động qua PyMuPDF xác nhận **0 ký tự Monospace** trên toàn bộ 91 trang tài liệu.
2. **Khơi thông dòng chảy đồ thị (Trang 68--76):**
   - Tháo gỡ các lệnh `\FloatBarrier` cứng gây ứ đọng đồ thị.
   - Sắp xếp đồ thị theo cặp đôi logic (Hình 4.13 & 4.14; Hình 4.16 & 4.17) và đưa bảng Bảng 4.7 vào vị trí tự nhiên sau phần diễn giải.
   - Loại bỏ hoàn toàn các khoảng trắng lớn bất thường, mang lại trải nghiệm đọc liền mạch, chuyên nghiệp.

---

## 5. TÌNH TRẠNG LIÊN KẾT VÀ BIÊN DỊCH

- Biên dịch thành công 100% bằng công cụ `tectonic`.
- Số lượng liên kết gãy (`??`): **0**.
- Tổng số trang duy trì ổn định: **91 trang**.

