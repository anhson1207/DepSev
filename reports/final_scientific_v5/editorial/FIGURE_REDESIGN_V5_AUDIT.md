# BIÊN BẢN TÁI THIẾT KẾ ĐỒ HỌA TOÀN DIỆN (FIGURE REDESIGN AUDIT) - PHIÊN BẢN V5

Biên bản này ghi nhận toàn bộ các xử lý kỹ thuật nhằm giải quyết dứt điểm các lỗi đồ họa được người dùng phản ánh:
- **Hiện tượng "mất bo góc":** Xảy ra khi các khối hình hộp bo tròn (`FancyBboxPatch`) đặt sát biên tọa độ hoặc tham số `pad` mở rộng ra ngoài phạm vi `[0, 1]`, khiến lệnh `savefig` cắt phẳng các góc bo tròn ngoài cùng.
- **Hiện tượng "dính vào nhau":** Các hộp nội dung đặt quá sát nhau (khoảng cách < 0.02), các mũi tên đâm xuyên qua đường viền hoặc bắt đầu từ bên trong khối hộp, chữ quá nhiều dòng bị ép trong không gian chật hẹp, nhãn chú thích đè lên mũi tên.
- **Hiện tượng "chưa rõ":** Phông chữ quá nhỏ (< 8pt), thiếu độ tương phản, các đường liên kết hai chiều vẽ chồng lên nhau gây nhòe nét.

---

## 1. Chuẩn hóa hệ thống thiết kế đồ họa mới (`generate_perfect_figures.py`)

1. **Hệ tọa độ lưới số nguyên chuẩn tắc `[0, 100] x [0, 100]`:**
   - Thay thế việc dùng tọa độ thập phân `[0.0, 1.0]` dễ gây sai số làm tròn bằng lưới số nguyên `[0, 100]`.
   - Giới hạn an toàn toàn bộ các thành phần đồ họa nằm nghiêm ngặt trong vùng `[3, 97]` trên trục X và `[4, 96]` trên trục Y.
   - Luôn duy trì vùng đệm an toàn tối thiểu $\ge 3\text{ đơn vị}$ cách mép canvas.

2. **Khắc phục triệt để hiện tượng mất bo góc (Zero Clipped Corners):**
   - Thiết lập cấu hình bo góc chuẩn: `boxstyle="round,pad=0.8,rounding_size=1.8"` đến `rounding_size=2.2`.
   - Lệnh lưu ảnh luôn kích hoạt `bbox_inches='tight', pad_inches=0.15`, bảo đảm luôn có dải biên trắng thuần khiết $\ge 15\text{ pixels}$ bao quanh 4 phía của mọi hình vẽ.
   - Kết quả kiểm định tự động: **100% các hình đều đạt chuẩn `CLEAN WHITE MARGIN (PASS)`**.

3. **Khắc phục triệt để hiện tượng dính vào nhau (Spacious Layout):**
   - **Khoảng cách giữa các hộp:** Tối thiểu từ $4$ đến $6\text{ đơn vị}$ lưới (tương đương khoảng hở thoáng đãng $\approx 1.5 - 2.0\text{ cm}$ trên bản in).
   - **Đầu mút mũi tên:** Luôn có khoảng cách hở $1.0 - 1.5\text{ đơn vị}$ trước khi chạm viền hộp, tuyệt đối không đâm xuyên viền và không bắt đầu từ trong lòng hộp.
   - **Nhãn mũi tên:** Đặt phía trên hoặc phía dưới mũi tên với khoảng cách riêng, không bao giờ đè lên thân mũi tên hay chạm viền hộp lân cận.
   - **Nội dung văn bản bên trong hộp:** Lược giản tối đa các đoạn văn dài, chỉ giữ lại các gạch đầu dòng từ khóa súc tích, độ giãn dòng `linespacing=1.2 - 1.3`, cỡ chữ đồng nhất từ $8.0\text{pt}$ đến $10.5\text{pt}$ đọc rõ ràng trên khổ giấy A4.

4. **Hệ màu sắc học thuật cao cấp và độ tương phản cao:**
   - **Màu nền hộp:** Sử dụng các tông màu pastel dịu mắt: Xanh nhạt `#EFF6FF`, Xanh ngọc `#ECFDF5`, Vàng hổ phách `#FFFBEB`, Đỏ nhạt `#FEF2F2`, Tím nhạt `#FAF5FF`.
   - **Đường viền hộp:** Tương phản sắc nét với độ dày $1.3 - 1.6\text{pt}$: Xanh đậm `#2563EB`, Lục đậm `#059669`, Cam đậm `#D97706`, Đỏ cờ `#DC2626`, Tím đậm `#7C3AED`.
   - **Màu chữ:** Tối màu `#1E293B` hoặc đồng màu đậm theo nhóm để tối đa hóa khả năng đọc (readability).

---

## 2. Danh mục 17 hình vẽ được tái thiết kế và chuẩn hóa

| STT | Mã tệp hình ảnh | Vị trí trong báo cáo | Tiêu đề hình vẽ | Các cải tiến thẩm mỹ chính |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `fig1_problem_formulation.png` | Hình 1.1 (Trang 20) | Sơ đồ tổng quan phát biểu bài toán DepSev-ML | 4 cột phân bố cân xứng X=3, 27.5, 52, 76.5; mũi tên nối cách viền 1.5 đơn vị; bo góc hoàn hảo. |
| 2 | `fig_research_progression_t5_to_t3.png` | Hình 1.2 (Trang 22) | Quy trình tiến triển nghiên cứu $T_5 \to T_3$ | Tách biệt rạch ròi 2 pha T5 và T3; mũi tên chuyển dịch ở giữa thoáng đãng; chữ không đè hộp. |
| 3 | `fig2_phq9_tiers.png` | Hình 2.1 (Trang 25) | Thang đo PHQ-9 (0-27) và ánh xạ sang $T_3$ | Khối dải màu 5 bậc bo góc mềm mại; trục số PHQ-9 rõ nét; dấu ngoặc ôm T3 thanh thoát bên trên. |
| 4 | `fig2_gut_brain_metabolic_axis.png` | Hình 2.2 (Trang 27) | Cơ chế y sinh học liên kết sinh lý với trầm cảm | 4 khối ngoại vi đặt 4 góc cân đối; mũi tên 2 chiều `<->` thanh mảnh không đè nhau; tâm điểm rõ ràng. |
| 5 | `fig2_imbalance_overlap.png` | Hình 2.3 (Trang 28) | Mất cân bằng lớp và chồng lấn phân phối | Cột mất cân bằng kèm tỷ lệ % rõ ràng; đường cong mật độ xác suất và vùng chồng lấn Bayes sắc nét. |
| 6 | `fig2_decision_tree_concept.png` | Hình 2.4 (Trang 30) | Nguyên lý phân nhánh Cây quyết định | Cấu trúc cây 3 tầng thoáng đãng; 4 nút lá xếp đều khoảng cách X=3, 27.5, 52, 76.5 không còn va chạm. |
| 7 | `fig2_random_forest_architecture.png` | Hình 2.5 (Trang 31) | Kiến trúc Rừng ngẫu nhiên | Luồng từ D $\to$ 3 mẫu Bootstrap $\to$ 3 Cây $\to$ Soft Voting; các khoảng hở phân bổ đều, không dính. |
| 8 | `fig2_modeling_paradigms.png` | Hình 2.6 (Trang 34) | 3 trường phái: Phẳng, Phân cấp, Thứ bậc | 3 bảng con (a, b, c) độc lập, các hộp rẽ nhánh phân cấp và ngưỡng cắt song song thứ bậc chuẩn xác. |
| 9 | `depsev_overall_architecture.png` | Hình 3.1 (Trang 37) | Kiến trúc tổng thể 7 module DepSev-ML | Thiết kế mới 3 tầng chức năng: Nguồn dữ liệu $\to$ Ingestion/Harmonization $\to$ In-fold Isolation $\to$ Models/Eval. |
| 10 | `fig3_infold_cv_pipeline.png` | Hình 3.2 (Trang 39) | Quy trình tiền xử lý in-fold Zero Data Leakage | Khung in-fold lớn bao bọc Train (Fit) và Test (Transform only); nhãn chuyển tham số nằm cạnh mũi tên. |
| 11 | `fig3_3_feature_groups.png` | Hình 3.3 (Trang 41) | Cấu trúc 4 không gian đặc trưng $F_1 \to F_4$ | 4 thanh ngang xếp chồng cách đều $Y=3, 26, 50, 74$; lề thụt đầu dòng thẳng hàng; bo góc thanh lịch. |
| 12 | `fig3_4_hard_hierarchical.png` | Hình 3.5 (Trang 44) | Phân cấp cứng và điểm nghẽn lan truyền sai số | Luồng M1 $\to$ M2 rõ ràng; hộp cảnh báo 120 ca nặng bị bỏ sót viền nét đứt đỏ nổi bật, lùi sâu cách viền. |
| 13 | `fig3_5_soft_cost_sensitive.png` | Hình 3.6 (Trang 45) | Phân cấp mềm và quyết định rủi ro Bayes | Khối kết hợp xác suất mềm và khối rủi ro Bayes đặt song song thoáng đãng; công thức hiển thị sắc nét. |
| 14 | `fig3_6_ordinal_classification.png` | Hình 3.7 (Trang 47) | Nguyên lý phân loại thứ bậc (OLR & CBC) | Subplot OLR hiển thị đường Sigmoid và 2 vạch ngưỡng cắt; Subplot CBC minh họa trực quan ca vi phạm. |
| 15 | `fig3_7_evaluation_protocol.png` | Hình 3.8 (Trang 49) | Hai tầng giao thức kiểm định thực nghiệm | Khung Tầng 1 (Nested CV) và Tầng 2 (Temporal Test) phân tầng logic, mũi tên liên kết chính giữa. |
| 16 | `fig4_algorithm_comparison_t5_t3.png` | Hình 4.6 (Trang 60) | Đối sánh thuật toán trên $T_5$ và $T_3$ | Biểu đồ cột đôi có nhãn số liệu trên đầu cột, khoảng cách cột vừa vặn, chú giải góc trên rõ nét. |
| 17 | `fig4_confusion_error_analysis.png` | Hình 4.7 (Trang 62) | Phân tích sai số lâm sàng và Recall từng lớp | Biểu đồ số ca bỏ sót với hộp làm nổi bật E6b giảm 74.2%; biểu đồ Recall 3 lớp đối sánh E1, E6b, E7a. |

---

## 3. Kết luận kiểm định chất lượng

- **Biên độ viền an toàn:** Đạt 100% viền trắng sạch quanh ảnh, không có bất kỳ pixel tử tế nào bị chạm hoặc cắt phẳng ở góc.
- **Trình bày trong PDF:** Khi chèn vào văn bản LaTeX với `width=\textwidth` hoặc `width=0.9\textwidth`, hình hiển thị sắc nét, khung bo góc tròn trịa, thanh thoát, kích thước chữ hài hòa hoàn hảo với font chữ chung của báo cáo (12pt Times New Roman).

