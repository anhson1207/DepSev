# NHẬT KÝ THAY ĐỔI NỘI DUNG BÁO CÁO DEPSEV-ML V7
*(Content Changelog Across Report Versions)*

Tài liệu này ghi nhận chi tiết toàn bộ các chỉnh sửa về mã nguồn, hình ảnh, bảng biểu và cấu trúc tài liệu giữa phiên bản V6 và phiên bản V7.

---

## 1. CÁC TẬP TIN HÌNH ẢNH MỚI VÀ ĐIỀU CHỈNH (`figures/`)

| Tên tập tin hình ảnh | Loại thay đổi | Mô tả chi tiết |
| :--- | :---: | :--- |
| `figures/granularity_t5_confusion_matrix.png` | **MỚI** | Biểu đồ nhiệt (Heatmap) Ma trận nhầm lẫn 5x5 của mô hình Random Forest trên bài toán 5 lớp $T_5$, hiển thị cả số lượng mẫu tuyệt đối và tỷ lệ phần trăm theo hàng. Trích xuất từ `results/tables/granularity_confusion_matrices.csv`. |
| `figures/multicycle_cohort_comparison.png` | **MỚI** | Biểu đồ đối sánh hiệu năng Macro-F1 và Balanced Accuracy theo độ chi tiết nhãn từ $T_2$ đến $T_5$ qua ba nhánh dữ liệu: Branch A (Chu kỳ đơn lẻ), Branch B (Gộp đa chu kỳ), Branch D (Kiểm định thời gian). Đặt tại Mục 4.5. |
| `figures/multicycle_soft_comparison.png` | **ĐIỀU CHỈNH** | Được di chuyển từ Mục 4.5 (nơi bị gán nhầm nội dung trong V6) sang đúng vị trí tại Mục 4.8 (Kiểm định ngoại suy theo dòng thời gian). |
| `figures/fig2_decision_tree_concept.png` | **CẬP NHẬT** | Chuẩn hóa nhãn "Cây quyết định" thành tiếng Anh "Decision Tree". |
| `figures/fig2_random_forest_architecture.png` | **CẬP NHẬT** | Chuẩn hóa toàn bộ nhãn các cây thành "Decision Tree 1 ... B". |
| `figures/two_stage_data_flow.png` | **SỬA LỖI** | Xóa bỏ một môi trường float bị lặp lại trong `chapter3.tex`. |

---

## 2. BẢNG BIỂU DỮ LIỆU ĐÃ ĐIỀU CHỈNH (`tables/`)

| Tên tập tin bảng | Mục áp dụng | Chi tiết chỉnh sửa |
| :--- | :---: | :--- |
| `tables/table_executive_decision_matrix.tex` | Mục 4.9 | Đổi tiêu đề từ "Ma trận quyết định điều hành" thành "Tổng hợp các mô hình tiêu biểu theo từng tiêu chí". Xóa bỏ từ ngữ phóng đại, tập trung vào 4 mô hình: RF (Tuned) trên $T_5$, $E_2$ trên $T_3$, $E_{7a}$ (Balanced OLR) và $E_{6b}$ (Soft Bayes Risk). |
| `tables/table_master_comparison.tex` | Mục 4.5 | Sửa toàn bộ số liệu cột $C_0 \to C_2^+$ trong Phần B để khớp chính xác với Out-of-fold Confusion Matrix gộp ($E_0: 316, E_1: 261, E_2: 222, E_{3a}: 300, E_{3b}: 279, E_4: 139, E_5: 176, E_{6a}: 215, E_{6b}: 294, E_{7a\text{ unw}}: 59, E_{7a\text{ bal}}: 357, E_{7b\text{ LR}}: 727, E_{7b\text{ RF}}: 313$). Thống nhất số ca bỏ sót $C_2^+ \to C_0$ của $E_2$ là **98 ca**. Thêm chú thích giải thích rõ phương pháp tính. |
| `tables/table_hard_hierarchical.tex` | Mục 4.5 | Sửa số ca bỏ sót của mô hình $E_2$ từ 95 thành **98 ca**. |
| `tables/table_soft_hierarchical.tex` | Mục 4.5 | Thêm ghi chú giải thích sự khác biệt giữa điểm trung bình cộng qua 5 fold và ma trận nhầm lẫn gộp (Pooled Confusion Matrix). |
| `tables/table_multicycle_results.tex` | Mục 4.8 | Làm rõ ngữ cảnh đánh giá của Branch D $M_2$. |

---

## 3. CÁC CHƯƠNG NỘI DUNG ĐÃ BIÊN TẬP (`chapters/`)

| Chương / Tệp | Nội dung biên tập chính |
| :--- | :--- |
| `chapters/chapter1.tex` | Mở đầu với văn phong khiêm tốn, dẫn dắt rõ ràng quá trình nghiên cứu hai giai đoạn $T_5 \to T_3$. Làm rõ $T_3$ là nhiệm vụ độc lập, không phải kết quả nâng cao trên cùng bài toán 5 lớp. |
| `chapters/chapter2.tex` | Giản lược các mục lý thuyết dàn trải của LR, DT, XGBoost. Đi sâu vào giải thích thuật toán cốt lõi **Random Forest** (Decision Tree, Bootstrap Sampling, ngẫu nhiên hóa đặc trưng, Soft Voting, cố định siêu tham số). Cắt bỏ công thức giải tích OOB 36.8% và công thức phương sai ensemble tổng quát. Thống nhất quy ước ma trận chi phí $\mathbf{C}(i, j)$ với $i$ là nhãn thực tế, $j$ là nhãn dự đoán. |
| `chapters/chapter3.tex` | Làm rõ quy trình huấn luyện in-fold: Stage 1 huấn luyện trên phân vùng train nội bộ của fold ($N_{\text{train}} \approx 3.236$), Stage 2 huấn luyện nghiêm ngặt trên các mẫu có nhãn thật $\text{PHQ-9} \ge 5$ trong chính phân vùng train này ($N_{\text{train, S2}} \approx 823$). Xóa hình ảnh lặp 3.6. |
| `chapters/chapter4.tex` | Chèn Hình 4.5 (Confusion Matrix 5x5 của $T_5$), thay thế Hình 4.8 bằng biểu đồ đối sánh các nhánh dữ liệu, chuyển biểu đồ ngoại suy thời gian sang Mục 4.8. Thống nhất số ca bỏ sót của $E_2$ là 98 ca. Điều chỉnh các khẳng định về PCA và tương quan sinh lý sang văn phong khoa học chừng mực. |
| `chapters/chapter5.tex` | Tinh gọn kết luận còn 3 trang súc tích. Trả lời trực diện 5 câu hỏi nghiên cứu (RQ1--RQ5), tổng kết 4 mô hình tiêu biểu, trình bày 3 hạn chế khách quan và 3 hướng nghiên cứu tương lai cụ thể. |
| `chapters/appendix.tex` | Sửa dứt điểm lỗi float của Bảng C.1 bằng việc thêm lệnh ngăn chặn float đỉnh trang (`\suppressfloats[t]`) ngay dưới tiêu đề Phụ lục C. Chuẩn hóa thuật ngữ sang tiếng Anh. |
| `uetgraduation.cls` | Bổ sung `\suppressfloats[t]` vào định nghĩa lệnh `\chapter` để ngăn ngừa triệt để hiện tượng float bảng biểu nổi lên phía trên tiêu đề chương. |
| `main.tex` | Rà soát toàn bộ cấu trúc văn bản, bảng chữ viết tắt và tài liệu tham khảo. |
| `references.bib` | Bổ sung 4 trích dẫn nghiên cứu liên quan có thật: `pratt2014depression`, `patel2016systematic`, `dipnall2016fusing`, `zhang2020machine`. |

