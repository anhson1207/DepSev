# BÁO CÁO DỰ ÁN KHOA HỌC DEPSEV-ML (PHIÊN BẢN V5)

> **Đề tài:** Nghiên cứu khung học máy cho dự đoán mức độ triệu chứng trầm cảm từ dữ liệu sinh lý (DepSev-ML)  
> **Sinh viên thực hiện:** Nguyễn Anh Sơn (English: Nguyen Anh Son)  
> **Ngành học:** Khoa học máy tính (Computer Science)  
> **Cán bộ hướng dẫn:** TS. Vũ Thị Hồng Nhạn (Dr. Vu Thi Hong Nhan)  
> **Đơn vị:** Trường Đại học Công nghệ -- Đại học Quốc gia Hà Nội (VNU-UET)  
> **Năm hoàn thành:** 2026  

---

## 1. Giới thiệu tổng quan phiên bản V5

Bản phân phối **V5** đại diện cho phiên bản hoàn thiện toàn diện về mặt phương pháp luận nghiên cứu, logic khoa học thực nghiệm, thiết kế trực quan hóa và dàn trang học thuật LaTeX chất lượng cao cho báo cáo dự án khoa học **DepSev-ML**.

### Tệp sản phẩm chính:
- **`DepSev_ML_Scientific_Report_V5.pdf`**: Tệp PDF hoàn thiện sẵn sàng nộp (82 trang, 0 lỗi tràn lề, siêu liên kết mục lục tương tác 100%, chuẩn kích thước font và khoảng cách chuẩn VNU-UET).
- **`source/`**: Toàn bộ mã nguồn LaTeX đã được module hóa chuẩn tắc (`main.tex`, `uetgraduation.cls`, `chapters/`, `figures/`, `tables/`).
- **`editorial/`**: Hồ sơ biên tập học thuật, ghi chép thay đổi nội dung, lược giản công thức và kế hoạch thiết kế hình ảnh.
- **`validation/`**: Báo cáo kiểm định chất lượng dàn trang bằng PyMuPDF, bảng đối sánh bằng chứng thực nghiệm và nhật ký biên dịch Tectonic.

---

## 2. Các nâng cấp cốt lõi trong Phiên bản V5

1. **Làm rõ quá trình nghiên cứu hai giai đoạn ($T_5 \to T_3$):**
   - **Giai đoạn 1 ($T_5$):** Khảo sát 5 mức độ lâm sàng chuẩn PHQ-9 ($C_0 \to C_4$). Chỉ ra bằng chứng thực nghiệm về ngưỡng trần thông tin của dữ liệu sinh lý ngoại vi (Macro-F1 bão hòa quanh 0.27), sự sụp đổ của lớp nhẹ $C_1$ (Recall 16.29\%), và hiện tượng chồng lấn phân phối dày đặc được chứng minh bằng tương quan sinh hóa và PCA 2D.
   - **Giai đoạn 2 ($T_3$):** Tái cấu trúc sang không gian 3 nhóm lâm sàng thực dụng ($C_0: 0-4$, $C_1: 5-9$, $C_2^+: 10-27$). Thiết kế kiến trúc phân cấp hai bước: Stage 1 sàng lọc nhị phân, Stage 2 phân tầng mức độ.
   - **Làm rõ bản chất huấn luyện và suy luận:** Stage 2 được huấn luyện **CHỈ trên các mẫu có nhãn thật $\text{PHQ-9} \ge 5$** ($N = 1.029$ ca). Trong giai đoạn suy luận, cơ chế định tuyến cứng (Hard Routing) làm phát sinh điểm nghẽn nghiêm trọng (120 ca nặng $C_2^+$ bị Stage 1 gán nhầm thành $C_0$ và loại bỏ vĩnh viễn), trong khi cơ chế định tuyến mềm kết hợp tối ưu chi phí Bayes ($E_{6b}$) giải quyết triệt để vấn đề này (giảm 74.2\% số ca bỏ sót).
   - Khẳng định $T_3$ là bài toán độc lập với không gian nhãn tái cấu trúc, không phải kết quả cải thiện trực tiếp trên cùng không gian 5 lớp.

2. **Bổ sung mục giới thiệu 4 thuật toán cốt lõi trong Chương 2:**
   - Trình bày có hệ thống Decision Tree, Logistic Regression, Random Forest và XGBoost.
   - Mỗi thuật toán phân tích đồng nhất theo 4 khía cạnh: (1) Nguyên lý hoạt động; (2) Ưu/nhược điểm trên dữ liệu bảng y tế; (3) Vai trò trong DepSev-ML; (4) Lý do lựa chọn.

3. **Tiểu mục riêng trình bày chi tiết 8 bước áp dụng Random Forest trong Chương 3:**
   - Chi tiết hóa 8 bước: Chuẩn bị dữ liệu in-fold $\to$ Phân chia dữ liệu Stratified 5-Fold / Nested CV $\to$ Huấn luyện cây quyết định $\to$ Từ cây đến rừng (Bootstrap, max_features=0.5, Soft voting) $\to$ Cấu hình chuẩn Frozen RF Spec $\to$ Tinh chỉnh và chọn mô hình cơ sở $\to$ Đánh giá hiệu năng và phân tích hạn chế $\to$ Chuyển giao RF sang kiến trúc phân cấp $T_3$.

4. **Hệ thống 8 hình ảnh trọng tâm được thiết kế mới hoàn toàn:**
   - Đồ họa tối giản, trang nhã, chỉ dùng từ khóa ngắn gọn, độ tương phản cao, kích thước font tối thiểu $\ge 9\text{pt}$ đọc rõ trên khổ giấy in A4, độ phân giải 300 DPI.

5. **Lược bỏ công thức toán rườm rà:**
   - Rà soát và loại bỏ các công thức sách giáo khoa cơ bản, chỉ giữ lại các biểu thức toán học thực sự cần thiết và đặc thù cho phương pháp luận của đề tài.

6. **Hoàn thiện thiết kế học thuật và dàn trang:**
   - 3 trang bìa có logo Đại học Công nghệ (VNU-UET) sắc nét, căn lề chuẩn mực.
   - Mục lục, Danh mục hình vẽ, Danh mục bảng có cỡ chữ lớn (`\Huge`), toàn bộ các dòng liên kết tương tác hoạt động như thẻ `<a>`.
   - Kiểm định PyMuPDF tự động: **0 lỗi tràn lề** trên toàn bộ 82 trang.

---

## 3. Hướng dẫn biên dịch lại báo cáo

Để biên dịch lại báo cáo từ mã nguồn, sử dụng trình biên dịch Tectonic:
```bash
cd source
/Users/nguyenanhson/.local/bin/tectonic main.tex -o ..
cp ../main.pdf ../DepSev_ML_Scientific_Report_V5.pdf
```
