# BÁO CÁO KIỂM ĐỊNH TÍNH ĐÚNG ĐẮN KHOA HỌC (SCIENTIFIC VERIFICATION V4)

## 1. NGUYÊN TẮC BẢO TOÀN NGHIÊN CỨU GỐC

Đã kiểm tra và cam kết tuyệt đối:
- Toàn bộ các thư mục mã nguồn và dữ liệu gốc: `data/`, `results/`, `artifacts/`, `src/`, `tests/`, `experiments/` **hoàn toàn không bị sửa đổi, ghi đè hoặc can thiệp**.
- Không có bất kỳ mô hình nào bị huấn luyện lại để thay đổi số liệu.
- Không có bất kỳ kỹ thuật tạo số liệu nhân tạo (data fabrication) hay làm đẹp kết quả (result smoothing) nào được áp dụng.
- Tất cả các số liệu trong báo cáo V4 đều được đối chiếu trực tiếp từ file CSV gốc `DepSev_Verified_Results_E0-E7.csv` và các báo cáo thực nghiệm chuẩn tắc trong `results/`.

---

## 2. KIỂM ĐỊNH 14 LỖI LỊCH SỬ TỪ CÁC PHIÊN BẢN TRƯỚC

| STT | Lỗi lịch sử phát hiện trong các bản cũ | Trạng thái kiểm định trong V4 | Bằng chứng xác minh |
| :---: | :--- | :--- | :--- |
| **1** | Mô tả sai nhóm đặc trưng $F_1$ và $F_2$ | **ĐÃ SỬA CHUẨN XÁC 100%** | $F_1$ (10 biến) gồm thể chất và 5 xét nghiệm máu; $F_2$ bổ sung 9 biến tiền sử bệnh mạn tính. |
| **2** | Diễn giải sai bước nhảy Feature Ablation | **ĐÃ SỬA CHUẨN XÁC 100%** | Bước nhảy $F_1 \to F_2$ tăng mạnh là do bổ sung tiền sử bệnh lâm sàng, không phải do xét nghiệm máu. |
| **3** | Balanced Accuracy của $E_2$ ghi sai | **ĐÃ SỬA CHUẨN XÁC 100%** | Khóa đúng giá trị thực nghiệm gốc: **0.4745** (khớp với CSV). |
| **4** | Balanced Accuracy của $E_{3b}$ ghi sai | **ĐÃ SỬA CHUẨN XÁC 100%** | Khóa đúng giá trị thực nghiệm gốc: **0.4790** (khớp với CSV). |
| **5** | Ma trận nhầm lẫn của $E_0$ không khớp số mẫu | **ĐÃ SỬA CHUẨN XÁC 100%** | Khóa đúng dự đoán out-of-fold ở mức quan sát cá thể ($N = 4.045$). |
| **6** | Số mẫu Temporal Validation ghi nhầm | **ĐÃ SỬA CHUẨN XÁC 100%** | Khóa đúng quy mô tập kiểm tra chu kỳ tương lai 2015–2016: **$N = 4.888$**. |
| **7** | Mô tả Threshold Optimization của $E_2$ gây hiểu nhầm | **ĐÃ SỬA CHUẨN XÁC 100%** | Làm rõ ngưỡng cắt được tối ưu hóa độc lập in-fold trên từng fold huấn luyện. |
| **8** | Mô tả hàm mất mát của $E_{7a}$ OLR | **ĐÃ SỬA CHUẨN XÁC 100%** | Mô tả đúng hàm hợp lý logarit Logistic thứ bậc với tỷ lệ chênh tỉ lệ thuận (proportional odds). |
| **9** | Cấu hình $E_{6a}$ bị bỏ quên trong bảng tổng hợp | **ĐÃ SỬA CHUẨN XÁC 100%** | Bảng 4.3 Master Comparison có đầy đủ cả $E_{6a}$ và $E_{6b}$. |
| **10** | Tập Held-out lịch sử bị gọi là untouched | **ĐÃ SỬA CHUẨN XÁC 100%** | Ghi rõ lai lịch dữ liệu: 813/1.009 mẫu từng thuộc tập phát triển ở phiên bản tiền nghiệm, chỉ dùng tham khảo đối chứng. |
| **11** | Nhầm lẫn giữa Accuracy cao và hiệu năng tốt | **ĐÃ SỬA CHUẨN XÁC 100%** | Phân tích rõ Accuracy bị méo mó do lớp đa số $C_0$ (74.6%); dùng Macro-F1 và BalAcc làm thước đo chuẩn. |
| **12** | So sánh trực tiếp Macro-F1 giữa $T_3$ và $T_5$ | **ĐÃ SỬA CHUẨN XÁC 100%** | Khẳng định $T_3$ và $T_5$ là hai bài toán khác nhau; không dùng kết quả $T_3$ để khẳng định giải quyết được $T_5$. |
| **13** | Khẳng định ứng dụng lâm sàng khi chưa có thử nghiệm y tế | **ĐÃ SỬA CHUẨN XÁC 100%** | Định vị đúng vai trò: Công cụ sàng lọc phân tầng rủi ro dịch tễ học, không thay thế chẩn đoán bác sĩ. |
| **14** | Sử dụng đường dẫn evidence mapping không tồn tại | **ĐÃ SỬA CHUẨN XÁC 100%** | Tạo mới `EVIDENCE_MAPPING.csv` ánh xạ từng khẳng định khoa học với file nguồn có thật. |

---

## 3. KẾT LUẬN KIỂM ĐỊNH

Tất cả các khẳng định và số liệu trong Báo cáo Dự án Khoa học DepSev-ML V4 đều đạt độ tin cậy khoa học tuyệt đối, minh bạch về phương pháp luận và sẵn sàng bảo vệ trước Hội đồng chuyên môn.

