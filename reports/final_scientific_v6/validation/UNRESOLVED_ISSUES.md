# DANH MỤC CÁC VẤN ĐỀ TỒN TẠI VÀ ĐIỀU KIỆN BIÊN CỦA DỮ LIỆU (UNRESOLVED ISSUES)
**Dự án:** DepSev-ML (Báo cáo Khoa học V6)  
**Ngày lập:** 30/10/2026  

---

## 1. Giới hạn bản chất của tập dữ liệu NHANES
Mặc dù các lỗi số liệu và mâu thuẫn phương pháp luận trong Báo cáo V5 đã được khắc phục hoàn toàn trong V6, nghiên cứu vẫn chịu sự chi phối của những rào cản khách quan nội tại từ miền dữ liệu:

1. **Bản chất khảo sát cắt ngang (Cross-Sectional Limitation):**
   - Dữ liệu CDC NHANES thu thập các chỉ số tại một thời điểm duy nhất trên từng đối tượng tham gia khảo sát.
   - Không có dữ liệu theo dõi dọc (longitudinal monitoring), do đó mô hình không thể quan sát được sự biến thiên động của các dấu hiệu cận lâm sàng trước và sau các cơn trầm cảm khởi phát.

2. **Tính gián tiếp của các chỉ dấu sinh lý ngoại vi (Peripheral Proxies):**
   - Các xét nghiệm sinh hóa máu (Glucose, Cholesterol, Triglycerides, Albumin) và chỉ số thể chất (BMI, Huyết áp) phản ánh rối loạn chuyển hóa toàn thân thứ phát, không phải là chỉ dấu sinh học trực tiếp của hoạt động dẫn truyền thần kinh trung ương (như Serotonin, Dopamine tại synapse não bộ).
   - Do đó, hiệu năng phân loại bão hòa ở mức Macro-F1 $\approx 0.27$ trên 5 lớp phẳng là phản ánh trung thực rào cản thông tin y sinh học, không thể khắc phục đơn thuần bằng việc tăng cỡ mẫu hay thay đổi thuật toán.

3. **Tính chủ quan của thang đo tự báo cáo PHQ-9:**
   - Điểm số PHQ-9 dựa trên việc tự điền bảng hỏi của người tham gia, có thể chịu ảnh hưởng bởi định kiến xã hội (stigma), khả năng tự nhận thức hoặc trạng thái tâm lý nhất thời trong ngày khảo sát.
   - Điểm số $0$--$4$ trong nghiên cứu này quy ước là nhóm "triệu chứng tối thiểu" (Minimal symptoms) trong sàng lọc cộng đồng, không đồng nghĩa với việc đối tượng hoàn toàn miễn nhiễm với các rối loạn tâm thần.

---

## 2. Các rào cản kỹ thuật đã được phân tích minh bạch trong Báo cáo V6

1. **Sự sụp đổ của lớp triệu chứng nhẹ $C_1$ ($s \in [5, 9]$):**
   - Do phân phối các chỉ số sinh lý của nhóm $C_1$ gần như trùng khít hoàn toàn với nhóm bình thường $C_0$, cả mô hình phẳng $T_5$ và các mô hình phân cấp đều có xu hướng gán nhầm trên 50% số mẫu $C_1$ thành $C_0$.
   - Khi áp dụng giải thuật phân loại nhị phân tích lũy ($E_{7b}$ CumBin) kết hợp cắt xác suất đơn điệu (Isotonic Clipping), xác suất của lớp $C_1$ bị ép về 0 tại các điểm vi phạm, khiến Recall của $C_1$ suy biến về $0.00\%$ (với LR) và $0.16\%$ (với RF). Báo cáo V6 đã chỉ rõ nghịch lý này thay vì che giấu.

2. **Sự đánh đổi không thể tránh khỏi trong bài toán nhạy chi phí ($E_{6b}$):**
   - Việc phạt nặng lỗi bỏ sót ca nặng ($\mathbf{C}(2, 0) \ge 3.0$) giúp giảm mạnh số ca nặng bị bỏ sót từ 120 ca xuống 62 ca ($M_1$), 19 ca ($M_2$), hoặc 7 ca ($M_3$).
   - Tuy nhiên, sự an toàn này phải đánh đổi bằng số lượng ca báo động nhầm từ lớp bình thường tăng vọt lên tương ứng 1.150 ca, 1.906 ca và 2.294 ca. Báo cáo V6 khẳng định đây là đặc trưng đánh đổi phân phối trong y tế dự phòng, cung cấp công cụ linh hoạt để chuyên gia y tế công cộng chủ động cân đối ngân sách sàng lọc.

3. **Sự suy thoái của mô hình phi tuyến khi kiểm định dòng thời gian:**
   - Khi ngoại suy sang chu kỳ tương lai NHANES 2015–2016 ($N = 4.888$), các mô hình phi tuyến phức tạp và hàm chi phí cực đoan ($E_{6b}$ với $M_2$) bị sụt giảm độ chính xác tổng thể do trôi phân phối ranh giới.
   - Ngược lại, mô hình tuyến tính có cấu trúc thứ bậc như Hồi quy Logistic thứ bậc cân bằng ($E_{7a}$) thể hiện độ ổn định ngoại suy vượt trội (Balanced Accuracy = 0.5073 so với 0.5151 ở In-cycle CV).

---

## 3. Khuyến nghị cho các nghiên cứu tiếp nối
- Chuyển hướng sang tích hợp dữ liệu đa phương thức (Multimodal Fusion): kết hợp dữ liệu bảng NHANES với tín hiệu âm thanh giọng nói, biểu cảm khuôn mặt và văn bản tự sự.
- Thu thập dữ liệu cảm biến đeo liên tục (Wearable devices) như PPG, HRV, gia tốc kế vận động và chất lượng giấc ngủ để bổ sung thành phần chuỗi thời gian.
- Phát triển mạng nơ-ron học thứ bậc sâu (Deep Ordinal Neural Networks) với hàm mất mát tùy biến trực tiếp trên rủi ro Bayes.

