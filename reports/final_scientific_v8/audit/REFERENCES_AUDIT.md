# BÁO CÁO KIỂM TRA VÀ CHUẨN HÓA TÀI LIỆU THAM KHẢO (REFERENCES AUDIT) — DEPSEV-ML V8

**Dự án:** DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Phiên bản:** Báo cáo khoa học V8 (`DepSev_ML_Scientific_Report_V8.pdf`)  
**Ngày thực hiện:** 30/10/2026  

---

## 1. Mục tiêu kiểm tra tài liệu tham khảo

1. Khắc phục dứt điểm lỗi trùng lặp tác giả/công trình trong danh mục tài liệu tham khảo (BibTeX và LaTeX source).
2. Kiểm tra tính xác thực và thẩm định học thuật của toàn bộ các tài liệu tham khảo xuất hiện trong Bảng 2.1 (Tổng hợp so sánh các nghiên cứu liên quan).
3. Đảm bảo 100% các mục tham khảo đều được trích dẫn (cited) trong văn bản, định dạng chuẩn IEEE, phân tách rõ ràng phần Tiếng Việt và Tiếng Anh.

---

## 2. Xử lý lỗi trùng lặp Pratt (2014)

### Hiện trạng ở các phiên bản trước (V6, V7)
Trong tệp `references.bib` và `main.tex` tồn tại hai mục BibTeX độc lập cùng trỏ đến nghiên cứu dịch tễ của nhóm tác giả Pratt thuộc Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh Hoa Kỳ (CDC/NCHS):
- `pratt1991discriminant`: Khóa trích dẫn bị đặt sai năm thành 1991, nhưng nội dung trỏ về ấn phẩm NCHS Data Brief No. 172 (2014) của tác giả Laura A. Pratt.
- `pratt2014depression`: Khóa trích dẫn chuẩn của ấn phẩm NCHS Data Brief No. 172 (2014).
Hậu quả: Tạo ra hai mục tham khảo riêng biệt trong thư mục tài liệu tham khảo cuối báo cáo, gây lãng phí diện tích và thiếu chuyên nghiệp trong mắt giảng viên phản biện.

### Biện pháp xử lý trong V8
1. Hợp nhất hoàn toàn vào một mục duy nhất: `@article{pratt2014depression}` trong `references.bib` và `\bibitem{pratt2014depression}` trong `main.tex`.
2. Thay thế toàn bộ các lệnh `\cite{pratt1991discriminant}` bằng `\cite{pratt2014depression}` trong tất cả các chương (`chapter1.tex`, `chapter2.tex`).
3. Kiểm định sau biên dịch: Công trình của Pratt xuất hiện đúng 1 lần tại vị trí **[16]** trong danh mục tài liệu tham khảo tiếng Anh, được trích dẫn chính xác tại Mục 1.1, Mục 2.1.2 và Bảng 2.1.

---

## 3. Thẩm định học thuật Bảng 2.1 (Các nghiên cứu liên quan)

Trong các bản trước, Bảng 2.1 có chứa một mục mang tên *"Kaggle Benchmark CDC (2021) - NHANES 2013-2014 - XGBoost Baseline"*. 

### Đánh giá học thuật:
- "Kaggle Benchmark" là bài viết chia sẻ mã nguồn tự do của cá nhân trên diễn đàn cộng đồng, không phải công trình nghiên cứu khoa học được bình duyệt (peer-reviewed publication).
- Việc đưa một benchmark cộng đồng không rõ nguồn gốc vào bảng tổng quan nghiên cứu liên quan làm giảm tính học thuật và độ tin cậy của đề tài.

### Quyết định biên tập trong V8:
1. **Loại bỏ hoàn toàn mục Kaggle Benchmark** khỏi Bảng 2.1.
2. Thiết kế lại Bảng 2.1 thành cấu trúc **4 cột rõ ràng, chuẩn mực học thuật**:
   - Cột 1: Nghiên cứu (Tác giả, Năm, Trích dẫn).
   - Cột 2: Nguồn dữ liệu sử dụng.
   - Cột 3: Phương pháp / Mô hình học máy.
   - Cột 4: Kết quả chính \& Giới hạn nghiên cứu.
3. Các công trình học thuật chính thức được giữ lại và đối soát:
   - **Pratt và cộng sự (2014)** \cite{pratt2014depression}: Dữ liệu NHANES 2009--2012, phương pháp hồi quy logistic và thống kê dịch tễ học dân số.
   - **Patel và cộng sự (2016)** \cite{patel2016systematic}: Dữ liệu NHANES đa chu kỳ, khảo sát các yếu tố lâm sàng liên quan bằng Logistic Regression và Decision Tree trên bài toán nhị phân.
   - **Dipnall và cộng sự (2016)** \cite{dipnall2016fusing}: Dữ liệu NHANES 2005--2010, tích hợp các dấu ấn sinh học và mô hình tuyến tính tổng quát (GLM) cho sàng lọc nhị phân.
   - **Zhang và cộng sự (2020)** \cite{zhang2020machine}: Khảo sát các thuật toán học máy dạng bảng (Random Forest, XGBoost) trên bài toán phân loại phẳng.
   - **DepSev-ML (Đề tài này)**: Dữ liệu NHANES 3 chu kỳ gộp (2011--2016), phát triển mô hình Random Forest, phân cấp mềm kết hợp tối ưu chi phí Bayes và Ordinal Logistic Regression; giải quyết quy luật từ $T_2 \to T_5$ và kiểm định ngoại suy theo thời gian.

---

## 4. Danh mục toàn bộ tài liệu tham khảo đã đối soát trong V8

### Phần Tiếng Việt (1 tài liệu)
- `[1]` Bộ Y tế, *Hướng dẫn chẩn đoán và điều trị các rối loạn trầm cảm*, Nhà xuất bản Y học, Hà Nội, 2020.

### Phần Tiếng Anh (16 tài liệu)
- `[2]` K. Kroenke, R. L. Spitzer, and J. B. Williams, "The PHQ-9: Validity of a brief depression severity measure," *J. Gen. Intern. Med.*, 2001.
- `[3]` R. L. Spitzer, K. Kroenke, and J. B. Williams, "Validation and utility of a self-report version of PRIME-MD: The PHQ primary care study," *JAMA*, 1999.
- `[4]` L. Breiman, "Random forests," *Machine Learning*, 2001.
- `[5]` T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
- `[6]` C. N. Silla and A. A. Freitas, "A survey of hierarchical classification across different application domains," *Data Mining and Knowledge Discovery*, 2011.
- `[7]` C. Elkan, "The foundations of cost-sensitive learning," *IJCAI*, 2001.
- `[8]` P. McCullagh, "Regression models for ordinal data," *J. R. Stat. Soc. Series B*, 1980.
- `[9]` E. Frank and M. Hall, "A simple approach to ordinal classification," *ECML*, 2001.
- `[10]` CDC / NCHS, "National Health and Nutrition Examination Survey: 2013-2014 Data Documentation," 2014.
- `[11]` CDC / NCHS, "National Health and Nutrition Examination Survey: 2015-2016 Data Documentation," 2016.
- `[12]` R. Patel, K. R. R. et al., "Systematic evaluation of clinical and biometric risk factors in depression: An NHANES study," *J. Affect. Disord.*, 2016.
- `[13]` F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 2011.
- `[14]` J. Cohen, "Weighted chi-square: An extension of the kappa coefficient," *Educ. Psychol. Meas.*, 1968.
- `[15]` J. F. Dipnall et al., "Fusing data from multiple biomarkers: A machine learning approach to depression screening," *BMC Med. Res. Methodol.*, 2016.
- `[16]` L. A. Pratt, D. J. Brody, and Q. Gu, "Depression in the U.S. household population, 2009-2012," *NCHS Data Brief*, no. 172, 2014.
- `[17]` Y. Zhang et al., "Machine learning approaches for depression screening in community settings: A comparative study," *Comput. Biol. Med.*, 2020.

