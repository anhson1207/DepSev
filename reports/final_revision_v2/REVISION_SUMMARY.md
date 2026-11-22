# REVISION SUMMARY: BÁO CÁO NGHIÊN CỨU KHOA HỌC DEPSEV-ML (V2)

**Dự án**: DepSev-ML – A Machine Learning Framework for Depression Severity Prediction from Physiological Data  
**Nhánh Git**: `report/final-scientific-revision`  
**Định dạng biên dịch**: Chuẩn khóa luận tốt nghiệp Đại học Công nghệ – ĐHQGHN (`uetgraduation.cls`)  
**Tập tin PDF hoàn thiện**: `reports/final_revision_v2/DepSev_ML_Final_Revised.pdf` (70 trang)  
**Ngày hoàn thành**: 29/10/2026  

---

## 1. Mục tiêu và Nguyên tắc sửa chữa
Đợt sửa chữa toàn diện này khắc phục triệt để các phát hiện kiểm định độc lập được ghi nhận trong `DepSev_Repository_Audit_VI.md`, tuân thủ nghiêm ngặt các nguyên tắc:
1. **Bảo tồn tuyệt đối kho mã nguồn và dữ liệu gốc**: Không sửa, xóa, ghi đè bất kỳ file nào trong `data/`, `results/`, `artifacts/`, `src/`, `tests/`, `experiments/` và các báo cáo lịch sử.
2. **Sự thật số liệu 100%**: Mọi số liệu trong bảng biểu, biểu đồ và văn bản được trích xuất trực tiếp từ các file kết quả thực nghiệm thực tế, đối chiếu khớp từng chữ số thập phân.
3. **Tính liêm chính và trung thực khoa học**: Loại bỏ hoàn toàn ngôn từ quảng bá quá mức ("đột phá", "trần tự nhiên", "giải quyết triệt để"), thẳng thắn thừa nhận các giới hạn dịch tễ học và đánh đổi lâm sàng.

---

## 2. Bảng đối chiếu sửa chữa các lỗi trọng yếu (P0 & P1)

| Mã lỗi | Phân loại | Vấn đề trong phiên bản trước | Sửa chữa và Chuẩn hóa trong Phiên bản V2 | Nguồn chứng cứ thực tế |
|---|---|---|---|---|
| **P0-1** | Nhóm đặc trưng F1--F4 | Ghi nhầm $F_1$ có tiền sử bệnh lý và $F_2$ mới bổ sung xét nghiệm máu; ghi nhầm XGBoost F4 Macro-F1 = 0.2476 | **Chuẩn hóa chính xác**: $F_1$ (10 biến: 5 thể chất/sinh tồn + 5 xét nghiệm máu HbA1c, Cholesterol, HDL, Creatinine, Albumin); $F_2$ (19 biến: $F_1$ + 9 tiền sử chẩn đoán lâm sàng); $F_3$ (29 biến: $F_2$ + 10 lối sống); $F_4$ (34 biến: $F_3$ + 5 nhân khẩu học). **XGBoost F4 Macro-F1 = 0.2707 $\pm$ 0.0428**. | `results/tables/ablation_feature_inventory_final_protocol.csv`, `feature_ablation_results_final_protocol.csv` |
| **P0-2** | Balanced Accuracy trong Bảng so sánh | Ghi sai BalAcc của $E_2$ thành 0.4902 và $E_{3b}$ thành 0.4856; bỏ quên $E_{6a}$ trong Bảng tổng hợp | **Đã sửa chính xác**: $E_2$ BalAcc = **0.4745**, $E_{3b}$ BalAcc = **0.4790**. **Đã bổ sung đầy đủ $E_{6a}$**: Macro-F1 = 0.4535, BalAcc = 0.4904, Acc = 56.91%, Recall $C_1$ = **40.92%** (cao nhất toàn bộ mô hình), Recall $C_2^+$ = 44.25%, $C_2^+ \to C_0$ = 62 ca. | `experiments/hierarchical_t3/outputs/model_comparison.csv`, `experiments/soft_hierarchical_t3/outputs/model_comparison.csv` |
| **P0-3** | Ma trận nhầm lẫn E0 ($T_3$) | Ghi nhầm 436 ca $C_1 \to C_0$ và 98 ca $C_1 \to C_2^+$ | **Đã sửa số liệu thực tế**: Trong 638 mẫu $C_1$, có **347 mẫu (54.39%)** bị đoán thành $C_0$, **104 mẫu (16.30%)** đoán đúng, và **187 mẫu (29.31%)** đoán thành $C_2^+$. | `results/tables/granularity_oof_predictions.csv` |
| **P0-4** | Cỡ mẫu Temporal Evaluation (Branch D) | Ghi $N = 4.630$ và ngụy tạo lý do "loại bỏ 258 mẫu thiếu" | **Đã sửa chính xác**: Tất cả các mô hình trên Branch D được đánh giá trên toàn bộ **$N = 4.888$** mẫu của chu kỳ NHANES 2015--2016, không có bất kỳ mẫu nào bị loại bỏ. | `experiments/ordinal_t3/outputs/multicycle_comparison.csv`, `data/processed_multicycle/cohort_2015_2016.csv` |
| **P0-5** | Tính xác thực của Chứng cứ | File `evidence_mapping.csv` cũ chứa 47 đường dẫn không tồn tại trên đĩa | **Xây dựng mới `EVIDENCE_VERIFICATION.csv`**: 100% đường dẫn là file có thực trên đĩa, có mã băm mật mã SHA-256 đối chiếu và phương pháp kiểm chứng rõ ràng. | Kho file thực tế trên hệ thống |
| **P0-6** | Lai lịch tập Held-out (Provenance) | Mô tả tập Held-out ($N = 1.009$) như một tập độc lập tuyệt đối từ đầu | **Minh bạch hóa học thuật**: Ghi chú rõ 813 đối tượng từng thuộc tập phát triển trong các bản tách trước đây (`historically_untouched=false`), tập này chỉ dùng làm đối chứng tham khảo; toàn bộ chứng cứ khoa học cốt lõi dựa trên 5-Fold Stratified CV trên tập phát triển ($N = 4.045$). | `README.md` |
| **P1-7** | Tối ưu hóa ngưỡng cắt $E_2$ | Ghi ngưỡng cố định $\tau_1 = 0.40, \tau_2 = 0.50$ | **Làm rõ quy trình thực tế**: Ngưỡng được tối ưu thích ứng trên từng outer fold qua Inner CV: Fold 1 (0.50/0.50), Fold 2 (0.50/0.55), Fold 3 (0.45/0.55), Fold 4 (0.45/0.55), Fold 5 (0.45/0.50), trung bình là $\tau_1 = 0.47, \tau_2 = 0.53$. | `experiments/hierarchical_t3/outputs/threshold_optimization.csv` |
| **P1-8** | Hàm mục tiêu tối ưu của $E_{7a}$ | Diễn giải rằng thuật toán tối ưu hóa trực tiếp hàm mất mát Class MAE | **Đính chính cơ sở lý thuyết**: $E_{7a}$ tối ưu hóa hàm **Negative Log-Likelihood với điều chuẩn $L_2$** trên các trọng số $\mathbf{w}$ của mô hình logit tích lũy; Class MAE là chỉ số đánh giá hình học hậu kỳ (post-hoc metric). Hiện tượng triệt tiêu $C_1$ ở bản unweighted là do Maximum Likelihood dưới mất cân bằng dồn xác suất về $C_0$. | `src/ordinal_models.py` |
| **P1-9** | Ngôn từ khẳng định quá mức | Dùng các từ "đột phá", "giải quyết triệt để", "trần tự nhiên", "an toàn lâm sàng" | **Điều chỉnh toàn bộ văn phong**: Giữ giọng văn khoa học khách quan, trung lập; phân tích rõ các đánh đổi và giới hạn y sinh học. | Toàn văn các chương 1, 4, 5 |
| **P1-10** | Thiếu sót mô hình trong Bảng tổng hợp | Bảng 4.6 thiếu mô hình $E_{6a}$ và mô hình Majority-class baseline | **Đã tích hợp đầy đủ**: Bảng 4.6 có đủ 10 mô hình ($E_0, E_1, E_2, E_{3a}, E_{3b}, E_4, E_5, E_{6a}, E_{6b}, E_{7a\text{ unw}}, E_{7a\text{ bal}}, E_{7b\text{ LR}}, E_{7b\text{ RF}}$) cùng Majority Baseline ($C_0$). | `tables/table_master_comparison.tex` |
| **P1-12** | Khả năng tái lập và Trọng số khảo sát | Chưa làm rõ trạng thái tái lập và bỏ qua thiết kế phức hợp của NHANES | **Tài liệu hóa chi tiết**: Bổ sung phân tích về Complex Survey Design (chưa có trọng số trong tiêu chuẩn ML benchmark) và kiểm kê các script có thể chạy độc lập (`src/09` -- `src/14`). | Chương 3 (Mục 3.5), Chương 5 (Mục 5.3), `REPRODUCIBILITY_NOTES.md` |
| **P1-18** | Thông tin cán bộ hướng dẫn | Sử dụng chuỗi giả định "TS. Giảng viên Hướng dẫn" | **Chuẩn hóa quy cách**: Thay thế bằng định dạng khung giữ chỗ chuẩn `[Họ và tên Cán bộ Hướng dẫn]` mà không ngụy tạo danh tính. | `main.tex` |

---

## 3. Danh mục các tập tin chuyển giao trong `reports/final_revision_v2/`

```text
reports/final_revision_v2/
├── DepSev_ML_Final_Revised.pdf          # File PDF khóa luận hoàn chỉnh 70 trang chuẩn VNU-UET
├── main.pdf                             # Bản PDF biên dịch gốc từ Tectonic
├── REVISION_SUMMARY.md                  # Bản tóm tắt sửa chữa và đối chiếu khoa học (file này)
├── SCIENTIFIC_AUDIT.md                  # Báo cáo kiểm định chất lượng nội bộ và tự đánh giá điểm
├── REPRODUCIBILITY_NOTES.md             # Đánh giá độ tin cậy và hướng dẫn tái lập thực nghiệm
├── EVIDENCE_VERIFICATION.csv            # Bản đồ 13 chứng cứ khoa học với mã SHA-256 thực tế
├── run_manifest.json                    # Metadata thông số chạy của toàn bộ quá trình sửa chữa
├── build_artifacts.py                   # Script Python tự động sinh bảng LaTeX và trích xuất số liệu
├── patch_chapter4.py                    # Script Python tự động áp dụng các bản vá khoa học cho Chương 4
├── validation/
│   ├── latex_build_log.txt              # Log biên dịch Tectonic chi tiết (0 lỗi, 0 undefined ref)
│   ├── numerical_checks.csv             # Bảng kiểm tra 20 chỉ số đối chiếu PASS 100%
│   ├── figure_table_checks.md           # Kiểm tra trạng thái 10 bảng và 9 hình vẽ
│   └── unresolved_issues.md             # Tài liệu hóa nợ kỹ thuật và các vấn đề mở
└── source/                              # Toàn bộ mã nguồn LaTeX đã được tinh chỉnh
    ├── main.tex                         # File LaTeX chủ đạo (khai báo, bìa, lời cảm ơn, tóm tắt)
    ├── uetgraduation.cls                # Document class chuẩn của Đại học Công nghệ
    ├── sizes.clo, UET logo.pdf          # Các tài nguyên định dạng và logo trường
    ├── references.bib                   # Cơ sở dữ liệu tài liệu tham khảo BibTeX
    ├── chapters/
    │   ├── chapter1.tex                 # Mở đầu, phát biểu bài toán, RQ1--RQ5, đóng góp
    │   ├── chapter2.tex                 # Cơ sở lý thuyết và tổng quan nghiên cứu liên quan
    │   ├── chapter3.tex                 # Khung phương pháp luận, F1--F4, chống rò rỉ
    │   ├── chapter4.tex                 # Kết quả thực nghiệm và thảo luận chuyên sâu
    │   └── chapter5.tex                 # Kết luận, đóng góp, hạn chế và hướng phát triển
    ├── tables/                          # 8 bảng số liệu LaTeX sinh tự động từ CSV gốc
    │   ├── table_feature_inventory.tex  # Bảng 3.3 (34 biến F1--F4)
    │   ├── table_ablation_results.tex   # Bảng 4.1 (Feature Ablation F1--F4)
    │   ├── table_granularity_results.tex# Bảng 4.2 (T2--T5 Granularity)
    │   ├── table_hard_hierarchical.tex  # Bảng 4.3 (E0--E3b Hard Hierarchical)
    │   ├── table_soft_hierarchical.tex  # Bảng 4.4 (E4--E6b Soft Cost-Sensitive)
    │   ├── table_ordinal_results.tex    # Bảng 4.5 (E7a--E7b Ordinal)
    │   ├── table_master_comparison.tex  # Bảng 4.6 (Master Table E0--E7 + Baseline)
    │   └── table_multicycle_results.tex # Bảng 4.7 (Branch D Temporal Evaluation)
    └── figures/                         # 34 tập tin hình ảnh đồ thị thực nghiệm độ phân giải cao
```

