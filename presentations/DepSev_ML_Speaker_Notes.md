# DepSev-ML — Speaker Notes

Người trình bày: **Nguyễn Anh Sơn**  
Thời lượng mục tiêu: **15–20 phút** cho 17 slide chính; appendix chỉ dùng khi hỏi đáp.  
Lưu ý chung: đây là nghiên cứu dự đoán **mức độ triệu chứng**, không phải công cụ chẩn đoán lâm sàng. Các kết quả thể hiện association/predictive signal, không chứng minh quan hệ nhân quả.

## Slide 1 — DepSev-ML (0:30)

**Mục đích trình bày**  
Giới thiệu đề tài và phạm vi của báo cáo tiến độ.

**Nội dung cần nói**  
“Em là Nguyễn Anh Sơn. DepSev-ML nghiên cứu một framework học máy để dự đoán mức độ triệu chứng trầm cảm từ dữ liệu NHANES. Báo cáo hôm nay tập trung vào quá trình xây dựng protocol, lựa chọn mô hình, ảnh hưởng của target granularity và mở rộng multi-cycle.”

**Ý nghĩa thực nghiệm**  
Đặt đúng phạm vi: prediction research trên dữ liệu khảo sát cắt ngang, không phải chẩn đoán hoặc khẳng định hiệu quả lâm sàng.

**Câu chuyển tiếp**  
“Trước hết, em bắt đầu từ câu hỏi vì sao bài toán severity prediction cần được xem xét riêng.”

**Nguồn**  
`results/final_report_integrated.md`

## Slide 2 — Từ screening nhị phân đến severity prediction (0:55)

**Mục đích trình bày**  
Nêu động cơ nghiên cứu và câu hỏi trung tâm.

**Nội dung cần nói**  
“Một quyết định nhị phân có thể che khuất cấu trúc ordinal của PHQ-9. Trong cùng framework, em hỏi hiệu năng thay đổi ra sao khi target tăng từ 2 lên 5 lớp. Đồng thời, em kiểm tra liệu thêm dữ liệu từ nhiều cycle có giúp các lớp hiếm hay không.”

**Ý nghĩa thực nghiệm**  
Tách hai vấn đề: độ khó do chi tiết của target và độ khó do dữ liệu/lớp hiếm.

**Câu chuyển tiếp**  
“Để trả lời có kiểm soát, project được tổ chức thành một chuỗi thí nghiệm có provenance rõ ràng.”

**Nguồn**  
`results/final_report_granularity.md`; `results/multicycle/multicycle_experiment_summary.md`

## Slide 3 — Khung nghiên cứu theo giai đoạn (0:50)

**Mục đích trình bày**  
Cho thấy tiến trình từ cohort đến robustness analysis.

**Nội dung cần nói**  
“Pipeline gồm năm giai đoạn: cohort và feature inventory; feature ablation; nested CV và ensemble; target granularity; cuối cùng là harmonization và controlled multi-cycle. Các thí nghiệm cũ được giữ lại như bằng chứng cho quá trình xây dựng protocol.”

**Ý nghĩa thực nghiệm**  
Mỗi giai đoạn trả lời một câu hỏi khác nhau; không trộn kết quả exploratory với bằng chứng cuối của từng protocol.

**Câu chuyển tiếp**  
“Nguồn dữ liệu xuyên suốt là NHANES và target được tạo từ PHQ-9.”

**Nguồn**  
`results/final_report_integrated.md`, sections 2–11

## Slide 4 — NHANES: nhiều module, một participant key (0:55)

**Mục đích trình bày**  
Giải thích cấu trúc dữ liệu và cách liên kết module.

**Nội dung cần nói**  
“NHANES cung cấp demographic, questionnaire, examination, laboratory, dietary và medication data. Các bảng được merge theo SEQN. Cohort primary áp dụng tuổi từ 20 trở lên và yêu cầu đủ chín PHQ-9 items hợp lệ.”

**Ý nghĩa thực nghiệm**  
Age eligibility và routing logic phải được xử lý trước modeling để tránh missingness trở thành proxy ngoài ý muốn.

**Câu chuyển tiếp**  
“Sau merge, PHQ-9 chỉ tạo target; predictors đến từ bốn nhóm P, C, L và D.”

**Nguồn**  
`artifacts/nhanes_official_source_catalog.json`; `results/final_report_integrated.md`, section 3

## Slide 5 — PHQ-9 và feature groups (1:00)

**Mục đích trình bày**  
Định nghĩa target C0–C4 và cấu trúc feature set.

**Nội dung cần nói**  
“PHQ9_total được chia C0 đến C4 theo các ngưỡng 0–4, 5–9, 10–14, 15–19 và 20–27. Chín DPQ items không được đưa vào predictors. F4 ban đầu có 34 features thuộc physiological, clinical, lifestyle và demographic.”

**Ý nghĩa thực nghiệm**  
Ngăn target leakage trực tiếp từ các câu hỏi tạo nhãn.

**Câu chuyển tiếp**  
“Các feature này được đưa vào preprocessing leakage-safe và đánh giá bằng các folds cố định.”

**Nguồn**  
`artifacts/final_model_spec.json`; `results/tables/feature_inventory.csv`

## Slide 6 — Protocol đánh giá leakage-safe (1:00)

**Mục đích trình bày**  
Trình bày mô hình, preprocessing và metric.

**Nội dung cần nói**  
“Numerical features dùng median imputation và scaling; categorical features dùng imputation phù hợp rồi one-hot encoding. Preprocessor được fit trong từng training fold. Các baseline gồm LR, DT, RF và XGB. Primary metric là Macro-F1; Balanced Accuracy và per-class Recall/F1 bổ sung cách nhìn về imbalance.”

**Ý nghĩa thực nghiệm**  
Accuracy không đủ trong bài toán lệch lớp; toàn bộ lựa chọn mô hình được thực hiện trên development evidence.

**Câu chuyển tiếp**  
“Trước tuning, feature ablation cho biết incremental value của từng nhóm.”

**Nguồn**  
`results/ablation_report_final_protocol.md`; `artifacts/final_model_spec.json`

## Slide 7 — Feature ablation F1–F4 (1:05)

**Mục đích trình bày**  
Tóm tắt 16 thí nghiệm baseline trên cùng folds.

**Nội dung cần nói**  
“P-only cho Macro-F1 thấp. Clinical features tạo bước tăng rõ nhất. XGBoost baseline tăng dần đến F4 và đạt 0.2707, trong khi RF baseline không tăng đơn điệu. Điều này cho thấy incremental value phụ thuộc model family.”

**Ý nghĩa thực nghiệm**  
Không dùng một kết quả baseline để tuyên bố một feature group luôn tốt hoặc luôn xấu.

**Câu chuyển tiếp**  
“Sau đó nested CV được dùng để tách tuning khỏi outer evaluation.”

**Nguồn**  
`results/tables/feature_ablation_results_final_protocol.csv`

## Slide 8 — Model selection và ensemble (1:10)

**Mục đích trình bày**  
Giải thích vì sao RF/F4 được giữ làm final candidate.

**Nội dung cần nói**  
“Trong nested CV, RF/F4 tuned đạt 0.2755 ± 0.0270, cao hơn XGB/F4 và LR/F4. Ensemble experiment sau đó cho mean cao nhất ở Voting 1:2:1, nhưng mức tăng nhỏ và variance cao hơn. Theo conservative rule đã chốt trước, RF/F4 được giữ vì ổn định và đơn giản hơn.”

**Ý nghĩa thực nghiệm**  
Selection rule không mặc định ensemble tốt hơn và không sử dụng held-out để thay đổi model.

**Câu chuyển tiếp**  
“Với model đã frozen, câu hỏi tiếp theo là target granularity ảnh hưởng hiệu năng thế nào.”

**Nguồn**  
`results/tables/nested_cv_tuning_results.csv`; `results/tables/ensemble_results.csv`; `artifacts/final_model_spec.json`

## Slide 9 — Target granularity T2–T5 (1:15)

**Mục đích trình bày**  
Trả lời câu hỏi trung tâm bằng cùng model, data và folds.

**Nội dung cần nói**  
“T2 là minimal symptoms versus symptomatic, không phải healthy versus depressed. Khi tăng từ T2 đến T5, Macro-F1 giảm 0.6639 → 0.4354 → 0.3380 → 0.2630; Balanced Accuracy giảm 0.6659 → 0.4667 → 0.3661 → 0.2805. Bước giảm lớn nhất là T2 sang T3, Δ Macro-F1 bằng −0.2285. Macro-F1 SD theo T2–T5 lần lượt là 0.0113, 0.0075, 0.0177 và 0.0222.”

**Ý nghĩa thực nghiệm**  
Cùng framework nhưng phân biệt các mức severity lân cận khó hơn rõ rệt.

**Câu chuyển tiếp**  
“Confusion và overlap audit cho biết lỗi tập trung ở đâu, nhưng không được diễn giải như quan hệ nhân quả.”

**Nguồn**  
`results/tables/granularity_results.csv`; `results/final_report_granularity.md`

## Slide 10 — Lỗi chủ yếu giữa các mức lân cận (1:15)

**Mục đích trình bày**  
Giải thích pattern lỗi và ordinal behaviour.

**Nội dung cần nói**  
“Ở T3, 347 trên 638 mild cases bị dự đoán minimal, gợi ý current feature space khó tách mild khỏi minimal. Ở T4 chỉ 42 trên 148 trường hợp 15–27 được dự đoán đúng; ở T5, 40 trên 104 C3 chuyển sang C2 và chỉ 1 trên 44 C4 đúng. Tuy vậy within-one-class accuracy ở T3–T5 lần lượt 0.8952, 0.8690 và 0.8667.”

**Ý nghĩa thực nghiệm**  
Exact-class performance thấp nhưng phần lớn dự đoán vẫn gần lớp thật; ordinal metric phụ thuộc cách gộp lớp nên không hoàn toàn cùng đơn vị giữa tasks.

**Câu chuyển tiếp**  
“Để kiểm tra liệu thêm severe samples có thay đổi bức tranh, project được mở rộng sang ba cycle.”

**Nguồn**  
`results/figures/granularity_T5_confusion.png`; `results/figures/overlap_adjacent_classes.png`; `results/final_report_granularity.md`

## Slide 11 — Harmonization ba cycle và F4_common31 (1:05)

**Mục đích trình bày**  
Giới thiệu cohort multi-cycle và common-core features.

**Nội dung cần nói**  
“Ba cycle 2011–2012, 2013–2014 và 2015–2016 được build riêng rồi mới concat. Pooled cohort có N bằng 14.581; so với official 2013–2014, C3 tăng từ 132 lên 352 và C4 từ 55 lên 167. Để giữ comparability, SLD010H, PAQ605 và PAD680 bị loại, tạo F4_common31.”

**Ý nghĩa thực nghiệm**  
Số lượng lớp hiếm tăng về tuyệt đối, nhưng imbalance ratio không mặc nhiên biến mất.

**Câu chuyển tiếp**  
“Trên feature space chung này, bốn thiết kế A/B/C/D được dùng để phân biệt hiệu ứng pooling, sample size và thời gian.”

**Nguồn**  
`results/final_report_integrated.md`; `results/tables/multicycle_class_distribution.csv`; `artifacts/common31_harmonization_policy.json`

## Slide 12 — Controlled multi-cycle design A/B/C/D (1:10)

**Mục đích trình bày**  
Làm rõ bốn nhánh thực nghiệm và giới hạn so sánh.

**Nội dung cần nói**  
“A là official single-cycle control. B là pooled ba cycle. C dùng mười matched samples với đúng N và T5 counts của A nhưng lấy từ hai cycle ngoài control. D train hai cycle trước và đánh giá 2015–2016. RF config, F4_common31, preprocessing và targets được giữ cố định.”

**Ý nghĩa thực nghiệm**  
A nằm trong B; các sample C có thể overlap; D là temporal evaluation chứ không phải untouched external validation.

**Câu chuyển tiếp**  
“Kết quả A so với B cho thấy thêm dữ liệu không tạo improvement đồng nhất ở mọi metric.”

**Nguồn**  
`artifacts/multicycle/experiment_config.json`

## Slide 13 — Single-cycle A so với pooled B (1:10)

**Mục đích trình bày**  
So sánh trực tiếp Macro-F1 và Balanced Accuracy.

**Nội dung cần nói**  
“Macro-F1 B trừ A gần zero ở T2, T3 và T5, đồng thời giảm 0.0159 ở T4. Ngược lại Balanced Accuracy tăng ở cả bốn tasks; riêng T5 tăng từ 0.2841 lên 0.3076. Vì hai metric phản ánh các khía cạnh khác nhau, em không gọi pooling là thắng tổng thể.”

**Ý nghĩa thực nghiệm**  
Pooling có thể phân bố lại recall giữa lớp mà không nâng harmonic mean per-class một cách nhất quán.

**Câu chuyển tiếp**  
“Điều quan trọng hơn là kiểm tra trực tiếp C3 và C4.”

**Nguồn**  
`results/multicycle/tables/comparison_summary.csv`; `results/multicycle/tables/cv_fold_metrics.csv`

## Slide 14 — Rare-class behaviour C3/C4 (1:10)

**Mục đích trình bày**  
Đánh giá severe classes thay vì chỉ nhìn aggregate metric.

**Nội dung cần nói**  
“Pooling làm recall C3 tăng từ 0.1515 lên 0.2869, nhưng precision giảm từ 0.0922 xuống 0.0721 nên C3 F1 gần như không đổi: 0.1146 lên 0.1152. Với C4, recall tăng 0.0545 lên 0.1377 và F1 tăng 0.0600 lên 0.0962, nhưng mức tuyệt đối vẫn thấp. Thêm severe samples chưa giải quyết được bài toán rare class.”

**Ý nghĩa thực nghiệm**  
Absolute N là một yếu tố, nhưng feature overlap và weak signal vẫn là các limitation hợp lý.

**Câu chuyển tiếp**  
“Matched-N và temporal evaluation kiểm tra độ bền của pattern này.”

**Nguồn**  
`results/multicycle/tables/rare_class_comparison.csv`; `results/multicycle/tables/per_class_metrics.csv`

## Slide 15 — Matched-N và temporal robustness (1:10)

**Mục đích trình bày**  
Đặt các nhánh A/C/B/D cạnh nhau nhưng không đánh đồng estimand.

**Nội dung cần nói**  
“Ở T5, A đạt Macro-F1 0.2634, C đạt mean 0.2700 qua mười samples, B đạt 0.2619 và D đạt 0.2676. Tất cả vẫn cho thấy degradation khi tăng granularity. Các con số không phải bốn test scores tương đương: C là phân bố sample-level CV, còn D là một temporal evaluation.”

**Ý nghĩa thực nghiệm**  
Pattern fine-grained difficulty không phụ thuộc hoàn toàn vào một cycle hay một cỡ mẫu duy nhất.

**Câu chuyển tiếp**  
“Từ toàn bộ chuỗi thí nghiệm, có thể rút ra bốn phát hiện chính với mức diễn giải thận trọng.”

**Nguồn**  
`results/multicycle/tables/matched_n_metrics.csv`; `results/multicycle/tables/temporal_metrics.csv`

## Slide 16 — Bốn phát hiện chính (1:05)

**Mục đích trình bày**  
Tổng hợp kết quả mà không overclaim.

**Nội dung cần nói**  
“Thứ nhất, coarse-grained tasks dễ hơn fine-grained tasks với framework hiện tại. Thứ hai, Accuracy không đủ; T2 accuracy gần majority proportion. Thứ ba, feature distributions overlap mạnh và các association nhìn chung yếu đến trung bình. Thứ tư, imbalance và overlap đều là plausible contributing factors; experiment chưa tách riêng contribution của chúng.”

**Ý nghĩa thực nghiệm**  
Không nói overlap, correlation hoặc SHAP là causal effect; không kết luận 5-class thấp đồng nghĩa dataset không phù hợp.

**Câu chuyển tiếp**  
“Em kết thúc bằng đóng góp, giới hạn và bước nghiên cứu tiếp theo.”

**Nguồn**  
`results/final_report_granularity.md`; `results/multicycle/multicycle_experiment_summary.md`

## Slide 17 — Kết luận và hướng tiếp theo (0:55)

**Mục đích trình bày**  
Khép lại bằng câu trả lời trực tiếp cho câu hỏi nghiên cứu.

**Nội dung cần nói**  
“RF/F4 phân biệt tốt hơn đáng kể ở target coarse-grained so với năm lớp. Degradation lớn nhất xuất hiện khi mild được tách khỏi minimal. Multi-cycle làm tăng severe-class N và cải thiện C3, nhưng không tạo gain nhất quán cho Macro-F1 hoặc C4. Bước tiếp theo nên tập trung vào ordinal learning, survey-weight-aware analysis, calibration và external replication.”

**Ý nghĩa thực nghiệm**  
Đóng góp chính là framework kiểm soát và evidence về target granularity, không phải tuyên bố mô hình đã sẵn sàng dùng lâm sàng.

**Câu chuyển tiếp**  
“Em xin cảm ơn và sẵn sàng trao đổi; các slide appendix chứa chi tiết bổ sung.”

**Nguồn**  
`results/final_report_integrated.md`

---

## Appendix A — RF specification và model selection

**Mục đích trình bày**  
Trả lời câu hỏi về cấu hình RF và conservative selection rule.

**Nội dung cần nói**  
“Final RF dùng 150 trees, max depth 8, min samples split 10, min samples leaf 8, max features 0.5 và balanced_subsample. Specification được frozen trước supplementary held-out evaluation. Held-out có historically_untouched=false nên không được gọi là independent generalization estimate.”

**Nguồn**  
`artifacts/final_model_spec.json`; `results/tables/ensemble_results.csv`

## Appendix B — T5 confusion matrices A/B/C

**Mục đích trình bày**  
Cho phép xem trực tiếp pattern dự đoán ở three controlled settings.

**Nội dung cần nói**  
“Các confusion matrices cho thấy diagonal mạnh nhất ở C0 và nhầm lẫn chủ yếu giữa các mức lân cận. B cải thiện C3 nhưng C4 vẫn rất ít true positives. C là aggregate của matched-sample experiment nên cần đọc cùng protocol, không xem như một held-out test mới.”

**Nguồn**  
`results/figures/granularity_T5_confusion.png`; `results/multicycle/figures/A_T5_confusion.png`; `results/multicycle/figures/B_T5_confusion.png`

## Appendix C — Explainability và harmonization safeguards

**Mục đích trình bày**  
Trả lời câu hỏi về feature importance và độ tin cậy khi ghép cycle.

**Nội dung cần nói**  
“Permutation importance và SHAP chỉ mô tả behavior của final RF; không phải causal effect. Harmonization dùng F4_common31, giữ routing-aware derived features, top-code alcohol quantity chung và chuẩn hóa XPT near-zero ở processing layer mà không sửa raw files.”

**Nguồn**  
`results/tables/permutation_importance.csv`; `results/tables/shap_global_importance.csv`; `artifacts/common31_harmonization_policy.json`

