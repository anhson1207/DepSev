# Ensemble experiment — development F4

Candidate theo selection rule đã chốt: **RF/F4**. Voting 1:2:1 có Macro-F1 mean
cao nhất nhưng không đạt điều kiện SD ≤ RF. Chưa refit final model, chưa chạy SHAP,
chưa dự đoán hoặc tính metric trên held-out.

## Protocol và cấu hình đã khóa

- Protocol: `original_split_preserved_age20`. Development N=4.045,
  C0–C4 lần lượt 3.016 / 638 / 243 / 104 / 44. Cohort, split và 34 features F4 giữ nguyên.
- Outer: StratifiedKFold(5, shuffle=True, random_state=42), khớp assignments đã lưu.
- Inner: StratifiedKFold(3, shuffle=True, random_state=42), chỉ trong outer training.
- LR: **consensus LR configuration for ensemble**, C=0.3, solver=newton-cg,
  L2, class_weight=balanced, max_iter=3000, random_state=42. C=0.3 là mode và median
  của best-C {0.3, 0.01, 1, 100, 0.3}. Không gọi đây là globally optimal hyperparameter.
- RF: n_estimators=150, max_depth=8, min_samples_split=10, min_samples_leaf=8,
  max_features=0.5, class_weight=balanced_subsample, random_state=42, n_jobs=1.
  Toàn bộ estimator parameters được lấy đúng từ `tuned_model_configs.json`.
- XGB baseline: n_estimators=200, max_depth=4, learning_rate=0.05,
  min_child_weight=1, subsample=0.8, colsample_bytree=0.8, gamma=0,
  reg_alpha=0, reg_lambda=1, random_state=42, n_jobs=1.
  Evidence development đã có: baseline 0.2707389 ± 0.0428162 so với tuned
  0.2620868 ± 0.0353179. Lựa chọn được khóa trước ensemble experiment.
- `FoldBalancedXGBClassifier.fit` hard-code chính xác objective=multi:softprob,
  num_class=5, eval_metric=mlogloss. Audit kiểm tra constructor literals, và
  mỗi fit kiểm tra lại parameters của estimator XGB thực tế. Không cần merge
  `XGB_fixed` vào constructor của wrapper.
- Weights XGB được tính bên trong từng `fit` từ training y tương ứng.
  LR dùng balanced, RF dùng balanced_subsample như cấu hình đã chốt.
- Soft Voting chỉ chạy LR:RF:XGB = 1:1:1 và 1:2:1.
- Meta learner cố định: LR C=1, solver=lbfgs, L2, class_weight=balanced,
  max_iter=3000, random_state=42. Input gồm 15 probabilities (LR, RF, XGB × C0–C4),
  không passthrough features, không search meta learner, không calibration.

Trong scikit-learn 1.9.1, code dùng `l1_ratio=0.0` để biểu diễn L2 theo API mới.
Đây là cùng penalty đã chốt. Xem [tài liệu LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

Preprocessor numerical median → StandardScaler và categorical missing/unknown
category → OneHotEncoder(handle_unknown="ignore") được fit mới trong mỗi training
partition. Structural missingness representation được giữ nguyên từ ablation.

Trong mỗi outer fold, 3 inner folds tạo OOF probabilities cho toàn bộ outer training.
Meta learner chỉ học các OOF probabilities này. Sau đó ba base learners được refit
trên outer training để dự đoán outer validation. Chính các outer refits này cũng
phục vụ standalone và Voting, bảo đảm so sánh cùng predictions. Cách dùng predictions
cross-validated để train meta learner phù hợp với
[tài liệu StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html).

## Kết quả 5 outer folds

SD là sample SD giữa 5 folds, ddof=1. Secondary metrics là trung bình 5 folds.

| Model / F4 | Macro-F1 mean ± SD | Balanced Accuracy | Macro Precision | Macro Recall | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| LR consensus | 0.2411 ± 0.0165 | 0.3239 | 0.2524 | 0.3239 | 0.5582 |
| RF tuned fixed | 0.2630 ± 0.0222 | 0.2805 | 0.2630 | 0.2805 | 0.6339 |
| XGB baseline | 0.2707 ± 0.0428 | 0.2837 | 0.2692 | 0.2837 | 0.6344 |
| Voting 1:1:1 | 0.2691 ± 0.0295 | 0.3092 | 0.2624 | 0.3092 | 0.6159 |
| Voting 1:2:1 | 0.2781 ± 0.0377 | 0.3117 | 0.2695 | 0.3117 | 0.6226 |
| Stacking | 0.2159 ± 0.0093 | 0.2979 | 0.2347 | 0.2979 | 0.5430 |

RF 0.2630 ở đây đánh giá **một cấu hình cuối cố định**. Nested-CV RF 0.2755 ± 0.0270
trước đó đánh giá quá trình chọn tham số riêng trong từng outer training: folds 1–3
chọn max_features=sqrt, folds 4–5 chọn 0.5. Experiment hiện tại dùng 0.5 ở cả 5 folds
theo cấu hình được phê duyệt. Predictions RF ở folds 4–5 tái tạo đúng kết quả nested
cũ. Không thay đổi max_features sau khi thấy kết quả này.

| Fold | LR | RF | XGB | Voting 1:1:1 | Voting 1:2:1 | Stacking |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.2599 | 0.2962 | 0.3243 | 0.3091 | 0.3328 | 0.2212 |
| 2 | 0.2226 | 0.2426 | 0.2153 | 0.2376 | 0.2447 | 0.2060 |
| 3 | 0.2567 | 0.2648 | 0.2962 | 0.2861 | 0.2998 | 0.2074 |
| 4 | 0.2373 | 0.2425 | 0.2437 | 0.2450 | 0.2473 | 0.2281 |
| 5 | 0.2292 | 0.2690 | 0.2742 | 0.2676 | 0.2658 | 0.2166 |

## C0–C4

Mỗi ô là **F1 / Recall**, tính từ pooled outer OOF trên 4.045 participant.
Đây không phải trung bình per-class F1 của các folds. CSV lưu cả hai cách tổng hợp.

| Model | C0 (n=3016) | C1 (n=638) | C2 (n=243) | C3 (n=104) | C4 (n=44) |
|---|---:|---:|---:|---:|---:|
| LR | 0.6940 / 0.5799 | 0.1796 / 0.1834 | 0.1486 / 0.2222 | 0.1314 / 0.3558 | 0.0525 / 0.2727 |
| RF | 0.7965 / 0.7609 | 0.1660 / 0.1583 | 0.1705 / 0.2757 | 0.1496 / 0.1827 | 0.0286 / 0.0227 |
| XGB | 0.7843 / 0.7298 | 0.2348 / 0.2696 | 0.1532 / 0.2222 | 0.1130 / 0.1250 | 0.0779 / 0.0682 |
| Voting 1:1:1 | 0.7662 / 0.6976 | 0.2010 / 0.2163 | 0.1401 / 0.2058 | 0.1497 / 0.2404 | 0.0904 / 0.1818 |
| Voting 1:2:1 | 0.7741 / 0.7122 | 0.1981 / 0.2116 | 0.1528 / 0.2305 | 0.1618 / 0.2404 | 0.1037 / 0.1591 |
| Stacking | 0.6900 / 0.5719 | 0.1171 / 0.1097 | 0.1272 / 0.1934 | 0.0976 / 0.2981 | 0.0538 / 0.3182 |

Voting 1:2:1 tăng C3 đúng từ 19/104 (RF) lên 25/104, C4 từ 1/44 lên 7/44.
C4 F1 tăng từ 0.0286 lên 0.1037, nhưng còn thấp, với 7 true positives trong
91 dự đoán C4 (precision=0.0769). Đây là cải thiện quan sát được trong development,
chưa đủ bằng chứng về cải thiện ổn định hoặc có ý nghĩa thống kê.

Stacking có C4 recall cao hơn (14/44), nhưng dự đoán C4 tới 476 lần, precision=0.0294,
F1=0.0538. Recall tăng ở đây đi cùng nhiều false positives; Macro-F1 giảm ở cả 5 folds.
RF vẫn gần như không nhận diện được C4. Không model nào giải quyết tốt severe classes.

## Áp dụng selection rule đã pre-specify

Ensemble chỉ đủ điều kiện nếu đồng thời: mean gain ≥ 0.01 so với RF, thắng ≥ 4/5 folds,
và SD không cao hơn RF. Đây là ngưỡng thực hành bảo thủ, không phải kiểm định thống kê.

| Ensemble | Mean delta vs RF | Folds thắng RF | SD | Kết luận |
|---|---:|---:|---:|---|
| Voting 1:1:1 | +0.006074 | 3/5 | 0.029455 | Không đạt cả ba điều kiện |
| Voting 1:2:1 | +0.015076 | 4/5 | 0.037668 | Không đạt điều kiện SD ≤ 0.022226 |
| Stacking | −0.047131 | 0/5 | 0.009349 | Không đạt điều kiện gain và số folds thắng |

Voting 1:2:1 có delta từng fold: +0.036628, +0.002106, +0.035008, +0.004843,
−0.003208. Phần lớn mean gain đến từ folds 1 và 3. Không đạt rule để thay thế RF.
**Candidate: RF/F4**, đúng rule so sánh ensemble với RF đã được người dùng phê duyệt.
Đây không phải tuyên bố RF có mean cao nhất trong mọi base learner hay ensemble.

## Kiểm tra và giới hạn diễn giải

Đã chạy tuần tự:

```bash
.venv/bin/python src/06_ensemble.py --audit-only
.venv/bin/python src/06_ensemble.py
.venv/bin/python -m unittest discover -s tests -v
```

Audit pass trước full experiment. 33 tests pass, gồm 9 tests ensemble mới:
preserved hashes, original development membership, đúng 5×3 assignments, inner OOF
coverage, 60 base fits có training-partition imputation/scaling, 20 XGB weight audits,
Voting/Stacking probability reconstruction, metrics recalculation, selection rule.
XGB tái tạo đúng baseline predictions trên cùng folds. Có 5 meta fits, tất cả trên
outer training với inner OOF features. Không có full-development final fit.
Hình so sánh đã được kiểm tra trực quan.

Đây là experiment development với configurations đã chọn từ các phân tích development
trước đó. Cross-fitting ngăn in-sample meta features và preprocessing leakage khi fit,
nhưng không loại bỏ selection bias từ lịch sử tuning/chọn cấu hình trên cùng development.
Không gọi kết quả này là một đánh giá hoàn toàn độc lập của toàn bộ quá trình chọn model.
Fold scores tương quan; SD không phải confidence interval và không chứng minh significance.
Class balancing không phải NHANES survey weighting. Probabilities chưa calibration,
không diễn giải thành nguy cơ trầm cảm trong dân số.

Held-out giữ metadata `historically_untouched=false`. Không dùng held-out predictors,
predictions hoặc model metrics trong lựa chọn base learner, weights, meta learner hay candidate.
Các kiểm tra membership chỉ dùng ID/provenance và counts đã có.

## Outputs

- [Summary và secondary metrics](tables/ensemble_results.csv)
- [Per-class F1/Recall](tables/ensemble_per_class_results.csv)
- [Fold-by-fold và paired deltas](tables/ensemble_fold_comparison.csv)
- [Metrics từng fold, gồm C0–C4](tables/ensemble_fold_results.csv)
- [Hình so sánh](figures/ensemble_comparison.png)
- [Frozen configurations](../artifacts/ensemble_model_configs.json)
- [Candidate theo rule](../artifacts/ensemble_candidate.json)

OOF probabilities, inner OOF probabilities, actual-fit audit và input/output hashes
được lưu cùng experiment để kiểm tra lại. Các kết quả ablation/tuning cũ được bảo tồn.

Dừng ở candidate RF/F4, chờ người dùng xác nhận trước final refit.
