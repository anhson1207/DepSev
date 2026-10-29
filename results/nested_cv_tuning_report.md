# DepSev-ML — Nested cross-validation tuning

Ngày chạy: 2026-09-30. Protocol `original_split_preserved_age20`.
Đã hoàn tất nested tuning và một final parameter search cho candidate được chọn. Không thay cohort, feature definitions, train/test membership; không dùng held-out để fit, tuning, chọn family/configuration hay báo performance.

## 1. Kết luận chính

- Candidate dẫn đầu trong **bốn candidates tuned**: **Random Forest/F4**, Macro-F1 **0,2755 ± 0,0270**.
- LR/F4: 0,2407 ± 0,0173; XGB/F3: 0,2516 ± 0,0267; XGB/F4: 0,2621 ± 0,0353.
- RF/F4 tăng +0,0523 so với chính RF/F4 baseline, nhưng chỉ +0,0048 so với baseline mạnh nhất trước tuning là XGB/F4 (0,2707). Không coi đây là một bước nhảy lớn so với toàn bộ baseline.
- XGBoost tuning trong budget này **không vượt baseline** ở cả F3/F4. Không mở rộng search sau khi thấy outer results.
- RF cải thiện C3 rõ trong OOF và cả 5 outer folds; C4 vẫn gần như không nhận diện được.
- Final configuration được chọn bằng outer-CV mean, không theo một fold và không dùng held-out.

## 2. Nested protocol và leakage controls

Development_age20 giữ nguyên **4.045** participants, support C0–C4 = 3016 / 638 / 243 / 104 / 44.

1. Outer StratifiedKFold(n_splits=5, shuffle=True, random_state=42): dùng đúng assignments đã khóa trong `cv_folds_final_protocol.json`.
2. Inner StratifiedKFold(n_splits=3, shuffle=True, random_state=42): chỉ tạo trên outer-training partition. Các candidates dùng cùng inner indices cho một outer fold.
3. RandomizedSearchCV với **12 sampled configurations**, scoring Macro-F1, random_state=42, refit=False; cùng budget cho LR/F4, RF/F4, XGB/F3, XGB/F4.
4. Chọn best_params bằng mean inner validation Macro-F1; clone và fit lại pipeline trên outer-training, rồi dự đoán outer-validation đúng một lần.
5. Lưu metrics, params và OOF; tổng hợp đủ cả 5 folds trước khi chọn family/configuration.
6. Khóa lựa chọn tại `artifacts/nested_cv_selection.json`, sau đó chạy **một** RandomizedSearchCV trên toàn development cho candidate thắng, cùng search space, budget, seed và 3-fold CV.
7. Final search vẫn refit=False: chỉ lưu hyperparameters, **chưa fit final model trên toàn development**. Không xuất estimator .joblib.

Nested CV tách dữ liệu chọn hyperparameters khỏi dữ liệu chấm outer performance; dùng điểm tốt nhất của inner search làm performance sẽ optimistic. [Scikit-learn nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Numerical median → StandardScaler; categorical missing_or_unknown → OneHotEncoder(handle_unknown="ignore"), tất cả fit riêng trong training partition. Deterministic questionnaire coding/routing giữ nguyên, không học thống kê trước CV.

### XGBoost weights

`FoldBalancedXGBClassifier.fit` tự tính w(c)=n_fit/(5 × n_class_fit) từ chính y được truyền vào fit. Không truyền sample_weight tính trước vào RandomizedSearchCV. Nhờ vậy mỗi inner fit và outer-training refit dùng đúng class frequencies của riêng training partition đó.

370 audit records = 2 feature sets × 5 outer folds × (12 candidates × 3 inner folds + 1 outer refit). Mỗi record lưu context, hyperparameters, hash training SEQN, counts và weights; tests đối chiếu với assignments thật. Không có current/original held-out IDs trong các training/OOF partitions.

### Search spaces được khóa trước run

```text
LR/F4:
  C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
  solver = [lbfgs, newton-cg]
  L2, class_weight=balanced, max_iter=3000

RF/F4:
  n_estimators = [150, 300, 450]
  max_depth = [None, 8, 16]
  min_samples_split = [2, 10]
  min_samples_leaf = [1, 3, 8]
  max_features = [sqrt, 0.5]
  class_weight = [balanced, balanced_subsample]

XGB/F3 và XGB/F4:
  n_estimators = [100, 200, 350]
  max_depth = [2, 3, 4, 5]
  learning_rate = [0.03, 0.05, 0.1]
  min_child_weight = [1, 3, 5]
  subsample = [0.8, 1.0]
  colsample_bytree = [0.8, 1.0]
  gamma = [0, 0.2]
  reg_alpha = [0, 0.1]
  reg_lambda = [1, 5]
```

Có 24/216/3456 tổ hợp khả dĩ cho LR/RF/XGB, nhưng chỉ sample 12 mỗi search. XGB/F3 và F4 thử đúng cùng 12 parameter combinations. Không adaptive search budget, không early stopping. RF/F2 sensitivity không chạy ở bước này để giữ scope/budget bốn candidates; Decision Tree giữ baseline. LR và RF vẫn được giữ trong kết quả/outer-fold configs cho ensemble sau này.

Trong sklearn 1.9.1 đang cài, penalty L2 được biểu diễn bằng `l1_ratio=0.0` để tránh API `penalty` đã deprecated; không thay đổi loại regularization mà người dùng yêu cầu. Model seeds 42; estimator n_jobs=1, search n_jobs=2, BLAS threads giới hạn 1.

## 3. Nested-CV performance và baseline comparison

Mean ± sample SD (ddof=1) qua 5 outer folds. Các metrics validation không dùng sample weights. Balanced Accuracy bằng Macro Recall trong cách chấm multiclass này.

| Model/set | Macro-F1 mean ± SD | Balanced Acc | Macro Precision | Macro Recall | Weighted F1 | Baseline Macro-F1 | Δ tuned−baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR/F4 | 0.2407 ± 0.0173 | 0.3253 | 0.2526 | 0.3253 | 0.5599 | 0.2399 | +0.0009 |
| RF/F4 | 0.2755 ± 0.0270 | 0.2983 | 0.2693 | 0.2983 | 0.6437 | 0.2232 | +0.0523 |
| XGB/F3 | 0.2516 ± 0.0267 | 0.2658 | 0.2527 | 0.2658 | 0.6130 | 0.2588 | -0.0071 |
| XGB/F4 | 0.2621 ± 0.0353 | 0.2743 | 0.2657 | 0.2743 | 0.6196 | 0.2707 | -0.0087 |

CSV: [nested_cv_summary.csv](tables/nested_cv_summary.csv), [nested_cv_tuning_results.csv](tables/nested_cv_tuning_results.csv).

Outer Macro-F1 từng fold:

| Candidate | Outer 1 | Outer 2 | Outer 3 | Outer 4 | Outer 5 |
| --- | --- | --- | --- | --- | --- |
| LR/F4 | 0.2599 | 0.2198 | 0.2566 | 0.2382 | 0.2292 |
| RF/F4 | 0.3111 | 0.2615 | 0.2936 | 0.2425 | 0.2690 |
| XGB/F3 | 0.2686 | 0.2328 | 0.2899 | 0.2271 | 0.2398 |
| XGB/F4 | 0.3017 | 0.2207 | 0.2925 | 0.2341 | 0.2615 |

RF/F4 tuned vượt RF/F4 baseline ở **5/5 outer folds**; LR/F4 ở 3/5; XGB/F3 và XGB/F4 chỉ ở 1/5 mỗi candidate. Baseline được lấy từ final-protocol ablation, cùng development và đúng outer assignments; không trộn với age20_v2.

[nested_cv_baseline_comparison.csv](tables/nested_cv_baseline_comparison.csv).

## 4. XGB/F3 tuned vs XGB/F4 tuned

| Outer fold | XGB/F3 tuned | XGB/F4 tuned | ΔF4−F3 |
| --- | --- | --- | --- |
| 1 | 0.268579 | 0.301690 | +0.033111 |
| 2 | 0.232773 | 0.220691 | -0.012081 |
| 3 | 0.289899 | 0.292453 | +0.002554 |
| 4 | 0.227064 | 0.234069 | +0.007005 |
| 5 | 0.239797 | 0.261531 | +0.021734 |

Mean ΔF4−F3 = **0.010464**; F4 thắng 4/5 outer folds. F4 dẫn đầu trong hai XGB configurations, nhưng không phải candidate có mean cao nhất khi so thêm RF/F4.

C3 pooled OOF: F3 recall 0,1058 / F1 0,0870; F4 recall 0,1154 / F1 0,0941.
C4 pooled OOF: F3 recall 0,0455 / F1 0,0444; F4 recall 0,0682 / F1 0,0698.

Không coi thắng 4/5 folds hoặc mean cao hơn là bằng chứng significance; mỗi C4 validation fold chỉ có 8–9 người.

[nested_cv_xgb_F3_F4_comparison.csv](tables/nested_cv_xgb_F3_F4_comparison.csv).

## 5. Per-class F1 / Recall C0–C4

Pooled outer-OOF, mỗi development participant có đúng một prediction cho mỗi candidate. Pooled F1 không nhất thiết bằng mean fold F1. Mọi outer-fold per-class F1/Recall cũng nằm trong nested_cv_tuning_results.csv.

| Candidate | C0 F1 / Recall | C1 F1 / Recall | C2 F1 / Recall | C3 F1 / Recall | C4 F1 / Recall |
| --- | --- | --- | --- | --- | --- |
| LR/F4 | 0.6966 / 0.5836 | 0.1832 / 0.1850 | 0.1390 / 0.2016 | 0.1307 / 0.3558 | 0.0544 / 0.2955 |
| RF/F4 | 0.8065 / 0.7848 | 0.1611 / 0.1426 | 0.1992 / 0.3086 | 0.1825 / 0.2308 | 0.0270 / 0.0227 |
| XGB/F3 | 0.7611 / 0.6950 | 0.2172 / 0.2586 | 0.1500 / 0.2222 | 0.0870 / 0.1058 | 0.0444 / 0.0455 |
| XGB/F4 | 0.7682 / 0.7042 | 0.2221 / 0.2618 | 0.1483 / 0.2181 | 0.0941 / 0.1154 | 0.0698 / 0.0682 |

[nested_cv_per_class_results.csv](tables/nested_cv_per_class_results.csv), [nested_cv_oof_predictions.csv](tables/nested_cv_oof_predictions.csv), [nested_cv_confusion_matrices.csv](tables/nested_cv_confusion_matrices.csv).

### C3/C4 có cải thiện không?

| Candidate | Class | Recall baseline → tuned | F1 baseline → tuned | TP / support | Precision tuned |
| --- | --- | --- | --- | --- | --- |
| LR/F4 | C3 | 0.3558 → 0.3558 | 0.1329 → 0.1307 | 37 / 104 | 0.0801 |
| LR/F4 | C4 | 0.2727 → 0.2955 | 0.0523 → 0.0544 | 13 / 44 | 0.0300 |
| RF/F4 | C3 | 0.0096 → 0.2308 | 0.0175 → 0.1825 | 24 / 104 | 0.1509 |
| RF/F4 | C4 | 0.0000 → 0.0227 | 0.0000 → 0.0270 | 1 / 44 | 0.0333 |
| XGB/F3 | C3 | 0.0962 → 0.1058 | 0.0830 → 0.0870 | 11 / 104 | 0.0738 |
| XGB/F3 | C4 | 0.0682 → 0.0455 | 0.0682 → 0.0444 | 2 / 44 | 0.0435 |
| XGB/F4 | C3 | 0.1250 → 0.1154 | 0.1130 → 0.0941 | 12 / 104 | 0.0795 |
| XGB/F4 | C4 | 0.0682 → 0.0682 | 0.0779 → 0.0698 | 3 / 44 | 0.0714 |

- **RF/F4 C3:** từ 1/104 lên 24/104 ca đúng; recall 0,0096 → 0,2308; F1 0,0175 → 0,1825. Cả recall lẫn F1 tăng ở **5/5 outer folds**. Đây là cải thiện rõ và nhất quán trong development OOF đã quan sát.
- **RF/F4 C4:** từ 0/44 lên **1/44**, chỉ cải thiện ở 1/5 folds. Recall 0,0227, F1 0,0270; **chưa cải thiện đáng tin cậy**, vẫn bỏ sót 43/44 người C4.
- RF/F4 recall C2 tăng 0,0453 → 0,3086, trong khi recall C1 giảm 0,1755 → 0,1426 và C0 giảm 0,8962 → 0,7848. Macro-F1 tăng không đồng nghĩa mọi lớp đều tăng.
- LR/F4 C4 recall cao hơn RF, nhưng precision chỉ khoảng 3% (13 ca đúng trong 434 predictions C4); không thể dùng recall riêng để nói phân loại severe tốt.
- XGB/F4 tuning không cải thiện severe so với chính baseline: C3 13 → 12 ca đúng; C4 vẫn 3/44, F1 C4 giảm do false positives tăng.
- Chưa khẳng định significance hoặc hiệu quả ngoài cohort. Chưa có cơ sở cho việc sử dụng model như công cụ chẩn đoán.

[nested_cv_per_class_baseline_comparison.csv](tables/nested_cv_per_class_baseline_comparison.csv).

## 6. Best hyperparameters cho refit sau này

Model family: **Random Forest**. Feature configuration: **F4=P+C+L+D** (34 inputs trước one-hot).

Final search chỉ chạy **sau khi khóa RF/F4 bằng outer-CV mean**. Hyperparameters được chọn:

```json
{
  "n_estimators": 150,
  "min_samples_split": 10,
  "min_samples_leaf": 8,
  "max_features": 0.5,
  "max_depth": 8,
  "class_weight": "balanced_subsample"
}
```

Các fixed parameters gồm random_state=42, n_jobs=1, bootstrap=True, criterion="gini". Mỗi outer fold cũng chọn depth=8, min_samples_leaf=8, n_estimators=150; max_features là sqrt ở folds 1–3 và 0.5 ở folds 4–5. Không ép final search phải chọn mode của outer-fold params.

Best inner score của search trên toàn development: **0.269693**. Đây là score dùng chọn parameters, **không phải independent performance**; performance báo cáo vẫn là nested outer-CV phía trên. Các outer models có thể dùng parameters khác final config, nên không gán trực tiếp nested score như điểm đo riêng của một fitted final estimator.

`refit=False` không tạo best_estimator fitted trên toàn development. [Scikit-learn RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

Artifact: [tuned_model_configs.json](../artifacts/tuned_model_configs.json).
Chi tiết final search: [final_development_search_results.csv](tables/final_development_search_results.csv).

LR/F4 và RF/F4 outer-fold configurations được giữ trong [best_params_by_fold.csv](tables/best_params_by_fold.csv). Chưa chạy thêm full-development searches riêng cho LR/XGB hoặc dựng ensemble.

## 7. Best inner parameters từng outer fold

| Candidate | Outer fold | Best inner Macro-F1 | Best parameters (model__ prefix omitted) |
| --- | --- | --- | --- |
| LR/F4 | 1 | 0.2437 | `{"C":0.3,"solver":"newton-cg"}` |
| LR/F4 | 2 | 0.2387 | `{"C":0.01,"solver":"newton-cg"}` |
| LR/F4 | 3 | 0.2357 | `{"C":1,"solver":"newton-cg"}` |
| LR/F4 | 4 | 0.2502 | `{"C":100,"solver":"newton-cg"}` |
| LR/F4 | 5 | 0.2387 | `{"C":0.3,"solver":"newton-cg"}` |
| RF/F4 | 1 | 0.2552 | `{"class_weight":"balanced_subsample","max_depth":8,"max_features":"sqrt","min_samples_leaf":8,"min_samples_split":10,"n_estimators":150}` |
| RF/F4 | 2 | 0.2631 | `{"class_weight":"balanced_subsample","max_depth":8,"max_features":"sqrt","min_samples_leaf":8,"min_samples_split":10,"n_estimators":150}` |
| RF/F4 | 3 | 0.2627 | `{"class_weight":"balanced_subsample","max_depth":8,"max_features":"sqrt","min_samples_leaf":8,"min_samples_split":10,"n_estimators":150}` |
| RF/F4 | 4 | 0.2753 | `{"class_weight":"balanced_subsample","max_depth":8,"max_features":0.5,"min_samples_leaf":8,"min_samples_split":10,"n_estimators":150}` |
| RF/F4 | 5 | 0.2689 | `{"class_weight":"balanced_subsample","max_depth":8,"max_features":0.5,"min_samples_leaf":8,"min_samples_split":10,"n_estimators":150}` |
| XGB/F3 | 1 | 0.2557 | `{"colsample_bytree":0.8,"gamma":0.2,"learning_rate":0.1,"max_depth":4,"min_child_weight":5,"n_estimators":100,"reg_alpha":0.1,"reg_lambda":5,"subsample":0.8}` |
| XGB/F3 | 2 | 0.2639 | `{"colsample_bytree":0.8,"gamma":0.2,"learning_rate":0.03,"max_depth":5,"min_child_weight":1,"n_estimators":200,"reg_alpha":0.1,"reg_lambda":5,"subsample":1}` |
| XGB/F3 | 3 | 0.2551 | `{"colsample_bytree":0.8,"gamma":0.2,"learning_rate":0.03,"max_depth":5,"min_child_weight":1,"n_estimators":200,"reg_alpha":0.1,"reg_lambda":5,"subsample":1}` |
| XGB/F3 | 4 | 0.2872 | `{"colsample_bytree":0.8,"gamma":0,"learning_rate":0.05,"max_depth":4,"min_child_weight":3,"n_estimators":200,"reg_alpha":0,"reg_lambda":5,"subsample":0.8}` |
| XGB/F3 | 5 | 0.2820 | `{"colsample_bytree":1,"gamma":0.2,"learning_rate":0.05,"max_depth":4,"min_child_weight":5,"n_estimators":350,"reg_alpha":0.1,"reg_lambda":1,"subsample":0.8}` |
| XGB/F4 | 1 | 0.2613 | `{"colsample_bytree":0.8,"gamma":0,"learning_rate":0.05,"max_depth":4,"min_child_weight":3,"n_estimators":200,"reg_alpha":0,"reg_lambda":5,"subsample":0.8}` |
| XGB/F4 | 2 | 0.2737 | `{"colsample_bytree":0.8,"gamma":0.2,"learning_rate":0.03,"max_depth":5,"min_child_weight":1,"n_estimators":200,"reg_alpha":0.1,"reg_lambda":5,"subsample":1}` |
| XGB/F4 | 3 | 0.2512 | `{"colsample_bytree":1,"gamma":0.2,"learning_rate":0.05,"max_depth":4,"min_child_weight":5,"n_estimators":350,"reg_alpha":0.1,"reg_lambda":1,"subsample":0.8}` |
| XGB/F4 | 4 | 0.2898 | `{"colsample_bytree":0.8,"gamma":0.2,"learning_rate":0.05,"max_depth":3,"min_child_weight":1,"n_estimators":200,"reg_alpha":0.1,"reg_lambda":5,"subsample":0.8}` |
| XGB/F4 | 5 | 0.2760 | `{"colsample_bytree":0.8,"gamma":0,"learning_rate":0.05,"max_depth":4,"min_child_weight":3,"n_estimators":200,"reg_alpha":0,"reg_lambda":5,"subsample":0.8}` |


## 8. Giới hạn diễn giải và kiểm chứng

- Nested CV đánh giá quy trình tuning trong feature definitions/candidate space đã chốt. Các features/candidates và prior analyses từng dùng cùng development; chọn winner trong bốn outer-CV means vẫn tạo selection optimism. Không gọi score winner là hoàn toàn unbiased sau tất cả lựa chọn.
- Các outer folds có training sets chồng lấp. SD không phải confidence interval; không dùng t-test IID hoặc SD/√5 để tuyên bố significance.
- Held-out giữ `historically_untouched=false`; không dùng nó để giải quyết lựa chọn model hay xác nhận improvement.
- RF/F4 là tốt nhất **trong budget/candidates đã chạy**, không chứng minh tối ưu toàn cục. C4 performance vẫn rất thấp.
- Class-balancing weights không phải NHANES survey weights.

24 automated tests đạt, gồm: input/cohort/split/baseline hashes; original membership; nested assignments; equal budgets và identical sampled XGB parameters; chọn params chỉ từ inner scores; kiểm tra đủ OOF coverage và tính lại outer/per-class metrics; kiểm chứng 370 training-weight audits; final search chỉ cho candidate đã khóa; không full-development refit.

```bash
python src/05_nested_tuning.py
python -m unittest discover -s tests -v
```

Script lưu checkpoints từng outer experiment và khóa manifest. Khi chạy lại cùng protocol, dùng checkpoints và final config hiện có, không chạy lại final search. Thay inputs/code/search space sẽ khiến lock dừng để review.

Không cohort/split/feature changes; không SMOTE, Voting, Stacking, SHAP, held-out evaluation hoặc refit final model.

