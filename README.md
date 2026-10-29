# DepSev-ML

Nghiên cứu dự đoán mức độ triệu chứng trầm cảm từ NHANES 2013–2014.
Protocol active: **original_split_preserved_age20**.
Đã hoàn tất ablation, nested tuning, ensemble, final refit RF/F4,
**supplementary held-out evaluation** và development explainability.
Final model đã khóa; `historically_untouched=false`. Chưa dùng SMOTE.
Xem [báo cáo cuối](results/final_report.md).

## Membership và provenance

Không tạo random train/test split mới. Membership được lấy từ split đầu tiên
trong `artifacts/archive/pre_age20_v1/artifacts/split_indices.json`, chỉ lọc tuổi
`RIDAGEYR >= 20` riêng trong từng tập:

- Original development 4.297 → development_age20 **4.045**.
- Original held-out 1.075 → heldout_age20 **1.009**.
- Cohort đủ 9 PHQ-9 items vẫn **5.054**.
- Class counts development C0–C4: **3016 / 638 / 243 / 104 / 44**.
- Class counts held-out C0–C4: **748 / 162 / 60 / 28 / 11**.

`artifacts/split_indices_final_protocol.json` lưu membership, assertions, source
hashes và metadata; `artifacts/split_indices.json` là active alias giống hệt.
Mỗi experiment kiểm tra không có development ID từ original held-out và ngược lại.

**Giới hạn không thể đảo ngược:** 813 restored held-out participants đã tham gia
development trong age20_v2 trước khi khôi phục. Metadata giữ
`historically_untouched=false`. Bước correction trước đó chỉ kiểm tra membership;
sau khi người dùng xác nhận và khóa final RF mới chạy một lần
**supplementary held-out evaluation**. Primary performance evidence vẫn là
development CV / nested-CV, với các giới hạn selection bias đã ghi trong báo cáo.

age20_v2 đã retired và lưu đầy đủ, có hash trong `artifacts/archive/age20_v2/`.
Kết quả exploratory cũ không bị ghi đè. Outputs ablation/repeated-CV có hậu tố
`_final_protocol`; outputs tuning có prefix `nested_cv_` và giữ cùng membership.

## Target và feature groups

Target là tổng 9 PHQ-9 items, mỗi item hợp lệ 0–3:

```text
DPQ010 DPQ020 DPQ030 DPQ040 DPQ050 DPQ060 DPQ070 DPQ080 DPQ090
```

```yaml
C0: 0–4   Minimal
C1: 5–9   Mild
C2: 10–14 Moderate
C3: 15–19 Moderately severe
C4: 20–27 Severe
```

```ini
P = Physiological
C = Clinical
L = Lifestyle
D = Demographic

F1 = P             # 10 model inputs trước one-hot
F2 = P + C         # 19
F3 = P + C + L     # 29
F4 = P + C + L + D # 34
```

Giữ nguyên feature definitions và questionnaire routing đã audit.
Danh sách exact và missingness development:
`results/tables/ablation_feature_inventory_final_protocol.csv`.

Base models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
Dummy: most_frequent và stratified. Primary metric: **Macro-F1**.
Secondary: Balanced Accuracy, Macro Precision, Macro Recall, Weighted F1,
per-class C0–C4 Recall/F1.

## Setup macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu XGBoost báo thiếu libomp.dylib, dùng `brew install libomp`.
Versions thực tế của experiment được lưu trong ablation_config_final_protocol.json.

## Lịch sử pipeline development

Archive, restore và các experiments dưới đây đã hoàn tất. Các lệnh này mô tả thứ tự
development trước khi khóa final specification; không rebuild, re-split hoặc tuning
thêm sau supplementary held-out evaluation:

```bash
python src/02_restore_original_split.py
python src/05_feature_ablation.py --audit-only
python src/05_feature_ablation.py
python src/05_robustness.py
python -m unittest discover -s tests -v
```

`02_build_cohort.py` random-re-split và legacy runners 03/04 bị chặn ở protocol
hiện tại để bảo vệ membership và exploratory outputs. Scripts 06–08 đã hoàn tất.
Archive scripts chỉ dùng một lần trước migration; không chạy lại khi archive tồn tại.

## Evaluation protocol

- Fixed StratifiedKFold(5, shuffle=True, random_state=42), khóa theo SEQN trên
  development_age20; giống nhau cho 2 Dummy và 16 experiments.
- Numerical median → StandardScaler; categorical missing_or_unknown → one-hot.
  Preprocessor fit riêng trong training fold.
- LR/DT/RF class_weight="balanced"; XGBoost balanced sample_weight chỉ tính từ
  y_train của từng fold. Validation metrics không weighted.
- SMQ/ALQ structural missing được biểu diễn theo questionnaire routing, không
  mode-impute như nonresponse. Tagged zero cho quantity không áp dụng luôn đi
  cùng alcohol_status; không đồng nghĩa exposure bằng 0.
- RepeatedStratifiedKFold(5, 10 repeats, random_state=42), chỉ XGB/F3 và F4,
  paired splits. Repeat 1 trùng fixed-CV.
- Không chạy lại sensitivity 4-class ở protocol này. Kết quả sensitivity cũ
  chỉ là exploratory age20_v2, không phải final_protocol.

## Báo cáo và outputs

[Báo cáo cuối](results/final_report.md): final RF/F4 refit trên development N=4.045.
Supplementary held-out evaluation N=1.009 đạt Macro-F1=0,3006,
Balanced Accuracy=0,3197; C3 đúng 4/28, C4 đúng 1/11. Model SHA-256 giữ nguyên
trước/sau evaluation và explainability. Các artifacts chính:

- `artifacts/final_model_spec.json`: exact 34 features, preprocessing, RF config,
  target mapping, selection rule và hashes.
- `artifacts/final_model.pkl`: fitted preprocessing + RF Pipeline, lưu bằng joblib.
- `artifacts/final_feature_schema.json`, `artifacts/class_mapping.json`.
- `results/tables/final_heldout_predictions.csv`: một lượt predictions và probabilities.
- `results/tables/final_heldout_metrics.csv`, `final_heldout_per_class_results.csv`.
- `results/figures/final_confusion_matrix.png`.
- `results/tables/permutation_importance.csv`: development in-sample, global và C0–C4.
- `results/tables/shap_global_importance.csv`, `shap_per_class_importance.csv`.
- `results/figures/shap_summary_global.png`, `shap_summary_C0.png` … `C4.png`.

Các stages đã chạy tuần tự. Chạy lại chỉ kiểm tra artifact đã lưu, không refit hoặc
predict supplementary held-out evaluation thêm lần nữa:

```bash
python src/07_final_test.py --stage freeze
python src/07_final_test.py --stage refit
python src/07_final_test.py --stage evaluate
python src/08_explain.py
python -m unittest discover -s tests -v
```

Input của fitted pipeline là 34 predictors sau deterministic questionnaire coding,
trước imputation/encoding. SHAP chạy trên development, shape thực tế (4045,79,5),
cộng signed one-hot attributions về 34 features. HUQ010, INDFMPIR, BPXPLS đứng đầu
global. Không diễn giải attribution là causal effect hoặc calibrated population risk.

[Báo cáo ensemble](results/ensemble_report.md): đã khóa LR consensus/F4, RF tuned/F4
và XGB baseline/F4; so sánh standalone, Soft Voting 1:1:1, 1:2:1 và Stacking với
inner OOF trên cùng 5 development folds. Voting 1:2:1 đạt 0,2781 ± 0,0377, nhưng
không đạt rule SD ≤ RF. Candidate theo rule là **RF/F4** (fixed configuration:
0,2630 ± 0,0222). Đây là snapshot trước final refit và supplementary held-out evaluation.

```bash
python src/06_ensemble.py --audit-only
python src/06_ensemble.py
```

Audit kiểm tra XGB objective/num_class/eval_metric hard-coded trong wrapper và khóa
configuration, source/input hashes, outer/inner folds. Full run từ chối ghi đè
experiment đã hoàn tất. Các file `results/tables/ensemble_*.csv` có mean/SD,
per-class metrics và OOF evidence; `artifacts/ensemble_model_configs.json` giữ
parameters, `artifacts/ensemble_candidate.json` giữ kết quả áp dụng selection rule.
Consensus LR C=0.3 không được gọi là globally optimal. Stacking meta learner chỉ
học inner OOF predictions. Xem báo cáo để phân biệt fixed-config CV với nested-CV
tuning trước đó và các giới hạn selection bias.

[Báo cáo nested tuning mới nhất](results/nested_cv_tuning_report.md): RF/F4 dẫn đầu
trong bốn candidates tuned, Macro-F1 0,2755 ± 0,0270. C3 cải thiện rõ so với RF
baseline, C4 vẫn rất yếu. XGB/F3 và XGB/F4 tuned không vượt chính baseline của chúng.

Lệnh tuning đã sử dụng trước final freeze, sau ablation final protocol:

```bash
python src/05_nested_tuning.py
python -m unittest discover -s tests -v
```

Nested 5 outer × 3 inner folds, RandomizedSearchCV 12 candidates/search cho LR/F4,
RF/F4, XGB/F3 và XGB/F4. Preprocessor fit trong mỗi training partition. Wrapper XGB
tự tính balanced weights trong mỗi `fit`, không truyền weights tính trước vào search.

Sau khi khóa candidate bằng outer-CV mean, chỉ RF/F4 được search thêm một lần trên
toàn development với cùng space/budget và `refit=False`. Hyperparameters nằm trong
`artifacts/tuned_model_configs.json`; tại thời điểm kết thúc tuning chưa refit.
LR/RF outer-fold configurations được giữ làm provenance cho ensemble tiếp theo.

Các outputs tuning chính:

- `results/tables/nested_cv_tuning_results.csv`: metrics và best inner params từng outer fold.
- `results/tables/nested_cv_summary.csv`: mean/SD outer metrics và delta so baseline.
- `results/tables/nested_cv_per_class_results.csv`: pooled outer-OOF C0–C4 F1/Recall.
- `results/tables/best_params_by_fold.csv`: tham số tốt nhất của 20 outer experiments.
- `results/tables/nested_cv_xgb_F3_F4_comparison.csv`: paired outer differences.
- `artifacts/nested_tuning_protocol.json`: search spaces, seeds, budget, input/code hashes.
- `artifacts/nested_cv_assignments.json`: đúng outer/inner membership theo SEQN.
- `artifacts/nested_tuning_weight_audit/`: 370 bằng chứng training counts/weights XGB.

Script dùng checkpoints khi chạy lại cùng protocol và không rerun final search đã
hoàn thành. Nested CV không xóa selection bias từ prior development analyses hoặc
việc chọn winner trong nhiều outer-CV means; không gọi score winner hoàn toàn unbiased.

[Báo cáo final protocol](results/ablation_report_final_protocol.md) gồm provenance,
assertions, cohort counts, feature list/missingness, Dummy, 16 experiments,
per-class F1/Recall, fixed/repeated paired differences và giới hạn diễn giải.

Các file chính trong `results/tables/`:

- `class_distribution_final_protocol.csv`
- `feature_ablation_results_final_protocol.csv`
- `dummy_cv_results_final_protocol.csv`
- `per_class_cv_results_final_protocol.csv`
- `ablation_fold_results_final_protocol.csv`
- `ablation_oof_predictions_final_protocol.csv`
- `xgb_fixed_F3_F4_comparison_final_protocol.csv`
- `repeated_cv_summary_final_protocol.csv`
- `repeated_cv_paired_differences_final_protocol.csv`
- `repeated_cv_repeat_means_final_protocol.csv`
- `repeated_cv_difference_summary_final_protocol.csv`
- `repeated_cv_per_class_final_protocol.csv`

Fold assignments, code/input hashes và model parameters ở
`artifacts/*_final_protocol.json`. Các file weights lưu class counts và weights
của từng training fold. Hình so sánh ở `results/figures/*_final_protocol.png`.

SD dùng ddof=1; repeated folds không độc lập nên không suy significance từ SD/√50.
Class-balancing weights không phải NHANES survey weights. Target là mức độ triệu
chứng PHQ-9, không phải chẩn đoán lâm sàng. Suffix final_protocol là tên phiên bản,
không chứng nhận held-out historically untouched.
