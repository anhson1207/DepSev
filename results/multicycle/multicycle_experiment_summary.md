# Controlled multi-cycle experiment: RF/F4_common31

## Protocol and interpretation

Official CDC cohorts only; age >=20 and nine valid PHQ-9 items. A: 2013–2014, N=5,054. B: pooled 2011–2016, N=14,581. C: 10 exact-class-count matched samples, each N=5,054, from 2011–2012 and 2015–2016 only. D: train 2011–2014, N=9,693; temporal test 2015–2016, N=4,888.

Frozen 31 predictors and RF configuration. Numerical median imputation + StandardScaler; categorical constant missing_or_unknown + one-hot encoding. Fresh preprocessing inside every training partition. T5-stratified five-fold assignments (shuffle=True, seed=42) reused for T2–T5 within each dataset/sample. No cycle, PHQ items, total, severity or applicability metadata in X. No tuning, resampling of training folds, or threshold adjustment.

T2: 0–4 / 5–27 (minimal symptoms vs symptomatic); T3: 0–4 / 5–9 / 10–27; T4: 0–4 / 5–9 / 10–14 / 15–27; T5: 0–4 / 5–9 / 10–14 / 15–19 / 20–27.

A/B means and SD describe five validation folds. C means and SD describe 10 sample-level CV means, not 50 independent folds. D is one temporal test, without CV SD. Per-class A/B results use pooled OOF predictions; C averages 10 sample-level OOF metrics. No confidence intervals or paired A-vs-B significance tests are used.

## Main comparisons

### macro_f1

| Target | A_single | C_matched | B_full | D_temporal | B_minus_A |
| --- | --- | --- | --- | --- | --- |
| T2 | 0.6524 ± 0.0117 | 0.6503 ± 0.0055 | 0.6548 ± 0.0042 | 0.6500 | 0.0024 |
| T3 | 0.4417 ± 0.0198 | 0.4440 ± 0.0056 | 0.4436 ± 0.0113 | 0.4345 | 0.0019 |
| T4 | 0.3386 ± 0.0234 | 0.3401 ± 0.0112 | 0.3227 ± 0.0129 | 0.3250 | -0.0159 |
| T5 | 0.2634 ± 0.0141 | 0.2700 ± 0.0121 | 0.2619 ± 0.0063 | 0.2676 | -0.0015 |

### balanced_accuracy

| Target | A_single | C_matched | B_full | D_temporal | B_minus_A |
| --- | --- | --- | --- | --- | --- |
| T2 | 0.6555 ± 0.0121 | 0.6537 ± 0.0054 | 0.6633 ± 0.0049 | 0.6555 | 0.0078 |
| T3 | 0.4778 ± 0.0256 | 0.4752 ± 0.0054 | 0.4929 ± 0.0185 | 0.4908 | 0.0151 |
| T4 | 0.3720 ± 0.0326 | 0.3742 ± 0.0108 | 0.3756 ± 0.0278 | 0.3848 | 0.0036 |
| T5 | 0.2841 ± 0.0181 | 0.3006 ± 0.0164 | 0.3076 ± 0.0136 | 0.3135 | 0.0235 |

## T5 rare classes

| experiment | class | recall | recall_sd | f1 | f1_sd | support | true_positives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | C3 | 0.1515 | — | 0.1146 | — | 132.0000 | 20.0000 |
| B | C3 | 0.2869 | — | 0.1152 | — | 352.0000 | 101.0000 |
| C | C3 | 0.2303 | 0.0442 | 0.1405 | 0.0215 | 132.0000 | 30.4000 |
| D | C3 | 0.2323 | — | 0.1080 | — | 99.0000 | 23.0000 |
| A | C4 | 0.0545 | — | 0.0600 | — | 55.0000 | 3.0000 |
| B | C4 | 0.1377 | — | 0.0962 | — | 167.0000 | 23.0000 |
| C | C4 | 0.0891 | 0.0850 | 0.0713 | 0.0508 | 55.0000 | 4.9000 |
| D | C4 | 0.1698 | — | 0.1406 | — | 53.0000 | 9.0000 |

## Confusion patterns

C below aggregates prediction occurrences across overlapping samples, not unique participants.

| experiment | pattern | count | support | percent |
| --- | --- | --- | --- | --- |
| A | C1->C0 | 435 | 800 | 54.3750 |
| A | C3->C2 | 51 | 132 | 38.6364 |
| A | C4->C0 | 14 | 55 | 25.4545 |
| A | C4->C1 | 5 | 55 | 9.0909 |
| A | C4->C2 | 25 | 55 | 45.4545 |
| A | C4->C3 | 8 | 55 | 14.5455 |
| A | C4->C4 | 3 | 55 | 5.4545 |
| B | C1->C0 | 1210 | 2327 | 51.9983 |
| B | C3->C2 | 108 | 352 | 30.6818 |
| B | C4->C0 | 32 | 167 | 19.1617 |
| B | C4->C1 | 14 | 167 | 8.3832 |
| B | C4->C2 | 52 | 167 | 31.1377 |
| B | C4->C3 | 46 | 167 | 27.5449 |
| B | C4->C4 | 23 | 167 | 13.7725 |
| C | C1->C0 | 4116 | 8000 | 51.4500 |
| C | C3->C2 | 385 | 1320 | 29.1667 |
| C | C4->C0 | 99 | 550 | 18.0000 |
| C | C4->C1 | 106 | 550 | 19.2727 |
| C | C4->C2 | 170 | 550 | 30.9091 |
| C | C4->C3 | 126 | 550 | 22.9091 |
| C | C4->C4 | 49 | 550 | 8.9091 |
| D | C1->C0 | 445 | 841 | 52.9132 |
| D | C3->C2 | 42 | 99 | 42.4242 |
| D | C4->C0 | 5 | 53 | 9.4340 |
| D | C4->C1 | 2 | 53 | 3.7736 |
| D | C4->C2 | 20 | 53 | 37.7358 |
| D | C4->C3 | 17 | 53 | 32.0755 |
| D | C4->C4 | 9 | 53 | 16.9811 |

## Pooled-model OOF performance by source cycle

| cycle | target | n | macro_f1 | balanced_accuracy |
| --- | --- | --- | --- | --- |
| 2011_2012 | T2 | 4639 | 0.6608 | 0.6729 |
| 2011_2012 | T3 | 4639 | 0.4460 | 0.4934 |
| 2011_2012 | T4 | 4639 | 0.3198 | 0.3665 |
| 2011_2012 | T5 | 4639 | 0.2617 | 0.2945 |
| 2013_2014 | T2 | 5054 | 0.6552 | 0.6622 |
| 2013_2014 | T3 | 5054 | 0.4519 | 0.4954 |
| 2013_2014 | T4 | 5054 | 0.3356 | 0.3891 |
| 2013_2014 | T5 | 5054 | 0.2641 | 0.3078 |
| 2015_2016 | T2 | 4888 | 0.6489 | 0.6559 |
| 2015_2016 | T3 | 4888 | 0.4325 | 0.4906 |
| 2015_2016 | T4 | 4888 | 0.3120 | 0.3701 |
| 2015_2016 | T5 | 4888 | 0.2593 | 0.3236 |

## Granularity deltas

| experiment | metric | transition | delta |
| --- | --- | --- | --- |
| A | macro_f1 | T3-T2 | -0.2107 |
| A | macro_f1 | T4-T3 | -0.1031 |
| A | macro_f1 | T5-T4 | -0.0752 |
| A | balanced_accuracy | T3-T2 | -0.1778 |
| A | balanced_accuracy | T4-T3 | -0.1057 |
| A | balanced_accuracy | T5-T4 | -0.0879 |
| B | macro_f1 | T3-T2 | -0.2112 |
| B | macro_f1 | T4-T3 | -0.1209 |
| B | macro_f1 | T5-T4 | -0.0608 |
| B | balanced_accuracy | T3-T2 | -0.1704 |
| B | balanced_accuracy | T4-T3 | -0.1173 |
| B | balanced_accuracy | T5-T4 | -0.0680 |
| D | macro_f1 | T3-T2 | -0.2155 |
| D | macro_f1 | T4-T3 | -0.1095 |
| D | macro_f1 | T5-T4 | -0.0574 |
| D | balanced_accuracy | T3-T2 | -0.1648 |
| D | balanced_accuracy | T4-T3 | -0.1060 |
| D | balanced_accuracy | T5-T4 | -0.0712 |

## Ordinal behaviour

| target | metric | A_single | C_matched | B_full | D_temporal |
| --- | --- | --- | --- | --- | --- |
| T3 | mean_absolute_class_error | 0.4806 | 0.4733 | 0.4917 | 0.5008 |
| T3 | within_one_class_accuracy | 0.8880 | 0.8835 | 0.8744 | 0.8723 |
| T4 | mean_absolute_class_error | 0.5663 | 0.5839 | 0.6331 | 0.6215 |
| T4 | within_one_class_accuracy | 0.8583 | 0.8457 | 0.8177 | 0.8224 |
| T5 | mean_absolute_class_error | 0.5772 | 0.6213 | 0.6624 | 0.6348 |
| T5 | within_one_class_accuracy | 0.8520 | 0.8318 | 0.8049 | 0.8171 |

Ordinal distance is measured in task-specific class indices, not PHQ points. T3/T4/T5 units differ; absolute MAE cannot be compared as an identical unit across these tasks.

## Results interpretation

1. **Pooling did not consistently improve the primary metric.** B minus A Macro-F1 was +0.0024 (T2), +0.0019 (T3), -0.0159 (T4), and -0.0015 (T5). Balanced Accuracy increased by +0.0078, +0.0151, +0.0036, and +0.0235, respectively. These are descriptive contrasts, not significance tests. Lower fold SD in B is descriptive and does not establish an independent confidence interval or universal stability.

2. **Rare-class recognition improved selectively, with substantial errors remaining.** T5 C3 recall increased from 0.1515 (20/132) in A to 0.2869 (101/352) in B, but C3 F1 was nearly unchanged (0.1146 versus 0.1152). C3 precision decreased from 0.0922 to 0.0721. C4 recall increased from 0.0545 (3/55) to 0.1377 (23/167), and F1 from 0.0600 to 0.0962. Both severe classes still had low F1. Meanwhile, C1 F1 decreased from 0.1762 to 0.1544 and C2 F1 from 0.1789 to 0.1423. Thus better rare-class recall did not translate into better overall Macro-F1. This pattern is consistent with increased absolute rare-class sample size as one possible contributing factor, but does not isolate its effect from cycle composition and feature distributions.

3. **Matched-N does not support a clear pooled Macro-F1 advantage.** C sample-level CV Macro-F1 means were 0.6503, 0.4440, 0.3401, and 0.2700 for T2–T5. They were close to A overall; B did not exceed C consistently, especially for T4/T5. T5 C4 recall varied strongly across matched samples (0.0891 ± 0.0850), indicating sensitivity to which scarce severe participants were included. This is a robustness/domain-diversity comparison, not causal proof about sample size or source cycle.

4. **Granularity difficulty persisted in every setting.** Macro-F1 and Balanced Accuracy decreased monotonically from T2 to T5 in A, B, and D. The largest Macro-F1 drop was T2→T3: -0.2107, -0.2112, and -0.2155, respectively. Even at N=14,581, B T5 Macro-F1 was 0.2619. Expanding the pooled cohort under this fixed RF/common31 framework was insufficient to eliminate fine-grained severity classification difficulty. In B, 1,210/2,327 mild cases were predicted minimal and 108/352 C3 cases were predicted C2.

5. **Temporal results were numerically near the CV results, but fine-grained prediction remained weak.** D Macro-F1 was 0.6500, 0.4345, 0.3250, and 0.2676. There was no large aggregate deterioration relative to B in this one temporal comparison. However, T5 C4 identification was only 9/53 and C3 identification 23/99. One evaluated later cycle does not establish stable temporal generalization. D was never used to change features, preprocessing, RF parameters, or decision thresholds.

6. **Pooled-model cycle differences were modest for T5 Macro-F1, but class behaviour differed.** B OOF T5 Macro-F1 was 0.2617, 0.2641, and 0.2593 in 2011–2012, 2013–2014, and 2015–2016. Their C3 recalls were 0.2149, 0.3030, and 0.3535. This does not indicate an isolated aggregate collapse in one cycle, but pooled Macro-F1 alone would conceal class-specific variation. Ordinal error also did not uniformly improve with pooling: T5 mean absolute class error increased from 0.5772 in A to 0.6624 in B, and within-one-class accuracy decreased from 0.8520 to 0.8049. These distances remain task-specific class-index measures.

Accuracy is secondary throughout. T2 represents minimal symptoms versus symptomatic, not a disease/no-disease distinction. No population, clinical-effectiveness, or causal claim is made.

## Verification

- 244 RF fits completed: 20 in A, 20 in B, 200 in C, and 4 in D. No XGBoost experiment, tuning, over/undersampling of training folds, voting, stacking, or model-selection step was performed.
- All 11 new protocol/result tests passed, including exact sampling counts, target cutoffs, participant disjointness, saved-prediction metric recalculation, and training-only preprocessing statistics for all 244 fits.
- The project regression suite ran 70 tests: 69 passed and one legacy XGBoost-fitting test was explicitly skipped to respect the RF-only restriction. Legacy split-test write attempts were intercepted and compared with existing bytes/content rather than written.
- 912 pre-existing files remained checksum-identical. All cohort inputs, frozen experiment code, and configuration hashes were verified. Existing Kaggle model/results, README, and main reports were not changed.

## Limitations

- A and B differ in participants and overlap because B contains A. D test participants also occur in B/C analyses. These are separate frozen evaluations, not independent replications, and D must not be described as an untouched external dataset.
- C samples overlap each other. Their SD captures sampling sensitivity conditional on this external-cycle pool, not an independent-sample confidence interval. Matching N and class counts does not match all demographics, covariates or measurement conditions.
- RF hyperparameters were previously selected in the legacy 2013–2014 project. Reusing them controls the algorithm here but is not nested-CV evaluation of the entire historical selection process. Legacy 34-feature performance is not used as the control.
- Survey weights, clustering and strata are not incorporated; results describe the complete-PHQ adult cohorts, not population prevalence or design-based uncertainty. Random CV is not household/PSU-grouped.
- Laboratory platform/site changes, questionnaire context, missingness and distribution shifts may contribute to cycle differences. Sparse severe classes remain a limitation; overlap and imbalance effects are not isolated.
- The class_weight=balanced_subsample rule is fixed, but effective bootstrap weights change with target granularity and class frequencies. No training-fold over/undersampling was used; C cohort sampling is the prespecified matched control.
- Associations between larger pooled data and performance are descriptive, not causal evidence. No model/configuration is changed in response to these results.

## Files

All tables are in `tables/`, all figures in `figures/`, and protocol/checkpoint hashes in `artifacts/multicycle/`. Existing Kaggle results, model artifacts, main report and README are unchanged.
