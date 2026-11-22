# UNRESOLVED ISSUES AND TECHNICAL DEBT AUDIT

## 1. Unresolved Research & Methodological Issues

### 1.1 Unweighted Sampling vs Complex Survey Design
- **Status**: Documented as a transparent limitation in Section 3.5 and Section 5.3.
- **Context**: NHANES is a complex multi-stage stratified cluster probability sample with sampling weights (`WTINT2YR`, `WTMEC2YR`), strata (`SDMVSTRA`), and primary sampling units (`SDMVPSU`).
- **Limitation**: The entire machine learning benchmark pipeline in DepSev-ML operates under unweighted Empirical Risk Minimization (ERM). While this aligns with standard computer science and benchmark ML literature, it prevents direct inferential generalization to national population-level prevalence.
- **Action for Future Work**: Implement survey-weighted pseudo-likelihood or weighted loss functions for population-representative risk stratification.

### 1.2 Cross-Sectional Data Limitation
- **Status**: Documented in Section 5.3.
- **Context**: All physiological measurements and PHQ-9 scores are captured at a single cross-sectional point in time.
- **Limitation**: Depression severity exhibits temporal fluctuations, diurnal variations, and episodic courses that cannot be captured without longitudinal repeated measurements.

### 1.3 Peripheral Proxy Signals vs Biological Causality
- **Status**: Documented in Section 4.10 and Section 5.3.
- **Context**: Routine peripheral physiological indicators (blood pressure, blood glucose, lipid panel, kidney function) serve as non-specific proxy indicators of metabolic and systemic inflammatory status rather than direct neurobiological substrates of depressive symptoms.
- **Limitation**: Performance on fine-grained severity levels encounters a Bayes optimal error ceiling without specialized central or molecular biomarkers (e.g., salivary/serum free cortisol, high-sensitivity C-reactive protein, continuous heart rate variability).

---

## 2. Codebase Reproducibility & Artifact Debt

### 2.1 Missing Raw Artifacts in Fresh Clone
- **Status**: Documented in `REPRODUCIBILITY_NOTES.md`.
- **Details**:
  - `data/processed/cohort.csv`: Not tracked in git repository (due to size and CDC data redistribution policy). Can be generated from raw XPT/CSV using ingestion pipelines.
  - `artifacts/final_model.pkl`: Model weights are not committed to git repository; model specifications are fully preserved in `artifacts/final_model_spec.json`.
  - Intermediate numpy arrays (`*.npz`): Not stored in git.

### 2.2 Preserved Executable Scripts
- The experiment execution chain remains 100% executable and self-contained:
  - `src/09_run_hierarchical_classification.py`: Experiments E0, E1, E2, E3a, E3b.
  - `src/10_run_soft_hierarchical.py`: Experiments E4, E5, E6a, E6b.
  - `src/11_run_multicycle_hierarchical.py`: Cross-cycle evaluation of hierarchical models.
  - `src/12_run_ordinal_classification.py`: Experiments E7a and E7b.
  - `src/13_run_multicycle_ordinal.py`: Cross-cycle evaluation of ordinal models.
  - `src/14_audit_and_signoff.py`: Independent audit and sign-off verification.

