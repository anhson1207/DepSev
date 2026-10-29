"""Ordinal Classification (E7) Experiments for DepSev-ML.

Research Objectives:
1. Formulate and evaluate Ordinal Classification models for 3-class PHQ-9 depression severity
   (C0: 0-4 Minimal < C1: 5-9 Mild < C2+: 10-27 Severe convention).
2. E7a: Ordinal Logistic Regression (Proportional Odds / Cumulative Logit model) with
   guaranteed strictly monotonic thresholds (theta0 < theta1).
3. E7b: Cumulative Binary Classification (Frank & Hall 2001) modeling P(Y >= 1) and P(Y >= 2)
   with monotonicity violation detection and least-squares isotonic projection.
4. Comprehensive comparison against Flat (E0), Hard Hierarchical (E1, E3a),
   and Soft Hierarchical + Cost-Sensitive (E4, E5, E6b) baselines.
5. In-depth ordinal error analysis: Class MAE, Quadratic Weighted Kappa (QWK),
   1-step errors vs catastrophic 2-step errors (C0 <-> C2+).
6. Multi-cycle Generalization: Branch A (N=5,054), Branch B (N=14,581), Branch D Temporal (N=4,888).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# DepSev-ML internal modules
import final_protocol as frozen
from config import ARTIFACTS_DIR, PROJECT_ROOT, TABLES_DIR
import multicycle_protocol as mp
from ordinal_models import CumulativeBinaryClassifier, OrdinalLogisticRegression
from utils import load_json

EXPERIMENT_DIR = PROJECT_ROOT / "experiments/ordinal_t3"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
FIGURES_DIR = OUTPUT_DIR / "figures"

T3_CLASSES = [0, 1, 2]
T3_LABELS = ["C0 (Minimal)", "C1 (Mild)", "C2+ (Severe conv.)"]
T3_RANGES = ["0-4", "5-9", "10-27"]


def calculate_ordinal_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
) -> Dict[str, Any]:
    """Calculate standard classification and ordinal-specific metrics."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=classes, average="weighted", zero_division=0))
    macro_p = float(precision_score(y_t, y_p, labels=classes, average="macro", zero_division=0))
    macro_r = float(recall_score(y_t, y_p, labels=classes, average="macro", zero_division=0))

    prec_per_class = precision_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    rec_per_class = recall_score(y_t, y_p, labels=classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_t, y_p, labels=classes, average=None, zero_division=0)

    # Ordinal distance metrics
    abs_err = np.abs(y_t - y_p)
    mae = float(np.mean(abs_err))
    qwk = float(cohen_kappa_score(y_t, y_p, weights="quadratic"))
    within_one = float(np.mean(abs_err <= 1))

    # Error breakdown
    err_0step = int(np.sum(abs_err == 0))
    err_1step = int(np.sum(abs_err == 1))
    err_2step = int(np.sum(abs_err == 2))

    err_1step_rate = float(np.mean(abs_err == 1))
    err_2step_rate = float(np.mean(abs_err == 2))

    # Specific directional transitions
    c0_to_c2 = int(np.sum((y_t == 0) & (y_p == 2)))
    c2_to_c0 = int(np.sum((y_t == 2) & (y_p == 0)))
    c0_to_c1 = int(np.sum((y_t == 0) & (y_p == 1)))
    c1_to_c0 = int(np.sum((y_t == 1) & (y_p == 0)))
    c1_to_c2 = int(np.sum((y_t == 1) & (y_p == 2)))
    c2_to_c1 = int(np.sum((y_t == 2) & (y_p == 1)))

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "class_mae": mae,
        "quadratic_weighted_kappa": qwk,
        "within_one_class_accuracy": within_one,
        "error_1step_rate": err_1step_rate,
        "error_2step_rate": err_2step_rate,
        "error_0step_count": err_0step,
        "error_1step_count": err_1step,
        "error_2step_count": err_2step,
        "c0_to_c2_count": c0_to_c2,
        "c2_to_c0_count": c2_to_c0,
        "c0_to_c1_count": c0_to_c1,
        "c1_to_c0_count": c1_to_c0,
        "c1_to_c2_count": c1_to_c2,
        "c2_to_c1_count": c2_to_c1,
    }

    for i, c in enumerate(classes):
        metrics[f"c{c}_precision"] = float(prec_per_class[i])
        metrics[f"c{c}_recall"] = float(rec_per_class[i])
        metrics[f"c{c}_f1"] = float(f1_per_class[i])

    return metrics


def run_legacy_experiments() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete suite of models on Legacy Cohort (N=4,045)."""
    gran = sys.modules.get("09_granularity_experiment")
    if gran is None:
        import importlib
        gran = importlib.import_module("09_granularity_experiment")

    X, y_c0_c4, inventory, scores, folds, assignments, spec = gran.load_development()
    y_t3 = gran.map_target(y_c0_c4, "T3").to_numpy(dtype=int)

    import importlib
    H14 = importlib.import_module("14_hierarchical_t3_experiment")
    S15 = importlib.import_module("15_soft_hierarchical_experiment")

    fold_metric_rows: List[Dict[str, Any]] = []
    violation_records: List[Dict[str, Any]] = []

    oof_predictions = {
        "SEQN": X.index.to_numpy(),
        "fold": assignments,
        "PHQ9_total": scores.to_numpy(),
        "y_true": y_t3,
    }
    prob_predictions = {
        "SEQN": X.index.to_numpy(),
        "fold": assignments,
        "y_true": y_t3,
    }

    models_to_eval = [
        "E0_Flat_RF",
        "E1_Hard_RF_RF",
        "E3a_Hard_RF_LR",
        "E4_Soft_RF_RF",
        "E5_Soft_RF_LR",
        "E6b_Soft_RF_LR_CostM1",
        "E7a_Ordinal_LogReg",
        "E7a_Ordinal_LogReg_Balanced",
        "E7b_CumBin_LR",
        "E7b_CumBin_RF",
    ]

    for m in models_to_eval:
        oof_predictions[f"{m}_pred"] = np.zeros(len(X), dtype=int)

    prob_models = [
        "E7a_Ordinal_LogReg",
        "E7a_Ordinal_LogReg_Balanced",
        "E7b_CumBin_LR",
        "E7b_CumBin_RF",
    ]
    for pm in prob_models:
        for c in [0, 1, 2]:
            prob_predictions[f"{pm}_prob_c{c}"] = np.zeros(len(X), dtype=float)

    # Cost Matrix M1 for E6b reference
    M1_cost = S15.CANDIDATE_COST_MATRICES["M1_Moderate"]

    print("\n>>> Running Ordinal Classification Experiments on Legacy Cohort (N=4,045)...")

    for fold_number, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"  Processing Fold {fold_number}/5...", flush=True)
        X_train, y_train, s_train = X.iloc[train_idx], y_t3[train_idx], scores.iloc[train_idx]
        X_val, y_val, s_val = X.iloc[val_idx], y_t3[val_idx], scores.iloc[val_idx]

        def make_rf():
            return Pipeline([
                ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
                ("model", RandomForestClassifier(**spec["hyperparameters"])),
            ])

        def make_lr():
            return Pipeline([
                ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ])

        # 1. E0: Flat RF
        p_e0 = make_rf()
        p_e0.fit(X_train, y_train)
        pred_e0 = p_e0.predict(X_val)
        oof_predictions["E0_Flat_RF_pred"][val_idx] = pred_e0
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e0)
        m.update({"model": "E0_Flat_RF", "fold": fold_number})
        fold_metric_rows.append(m)

        # 2. E1: Hard RF-RF
        clf_e1 = H14.HierarchicalT3Classifier(make_rf(), make_rf(), 0.5, 0.5)
        clf_e1.fit(X_train, s_train)
        pred_e1 = clf_e1.predict(X_val)
        oof_predictions["E1_Hard_RF_RF_pred"][val_idx] = pred_e1
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e1)
        m.update({"model": "E1_Hard_RF_RF", "fold": fold_number})
        fold_metric_rows.append(m)

        # 3. E3a: Hard RF-LR
        clf_e3a = H14.HierarchicalT3Classifier(make_rf(), make_lr(), 0.5, 0.5)
        clf_e3a.fit(X_train, s_train)
        pred_e3a = clf_e3a.predict(X_val)
        oof_predictions["E3a_Hard_RF_LR_pred"][val_idx] = pred_e3a
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e3a)
        m.update({"model": "E3a_Hard_RF_LR", "fold": fold_number})
        fold_metric_rows.append(m)

        # 4. E4: Soft RF-RF (Argmax)
        clf_e4 = S15.SoftHierarchicalT3Classifier(make_rf(), make_rf())
        clf_e4.fit(X_train, s_train)
        pred_e4 = clf_e4.predict(X_val)
        oof_predictions["E4_Soft_RF_RF_pred"][val_idx] = pred_e4
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e4)
        m.update({"model": "E4_Soft_RF_RF", "fold": fold_number})
        fold_metric_rows.append(m)

        # 5. E5: Soft RF-LR (Argmax)
        clf_e5 = S15.SoftHierarchicalT3Classifier(make_rf(), make_lr())
        clf_e5.fit(X_train, s_train)
        pred_e5 = clf_e5.predict(X_val)
        oof_predictions["E5_Soft_RF_LR_pred"][val_idx] = pred_e5
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e5)
        m.update({"model": "E5_Soft_RF_LR", "fold": fold_number})
        fold_metric_rows.append(m)

        # 6. E6b: Soft RF-LR + Cost M1
        pred_e6b = clf_e5.predict_with_cost_matrix(X_val, M1_cost)
        oof_predictions["E6b_Soft_RF_LR_CostM1_pred"][val_idx] = pred_e6b
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e6b)
        m.update({"model": "E6b_Soft_RF_LR_CostM1", "fold": fold_number})
        fold_metric_rows.append(m)

        # 7. E7a: Ordinal Logistic Regression (Unweighted)
        p_e7a_unweighted = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", OrdinalLogisticRegression(C=1.0, class_weight=None, max_iter=1000, random_state=42)),
        ])
        p_e7a_unweighted.fit(X_train, y_train)
        probs_e7a_unw = p_e7a_unweighted.predict_proba(X_val)
        pred_e7a_unw = np.argmax(probs_e7a_unw, axis=1)
        oof_predictions["E7a_Ordinal_LogReg_pred"][val_idx] = pred_e7a_unw
        for c in range(3):
            prob_predictions[f"E7a_Ordinal_LogReg_prob_c{c}"][val_idx] = probs_e7a_unw[:, c]
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e7a_unw)
        m.update({"model": "E7a_Ordinal_LogReg", "fold": fold_number})
        fold_metric_rows.append(m)

        # 8. E7a: Ordinal Logistic Regression (Balanced)
        p_e7a_balanced = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", OrdinalLogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
        ])
        p_e7a_balanced.fit(X_train, y_train)
        probs_e7a_bal = p_e7a_balanced.predict_proba(X_val)
        pred_e7a_bal = np.argmax(probs_e7a_bal, axis=1)
        oof_predictions["E7a_Ordinal_LogReg_Balanced_pred"][val_idx] = pred_e7a_bal
        for c in range(3):
            prob_predictions[f"E7a_Ordinal_LogReg_Balanced_prob_c{c}"][val_idx] = probs_e7a_bal[:, c]
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e7a_bal)
        m.update({"model": "E7a_Ordinal_LogReg_Balanced", "fold": fold_number})
        fold_metric_rows.append(m)

        # 9. E7b: Cumulative Binary Classification (Logistic Regression)
        prep_b_lr = frozen.ablation.build_preprocessor(spec["features"], inventory)
        X_tr_trans = prep_b_lr.fit_transform(X_train)
        X_va_trans = prep_b_lr.transform(X_val)

        base_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf_e7b_lr = CumulativeBinaryClassifier(base_lr, apply_projection=True)
        clf_e7b_lr.fit(X_tr_trans, y_train)
        probs_e7b_lr = clf_e7b_lr.predict_proba(X_va_trans)
        pred_e7b_lr = np.argmax(probs_e7b_lr, axis=1)
        oof_predictions["E7b_CumBin_LR_pred"][val_idx] = pred_e7b_lr
        for c in range(3):
            prob_predictions[f"E7b_CumBin_LR_prob_c{c}"][val_idx] = probs_e7b_lr[:, c]
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e7b_lr)
        m.update({
            "model": "E7b_CumBin_LR",
            "fold": fold_number,
            "monotonicity_violation_rate": clf_e7b_lr.last_violation_rate_,
            "monotonicity_violation_count": clf_e7b_lr.last_violation_count_,
        })
        fold_metric_rows.append(m)
        violation_records.append({
            "model": "E7b_CumBin_LR",
            "fold": fold_number,
            "val_samples": len(val_idx),
            "violation_count": clf_e7b_lr.last_violation_count_,
            "violation_rate": clf_e7b_lr.last_violation_rate_,
        })

        # 10. E7b: Cumulative Binary Classification (Random Forest)
        prep_b_rf = frozen.ablation.build_preprocessor(spec["features"], inventory)
        X_tr_trans_rf = prep_b_rf.fit_transform(X_train)
        X_va_trans_rf = prep_b_rf.transform(X_val)

        base_rf = RandomForestClassifier(**spec["hyperparameters"])
        clf_e7b_rf = CumulativeBinaryClassifier(base_rf, apply_projection=True)
        clf_e7b_rf.fit(X_tr_trans_rf, y_train)
        probs_e7b_rf = clf_e7b_rf.predict_proba(X_va_trans_rf)
        pred_e7b_rf = np.argmax(probs_e7b_rf, axis=1)
        oof_predictions["E7b_CumBin_RF_pred"][val_idx] = pred_e7b_rf
        for c in range(3):
            prob_predictions[f"E7b_CumBin_RF_prob_c{c}"][val_idx] = probs_e7b_rf[:, c]
        m = calculate_ordinal_comprehensive_metrics(y_val, pred_e7b_rf)
        m.update({
            "model": "E7b_CumBin_RF",
            "fold": fold_number,
            "monotonicity_violation_rate": clf_e7b_rf.last_violation_rate_,
            "monotonicity_violation_count": clf_e7b_rf.last_violation_count_,
        })
        fold_metric_rows.append(m)
        violation_records.append({
            "model": "E7b_CumBin_RF",
            "fold": fold_number,
            "val_samples": len(val_idx),
            "violation_count": clf_e7b_rf.last_violation_count_,
            "violation_rate": clf_e7b_rf.last_violation_rate_,
        })

    df_folds = pd.DataFrame(fold_metric_rows)
    df_oof = pd.DataFrame(oof_predictions)
    df_probs = pd.DataFrame(prob_predictions)
    df_violations = pd.DataFrame(violation_records)

    # Compute aggregate model comparison
    summary_rows = []
    for model_name, sub in df_folds.groupby("model", sort=False):
        row = {
            "model": model_name,
            "macro_f1_mean": float(sub["macro_f1"].mean()),
            "macro_f1_std": float(sub["macro_f1"].std()),
            "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(sub["balanced_accuracy"].std()),
            "accuracy_mean": float(sub["accuracy"].mean()),
            "accuracy_std": float(sub["accuracy"].std()),
            "class_mae_mean": float(sub["class_mae"].mean()),
            "class_mae_std": float(sub["class_mae"].std()),
            "quadratic_weighted_kappa_mean": float(sub["quadratic_weighted_kappa"].mean()),
            "quadratic_weighted_kappa_std": float(sub["quadratic_weighted_kappa"].std()),
            "within_one_class_accuracy_mean": float(sub["within_one_class_accuracy"].mean()),
            "error_1step_rate_mean": float(sub["error_1step_rate"].mean()),
            "error_2step_rate_mean": float(sub["error_2step_rate"].mean()),
            "c0_precision_mean": float(sub["c0_precision"].mean()),
            "c0_recall_mean": float(sub["c0_recall"].mean()),
            "c0_f1_mean": float(sub["c0_f1"].mean()),
            "c1_precision_mean": float(sub["c1_precision"].mean()),
            "c1_recall_mean": float(sub["c1_recall"].mean()),
            "c1_f1_mean": float(sub["c1_f1"].mean()),
            "c2_precision_mean": float(sub["c2_precision"].mean()),
            "c2_recall_mean": float(sub["c2_recall"].mean()),
            "c2_f1_mean": float(sub["c2_f1"].mean()),
            "total_2step_errors": int(sub["error_2step_count"].sum()),
            "total_c2_to_c0": int(sub["c2_to_c0_count"].sum()),
            "total_c0_to_c2": int(sub["c0_to_c2_count"].sum()),
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    return df_folds, df_summary, df_oof, df_probs, df_violations


def run_multicycle_experiments(
    frames: Dict[str, pd.DataFrame],
    policy: Dict[str, Any],
    old_spec: Dict[str, Any],
) -> pd.DataFrame:
    """Execute complete suite across Multi-cycle Branch A, B, and D."""
    import importlib
    S15 = importlib.import_module("15_soft_hierarchical_experiment")
    H14 = importlib.import_module("14_hierarchical_t3_experiment")

    M2_cost = S15.CANDIDATE_COST_MATRICES["M2_High"]

    rows: List[Dict[str, Any]] = []

    def make_mc_rf():
        return mp.make_pipeline(policy, old_spec)

    def make_mc_lr():
        pipe = mp.make_pipeline(policy, old_spec)
        pipe.steps[-1] = ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        return pipe

    # 1. Branch A: Single-cycle 2013-2014 (N=5,054)
    print("\n>>> Running Multi-cycle Branch A (Single-cycle 2013-2014, N=5,054)...")
    frame_a = frames["2013_2014"]
    folds_a = mp.fixed_folds(frame_a)
    X_a = mp.predictor_frame(frame_a, policy)
    s_a = frame_a.PHQ9_total
    y_a = mp.target_values(s_a, "T3")

    for fold in range(1, 6):
        tr, va = (folds_a != fold), (folds_a == fold)
        X_tr, y_tr, s_tr = X_a.loc[tr], y_a[tr], s_a.loc[tr]
        X_va, y_va, s_va = X_a.loc[va], y_a[va], s_a.loc[va]

        # E0 Flat RF
        p_flat = make_mc_rf()
        p_flat.fit(X_tr, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, p_flat.predict(X_va))
        m.update({"branch": "A_SingleCycle", "model": "E0_Flat_RF", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E1 Hard RF-RF
        clf_e1 = H14.HierarchicalT3Classifier(make_mc_rf(), make_mc_rf(), 0.5, 0.5)
        clf_e1.fit(X_tr, s_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, clf_e1.predict(X_va))
        m.update({"branch": "A_SingleCycle", "model": "E1_Hard_RF_RF", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E4 Soft RF-RF
        clf_e4 = S15.SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_rf())
        clf_e4.fit(X_tr, s_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, clf_e4.predict(X_va))
        m.update({"branch": "A_SingleCycle", "model": "E4_Soft_RF_RF", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E5 Soft RF-LR
        clf_e5 = S15.SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_lr())
        clf_e5.fit(X_tr, s_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, clf_e5.predict(X_va))
        m.update({"branch": "A_SingleCycle", "model": "E5_Soft_RF_LR", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E6b Soft RF-LR Cost M2
        pred_e6b = clf_e5.predict_with_cost_matrix(X_va, M2_cost)
        m = calculate_ordinal_comprehensive_metrics(y_va, pred_e6b)
        m.update({"branch": "A_SingleCycle", "model": "E6b_Soft_RF_LR_CostM2", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E7a Ordinal Logistic Regression (Balanced)
        p_mc_base = make_mc_rf()
        preprocessor = clone(p_mc_base.named_steps["preprocessor"])
        X_tr_trans = preprocessor.fit_transform(X_tr)
        X_va_trans = preprocessor.transform(X_va)

        olr = OrdinalLogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        olr.fit(X_tr_trans, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, olr.predict(X_va_trans))
        m.update({"branch": "A_SingleCycle", "model": "E7a_Ordinal_LogReg_Balanced", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E7b CumBin LR
        base_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        cbc_lr = CumulativeBinaryClassifier(base_lr, apply_projection=True)
        cbc_lr.fit(X_tr_trans, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, cbc_lr.predict(X_va_trans))
        m.update({
            "branch": "A_SingleCycle",
            "model": "E7b_CumBin_LR",
            "fold": fold,
            "n_samples": len(frame_a),
            "monotonicity_violation_rate": cbc_lr.last_violation_rate_,
        })
        rows.append(m)

        # E7b CumBin RF
        base_rf = RandomForestClassifier(**old_spec["hyperparameters"])
        cbc_rf = CumulativeBinaryClassifier(base_rf, apply_projection=True)
        cbc_rf.fit(X_tr_trans, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, cbc_rf.predict(X_va_trans))
        m.update({
            "branch": "A_SingleCycle",
            "model": "E7b_CumBin_RF",
            "fold": fold,
            "n_samples": len(frame_a),
            "monotonicity_violation_rate": cbc_rf.last_violation_rate_,
        })
        rows.append(m)

    # 2. Branch B: Pooled 2011-2016 (N=14,581)
    print(">>> Running Multi-cycle Branch B (Pooled 2011-2016, N=14,581)...")
    frame_b = frames["2011_2016_common31"]
    folds_b = mp.fixed_folds(frame_b)
    X_b = mp.predictor_frame(frame_b, policy)
    s_b = frame_b.PHQ9_total
    y_b = mp.target_values(s_b, "T3")

    for fold in range(1, 6):
        tr, va = (folds_b != fold), (folds_b == fold)
        X_tr, y_tr, s_tr = X_b.loc[tr], y_b[tr], s_b.loc[tr]
        X_va, y_va, s_va = X_b.loc[va], y_b[va], s_b.loc[va]

        p_flat = make_mc_rf()
        p_flat.fit(X_tr, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, p_flat.predict(X_va))
        m.update({"branch": "B_Pooled", "model": "E0_Flat_RF", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        clf_e1 = H14.HierarchicalT3Classifier(make_mc_rf(), make_mc_rf(), 0.5, 0.5)
        clf_e1.fit(X_tr, s_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, clf_e1.predict(X_va))
        m.update({"branch": "B_Pooled", "model": "E1_Hard_RF_RF", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        clf_e4 = S15.SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_rf())
        clf_e4.fit(X_tr, s_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, clf_e4.predict(X_va))
        m.update({"branch": "B_Pooled", "model": "E4_Soft_RF_RF", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        clf_e5 = S15.SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_lr())
        clf_e5.fit(X_tr, s_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, clf_e5.predict(X_va))
        m.update({"branch": "B_Pooled", "model": "E5_Soft_RF_LR", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        pred_e6b = clf_e5.predict_with_cost_matrix(X_va, M2_cost)
        m = calculate_ordinal_comprehensive_metrics(y_va, pred_e6b)
        m.update({"branch": "B_Pooled", "model": "E6b_Soft_RF_LR_CostM2", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        p_mc_base = make_mc_rf()
        preprocessor = clone(p_mc_base.named_steps["preprocessor"])
        X_tr_trans = preprocessor.fit_transform(X_tr)
        X_va_trans = preprocessor.transform(X_va)

        olr = OrdinalLogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        olr.fit(X_tr_trans, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, olr.predict(X_va_trans))
        m.update({"branch": "B_Pooled", "model": "E7a_Ordinal_LogReg_Balanced", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        base_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        cbc_lr = CumulativeBinaryClassifier(base_lr, apply_projection=True)
        cbc_lr.fit(X_tr_trans, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, cbc_lr.predict(X_va_trans))
        m.update({
            "branch": "B_Pooled",
            "model": "E7b_CumBin_LR",
            "fold": fold,
            "n_samples": len(frame_b),
            "monotonicity_violation_rate": cbc_lr.last_violation_rate_,
        })
        rows.append(m)

        base_rf = RandomForestClassifier(**old_spec["hyperparameters"])
        cbc_rf = CumulativeBinaryClassifier(base_rf, apply_projection=True)
        cbc_rf.fit(X_tr_trans, y_tr)
        m = calculate_ordinal_comprehensive_metrics(y_va, cbc_rf.predict(X_va_trans))
        m.update({
            "branch": "B_Pooled",
            "model": "E7b_CumBin_RF",
            "fold": fold,
            "n_samples": len(frame_b),
            "monotonicity_violation_rate": cbc_rf.last_violation_rate_,
        })
        rows.append(m)

    # 3. Branch D: Temporal Generalization (Train 2011-2014 N=9,693, Test 2015-2016 N=4,888)
    print(">>> Running Multi-cycle Branch D (Temporal Generalization N=4,888)...")
    train_d = pd.concat([frames["2011_2012"], frames["2013_2014"]], ignore_index=True)
    test_d = frames["2015_2016"]
    X_tr_d = mp.predictor_frame(train_d, policy)
    s_tr_d = train_d.PHQ9_total
    y_tr_d = mp.target_values(s_tr_d, "T3")
    X_te_d = mp.predictor_frame(test_d, policy)
    s_te_d = test_d.PHQ9_total
    y_te_d = mp.target_values(s_te_d, "T3")

    # E0
    p_flat = make_mc_rf()
    p_flat.fit(X_tr_d, y_tr_d)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, p_flat.predict(X_te_d))
    m.update({"branch": "D_Temporal", "model": "E0_Flat_RF", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E1
    clf_e1_d = H14.HierarchicalT3Classifier(make_mc_rf(), make_mc_rf(), 0.5, 0.5)
    clf_e1_d.fit(X_tr_d, s_tr_d)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, clf_e1_d.predict(X_te_d))
    m.update({"branch": "D_Temporal", "model": "E1_Hard_RF_RF", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E4
    clf_e4_d = S15.SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_rf())
    clf_e4_d.fit(X_tr_d, s_tr_d)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, clf_e4_d.predict(X_te_d))
    m.update({"branch": "D_Temporal", "model": "E4_Soft_RF_RF", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E5
    clf_e5_d = S15.SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_lr())
    clf_e5_d.fit(X_tr_d, s_tr_d)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, clf_e5_d.predict(X_te_d))
    m.update({"branch": "D_Temporal", "model": "E5_Soft_RF_LR", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E6b
    pred_e6b_d = clf_e5_d.predict_with_cost_matrix(X_te_d, M2_cost)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, pred_e6b_d)
    m.update({"branch": "D_Temporal", "model": "E6b_Soft_RF_LR_CostM2", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E7a
    p_mc_base = make_mc_rf()
    preprocessor = clone(p_mc_base.named_steps["preprocessor"])
    X_tr_trans_d = preprocessor.fit_transform(X_tr_d)
    X_te_trans_d = preprocessor.transform(X_te_d)

    olr_d = OrdinalLogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    olr_d.fit(X_tr_trans_d, y_tr_d)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, olr_d.predict(X_te_trans_d))
    m.update({"branch": "D_Temporal", "model": "E7a_Ordinal_LogReg_Balanced", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E7b LR
    base_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    cbc_lr_d = CumulativeBinaryClassifier(base_lr, apply_projection=True)
    cbc_lr_d.fit(X_tr_trans_d, y_tr_d)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, cbc_lr_d.predict(X_te_trans_d))
    m.update({
        "branch": "D_Temporal",
        "model": "E7b_CumBin_LR",
        "fold": 1,
        "n_samples": len(test_d),
        "monotonicity_violation_rate": cbc_lr_d.last_violation_rate_,
    })
    rows.append(m)

    # E7b RF
    base_rf = RandomForestClassifier(**old_spec["hyperparameters"])
    cbc_rf_d = CumulativeBinaryClassifier(base_rf, apply_projection=True)
    cbc_rf_d.fit(X_tr_trans_d, y_tr_d)
    m = calculate_ordinal_comprehensive_metrics(y_te_d, cbc_rf_d.predict(X_te_trans_d))
    m.update({
        "branch": "D_Temporal",
        "model": "E7b_CumBin_RF",
        "fold": 1,
        "n_samples": len(test_d),
        "monotonicity_violation_rate": cbc_rf_d.last_violation_rate_,
    })
    rows.append(m)

    return pd.DataFrame(rows)


def generate_visualizations(
    df_summary: pd.DataFrame,
    df_oof: pd.DataFrame,
    df_probs: pd.DataFrame,
    df_violations: pd.DataFrame,
) -> None:
    """Generate professional scientific plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.sans-serif": "Arial", "figure.dpi": 300})

    # 1. Macro-F1 vs Class MAE Trade-off
    fig, ax = plt.subplots(figsize=(10, 6))
    models = df_summary["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    f1_means = df_summary["macro_f1_mean"].values
    f1_stds = df_summary["macro_f1_std"].values
    mae_means = df_summary["class_mae_mean"].values
    mae_stds = df_summary["class_mae_std"].values

    ax.bar(x - width/2, f1_means, width, yerr=f1_stds, label="Macro-F1 (Higher is better)", color="#2b5c8f", capsize=4)
    ax.bar(x + width/2, mae_means, width, yerr=mae_stds, label="Class MAE (Lower is better)", color="#d95f02", capsize=4)

    ax.set_ylabel("Metric Value")
    ax.set_title("Performance Comparison: Macro-F1 vs Class MAE (Legacy Cohort N=4,045)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "macro_f1_vs_mae.png")
    plt.close()

    # 2. 2-Step Error Comparison (C0 -> C2+ and C2+ -> C0)
    fig, ax = plt.subplots(figsize=(10, 6))
    c2_to_c0 = df_summary["total_c2_to_c0"].values
    c0_to_c2 = df_summary["total_c0_to_c2"].values

    p1 = ax.bar(models, c2_to_c0, label="C2+ -> C0 (Severe Missed as Minimal)", color="#e41a1c")
    p2 = ax.bar(models, c0_to_c2, bottom=c2_to_c0, label="C0 -> C2+ (Minimal False Alarm as Severe)", color="#377eb8")

    ax.set_ylabel("Total Samples Across 5 Folds (N=4,045)")
    ax.set_title("Severe 2-Step Errors: C2+ -> C0 Omissions vs C0 -> C2+ False Alarms")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add count labels
    for i, (miss, fa) in enumerate(zip(c2_to_c0, c0_to_c2)):
        total = miss + fa
        ax.text(i, total + 5, f"{total}\n({miss}/{fa})", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "2step_error_comparison.png")
    plt.close()

    # 3. Confusion Matrices Heatmap for Key Models
    key_models = [
        ("E0_Flat_RF_pred", "E0 Flat RF"),
        ("E1_Hard_RF_RF_pred", "E1 Hard RF-RF"),
        ("E5_Soft_RF_LR_pred", "E5 Soft RF-LR"),
        ("E6b_Soft_RF_LR_CostM1_pred", "E6b Soft Cost M1"),
        ("E7a_Ordinal_LogReg_Balanced_pred", "E7a Ordinal LogReg (Bal)"),
        ("E7b_CumBin_LR_pred", "E7b CumBin LR"),
        ("E7b_CumBin_RF_pred", "E7b CumBin RF"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    y_true = df_oof["y_true"].values
    for idx, (col_name, title) in enumerate(key_models):
        ax = axes[idx]
        y_pred = df_oof[col_name].values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=11, fontweight="bold")
        tick_marks = np.arange(3)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["C0", "C1", "C2+"])
        ax.set_yticklabels(["C0", "C1", "C2+"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontweight="bold")

    # Hide unused 8th subplot
    axes[7].axis("off")
    plt.suptitle("Out-of-Fold Confusion Matrices on Legacy Cohort (N=4,045)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices_comparison.png")
    plt.close()

    # 4. E7b Monotonicity Violation Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    v_lr = df_violations[df_violations["model"] == "E7b_CumBin_LR"]["violation_rate"].values * 100
    v_rf = df_violations[df_violations["model"] == "E7b_CumBin_RF"]["violation_rate"].values * 100

    folds = np.arange(1, 6)
    width = 0.35
    ax.bar(folds - width/2, v_lr, width, label="E7b CumBin LR", color="#4daf4a")
    ax.bar(folds + width/2, v_rf, width, label="E7b CumBin RF", color="#984ea3")

    ax.set_xlabel("Validation Fold")
    ax.set_ylabel("Violation Rate (%) [q2 > q1]")
    ax.set_title("Monotonicity Violation Rates across 5 Validation Folds")
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(5):
        ax.text(folds[i] - width/2, v_lr[i] + 0.2, f"{v_lr[i]:.1f}%", ha="center", fontsize=8)
        ax.text(folds[i] + width/2, v_rf[i] + 0.2, f"{v_rf[i]:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monotonicity_violation_distribution.png")
    plt.close()

    print("  Visualizations saved to:", FIGURES_DIR)


def save_confusion_matrices(df_oof: pd.DataFrame) -> None:
    """Save individual confusion matrix CSVs for all models."""
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    y_true = df_oof["y_true"].values
    pred_cols = [c for c in df_oof.columns if c.endswith("_pred")]

    for col in pred_cols:
        model_name = col.replace("_pred", "")
        y_pred = df_oof[col].values
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(
            cm,
            index=["True_C0", "True_C1", "True_C2+"],
            columns=["Pred_C0", "Pred_C1", "Pred_C2+"],
        )
        cm_df.to_csv(CONFUSION_DIR / f"{model_name}_cm.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)
    print(f"Starting Ordinal Classification Experiment Run at {start_time.isoformat()}...")

    # 1. Run Legacy Cohort Experiments
    df_folds, df_summary, df_oof, df_probs, df_violations = run_legacy_experiments()

    # Save Legacy CSV outputs
    df_folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    df_summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    df_oof.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    df_probs.to_csv(OUTPUT_DIR / "probability_predictions.csv", index=False)
    df_violations.to_csv(OUTPUT_DIR / "e7b_monotonicity_violations.csv", index=False)

    # Per-class metrics table
    per_class_cols = ["model"] + [c for c in df_summary.columns if "c0_" in c or "c1_" in c or "c2_" in c]
    df_per_class = df_summary[per_class_cols].copy()
    df_per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)

    # Ordinal error analysis table
    error_cols = [
        "model",
        "class_mae_mean",
        "class_mae_std",
        "quadratic_weighted_kappa_mean",
        "quadratic_weighted_kappa_std",
        "within_one_class_accuracy_mean",
        "error_1step_rate_mean",
        "error_2step_rate_mean",
        "total_2step_errors",
        "total_c2_to_c0",
        "total_c0_to_c2",
    ]
    df_ordinal_errors = df_summary[error_cols].copy()
    df_ordinal_errors.to_csv(OUTPUT_DIR / "ordinal_error_analysis.csv", index=False)

    save_confusion_matrices(df_oof)
    generate_visualizations(df_summary, df_oof, df_probs, df_violations)

    # 2. Run Multi-cycle Cohort Experiments
    print("\n>>> Loading Multi-cycle Data...")
    frames, policy, old_spec = mp.load_inputs()

    df_multicycle = run_multicycle_experiments(frames, policy, old_spec)
    df_multicycle.to_csv(OUTPUT_DIR / "multicycle_comparison.csv", index=False)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    print(f"\nCompleted in {duration:.1f}s.")

    # 3. Write config and run manifest
    config_dict = {
        "experiment_name": "ordinal_t3",
        "dataset_legacy": "NHANES 2013-2014 Development Cohort (N=4,045)",
        "features_legacy": "F4 (34 features)",
        "cv_scheme": "Stratified 5-Fold CV (fixed folds aligned with E0-E6)",
        "multicycle_cohorts": {
            "Branch_A": "2013-2014 Single-cycle (N=5,054, 5 folds)",
            "Branch_B": "2011-2016 Pooled Multi-cycle (N=14,581, 5 folds)",
            "Branch_D": "Temporal (Train 2011-2014 N=9,693, Test 2015-2016 N=4,888)",
        },
        "multicycle_features": "F4_common31 (31 features)",
        "models_evaluated": [
            "E0_Flat_RF",
            "E1_Hard_RF_RF",
            "E3a_Hard_RF_LR",
            "E4_Soft_RF_RF",
            "E5_Soft_RF_LR",
            "E6b_Soft_RF_LR_CostM1",
            "E7a_Ordinal_LogReg",
            "E7a_Ordinal_LogReg_Balanced",
            "E7b_CumBin_LR",
            "E7b_CumBin_RF",
        ],
    }
    with open(EXPERIMENT_DIR / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    manifest_dict = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "legacy_samples": 4045,
        "branch_a_samples": 5054,
        "branch_b_samples": 14581,
        "branch_d_train_samples": 9693,
        "branch_d_test_samples": 4888,
        "status": "COMPLETED",
    }
    with open(EXPERIMENT_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest_dict, f, indent=2)

    print("\nExperiment completed successfully! All artifacts exported to:", EXPERIMENT_DIR)


if __name__ == "__main__":
    main()

