"""Soft Hierarchical Classification & Cost-Sensitive Learning for DepSev-ML.

Research Objectives:
1. Address error propagation from hard sequential gating.
2. Formulate posterior probability fusion over 3 classes:
   P(C0 | X) = 1 - p1
   P(C1 | X) = p1 * (1 - p2)
   P(C2+ | X) = p1 * p2
3. Investigate Cost-Sensitive Decision Making via Bayes Risk minimization:
   R(j | X) = sum_{i} P(Ci | X) * K[i, j]
   y_hat = argmin_j R(j | X)
4. Evaluate Probability Calibration (Sigmoid & Isotonic) with Brier score and Log Loss.
5. Provide strict fair comparisons against E0 (Flat RF), E1 (Hard RF-RF), E2, E3a, E3b.
6. Evaluate Multi-cycle Generalization on Branch A (N=5,054), Branch B (N=14,581), and Branch D (N=4,888).
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
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# DepSev-ML internal modules
import final_protocol as frozen
from config import ARTIFACTS_DIR, PROJECT_ROOT, TABLES_DIR
import multicycle_protocol as mp
from utils import load_json

EXPERIMENT_DIR = PROJECT_ROOT / "experiments/soft_hierarchical_t3"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Canonical T3 target definition
T3_CLASSES = [0, 1, 2]
T3_LABELS = ["C0 (Minimal)", "C1 (Mild)", "C2+ (Severe conv.)"]
T3_RANGES = ["0-4", "5-9", "10-27"]

# Standard candidate cost matrices
# Row i = true class (0: C0, 1: C1, 2: C2+), Column j = predicted class
CANDIDATE_COST_MATRICES = {
    # M0: Uniform 0-1 loss (mathematically identical to argmax)
    "M0_Uniform": np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]),
    # M1: Moderate clinical penalty
    "M1_Moderate": np.array([
        [0.0, 1.0, 1.5],
        [1.5, 0.0, 1.0],
        [3.0, 1.5, 0.0],
    ]),
    # M2: High severe penalty
    "M2_High": np.array([
        [0.0, 1.0, 2.0],
        [2.0, 0.0, 1.0],
        [5.0, 2.0, 0.0],
    ]),
    # M3: Extreme severe penalty
    "M3_Extreme": np.array([
        [0.0, 1.0, 2.5],
        [2.5, 0.0, 1.0],
        [8.0, 3.0, 0.0],
    ]),
}


class SoftHierarchicalT3Classifier(BaseEstimator, ClassifierMixin):
    """Soft Hierarchical Classifier with joint probability fusion and cost-sensitive risk minimization."""

    def __init__(
        self,
        stage1_estimator: Pipeline,
        stage2_estimator: Pipeline,
        cost_matrix: Optional[np.ndarray] = None,
        calibrate: bool = False,
        calibration_method: str = "sigmoid",
    ):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.cost_matrix = cost_matrix
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: pd.DataFrame, s: pd.Series) -> SoftHierarchicalT3Classifier:
        """Fit Stage 1 on all data and Stage 2 strictly on symptomatic training subset (PHQ-9 >= 5)."""
        # 1. Stage 1 binary target: 0 if PHQ9 < 5, 1 if PHQ9 >= 5
        y1 = (s >= 5).astype(int)
        base_s1 = clone(self.stage1_estimator)
        if self.calibrate:
            self.stage1_ = CalibratedClassifierCV(base_s1, method=self.calibration_method, cv=3)
        else:
            self.stage1_ = base_s1
        self.stage1_.fit(X, y1)

        # 2. Stage 2 binary target: 0 if 5 <= PHQ9 <= 9, 1 if PHQ9 >= 10
        symptomatic_mask = (s >= 5)
        X2 = X.loc[symptomatic_mask]
        s2 = s.loc[symptomatic_mask]
        y2 = (s2 >= 10).astype(int)

        base_s2 = clone(self.stage2_estimator)
        if self.calibrate:
            self.stage2_ = CalibratedClassifierCV(base_s2, method=self.calibration_method, cv=3)
        else:
            self.stage2_ = base_s2
        self.stage2_.fit(X2, y2)

        return self

    def predict_proba_stages(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Obtain Stage 1 and Stage 2 positive-class posterior probabilities."""
        p1 = self.stage1_.predict_proba(X)[:, 1]
        p2 = self.stage2_.predict_proba(X)[:, 1]
        return p1, p2

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute exact joint 3-class posterior distribution via probability fusion."""
        p1, p2 = self.predict_proba_stages(X)
        p_c0 = 1.0 - p1
        p_c1 = p1 * (1.0 - p2)
        p_c2 = p1 * p2
        probs = np.column_stack([p_c0, p_c1, p_c2])
        # Ensure non-negativity and exact sum-to-one
        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 3 classes using Bayes risk minimization (or argmax if cost matrix is None)."""
        probs = self.predict_proba(X)
        if self.cost_matrix is None:
            return np.argmax(probs, axis=1)
        expected_risks = probs @ self.cost_matrix
        return np.argmin(expected_risks, axis=1)

    def predict_with_cost_matrix(self, X: pd.DataFrame, cost_matrix: np.ndarray) -> np.ndarray:
        """Predict classes for an arbitrary supplied cost matrix."""
        probs = self.predict_proba(X)
        expected_risks = probs @ cost_matrix
        return np.argmin(expected_risks, axis=1)


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[int] = (0, 1, 2),
) -> Dict[str, float]:
    """Calculate all primary, secondary, and per-class metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0))
    macro_p = float(precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    error = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    mae = float(error.mean())
    within_one = float((error <= 1).mean())

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "mean_absolute_class_error": mae,
        "within_one_class_accuracy": within_one,
    }
    for i, c in enumerate(classes):
        metrics[f"c{c}_precision"] = float(prec_per_class[i])
        metrics[f"c{c}_recall"] = float(rec_per_class[i])
        metrics[f"c{c}_f1"] = float(f1_per_class[i])

    return metrics


def compute_probability_calibration_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """Evaluate multiclass Brier score and Log Loss for probability calibration."""
    n_samples, n_classes = probs.shape
    # One-hot encode true labels
    y_onehot = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        y_onehot[:, i] = (y_true == i).astype(float)

    # Multi-class Brier score: sum of squared differences across all classes, averaged over samples
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    # Clip probabilities for stable log loss computation
    probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    loss = float(log_loss(y_true, probs_clipped, labels=[0, 1, 2]))

    # Per-class Brier score
    brier_per_class = {}
    for c in range(n_classes):
        brier_per_class[f"c{c}_brier"] = float(brier_score_loss(y_onehot[:, c], probs[:, c]))

    metrics = {
        "multiclass_brier_score": brier,
        "multiclass_log_loss": loss,
    }
    metrics.update(brier_per_class)
    return metrics


def optimize_cost_matrix_inner_cv(
    X_train: pd.DataFrame,
    s_train: pd.Series,
    spec: Dict[str, Any],
    inventory: Dict[str, Any],
    model_type: str = "RF_RF",
    n_splits: int = 3,
) -> Tuple[str, np.ndarray, pd.DataFrame]:
    """Inner CV selection of cost matrix with ZERO outer test leakage."""
    y_t3_train = np.digitize(s_train, [5, 10])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    inner_probs = np.zeros((len(X_train), 3))

    for in_tr, in_va in skf.split(X_train, y_t3_train):
        X_in_tr, s_in_tr = X_train.iloc[in_tr], s_train.iloc[in_tr]
        X_in_va = X_train.iloc[in_va]

        p_s1 = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", RandomForestClassifier(**spec["hyperparameters"])),
        ])

        if model_type == "RF_RF":
            p_s2 = Pipeline([
                ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
                ("model", RandomForestClassifier(**spec["hyperparameters"])),
            ])
        else:  # RF_LR
            p_s2 = Pipeline([
                ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ])

        clf = SoftHierarchicalT3Classifier(p_s1, p_s2)
        clf.fit(X_in_tr, s_in_tr)
        inner_probs[in_va] = clf.predict_proba(X_in_va)

    # Evaluate candidate matrices on out-of-fold inner probabilities
    grid_records = []
    best_score = -1.0
    best_name = "M0_Uniform"
    best_matrix = CANDIDATE_COST_MATRICES["M0_Uniform"]

    for name, K in CANDIDATE_COST_MATRICES.items():
        risks = inner_probs @ K
        preds = np.argmin(risks, axis=1)
        score = float(f1_score(y_t3_train, preds, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_t3_train, preds))
        c2_rec = float(recall_score(y_t3_train == 2, preds == 2, zero_division=0))
        grid_records.append({
            "matrix_name": name,
            "inner_macro_f1": score,
            "inner_balanced_accuracy": bal_acc,
            "inner_c2_recall": c2_rec,
        })
        if score > best_score:
            best_score = score
            best_name = name
            best_matrix = K

    return best_name, best_matrix, pd.DataFrame(grid_records)


def run_legacy_experiments() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute complete suite of Soft Hierarchical & Cost-Sensitive models on Legacy Cohort (N=4,045)."""
    gran = sys.modules.get("09_granularity_experiment")
    if gran is None:
        import importlib
        gran = importlib.import_module("09_granularity_experiment")

    X, y_c0_c4, inventory, scores, folds, assignments, spec = gran.load_development()
    y_t3 = gran.map_target(y_c0_c4, "T3").to_numpy(dtype=int)

    fold_metric_rows = []
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
        "E1_Hierarchical_RF",
        "E3a_Hierarchical_RF_LR",
        "E4_Soft_RF_RF",
        "E5_Soft_RF_LR",
        "E6a_Soft_RF_RF_CostSensitive",
        "E6b_Soft_RF_LR_CostSensitive",
        "E4_Calibrated_Sigmoid",
        "E5_Calibrated_Sigmoid",
    ]
    for m in models_to_eval:
        oof_predictions[f"{m}_pred"] = np.zeros(len(X), dtype=int)

    prob_models = ["E4_Soft_RF_RF", "E5_Soft_RF_LR", "E4_Calibrated_Sigmoid", "E5_Calibrated_Sigmoid"]
    for pm in prob_models:
        for c in [0, 1, 2]:
            prob_predictions[f"{pm}_prob_c{c}"] = np.zeros(len(X), dtype=float)

    # For sensitivity matrix tracking
    sensitivity_records = []

    print("\n>>> Running Soft Hierarchical & Cost-Sensitive Experiments (Legacy Cohort N=4,045)...")

    for fold_number, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"  Processing Fold {fold_number}/5...", flush=True)
        X_train, y_train, s_train = X.iloc[train_idx], y_t3[train_idx], scores.iloc[train_idx]
        X_val, y_val, s_val = X.iloc[val_idx], y_t3[val_idx], scores.iloc[val_idx]

        # Shared Pipeline Builders
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

        # --- E0: Flat RF ---
        p_e0 = make_rf()
        p_e0.fit(X_train, y_train)
        pred_e0 = p_e0.predict(X_val)
        oof_predictions["E0_Flat_RF_pred"][val_idx] = pred_e0
        m_e0 = calculate_comprehensive_metrics(y_val, pred_e0)
        m_e0.update({"experiment": "Legacy_Dev", "model": "E0_Flat_RF", "fold": fold_number})
        fold_metric_rows.append(m_e0)

        # --- E1: Hard Hierarchical RF-RF ---
        p_s1_hard = make_rf()
        p_s2_rf_hard = make_rf()
        import importlib
        H14 = importlib.import_module("14_hierarchical_t3_experiment")
        clf_e1 = H14.HierarchicalT3Classifier(p_s1_hard, p_s2_rf_hard, 0.5, 0.5)
        clf_e1.fit(X_train, s_train)
        pred_e1 = clf_e1.predict(X_val)
        oof_predictions["E1_Hierarchical_RF_pred"][val_idx] = pred_e1
        m_e1 = calculate_comprehensive_metrics(y_val, pred_e1)
        m_e1.update({"experiment": "Legacy_Dev", "model": "E1_Hierarchical_RF", "fold": fold_number})
        fold_metric_rows.append(m_e1)

        # --- E3a: Hard RF-LR ---
        clf_e3a = H14.HierarchicalT3Classifier(make_rf(), make_lr(), 0.5, 0.5)
        clf_e3a.fit(X_train, s_train)
        pred_e3a = clf_e3a.predict(X_val)
        oof_predictions["E3a_Hierarchical_RF_LR_pred"][val_idx] = pred_e3a
        m_e3a = calculate_comprehensive_metrics(y_val, pred_e3a)
        m_e3a.update({"experiment": "Legacy_Dev", "model": "E3a_Hierarchical_RF_LR", "fold": fold_number})
        fold_metric_rows.append(m_e3a)

        # --- E4: Soft Hierarchical RF-RF (Argmax) ---
        clf_e4 = SoftHierarchicalT3Classifier(make_rf(), make_rf())
        clf_e4.fit(X_train, s_train)
        probs_e4 = clf_e4.predict_proba(X_val)
        pred_e4 = np.argmax(probs_e4, axis=1)
        oof_predictions["E4_Soft_RF_RF_pred"][val_idx] = pred_e4
        for c in range(3):
            prob_predictions[f"E4_Soft_RF_RF_prob_c{c}"][val_idx] = probs_e4[:, c]
        m_e4 = calculate_comprehensive_metrics(y_val, pred_e4)
        m_e4.update({"experiment": "Legacy_Dev", "model": "E4_Soft_RF_RF", "fold": fold_number})
        fold_metric_rows.append(m_e4)

        # --- E5: Soft Hierarchical RF-LR (Argmax) ---
        clf_e5 = SoftHierarchicalT3Classifier(make_rf(), make_lr())
        clf_e5.fit(X_train, s_train)
        probs_e5 = clf_e5.predict_proba(X_val)
        pred_e5 = np.argmax(probs_e5, axis=1)
        oof_predictions["E5_Soft_RF_LR_pred"][val_idx] = pred_e5
        for c in range(3):
            prob_predictions[f"E5_Soft_RF_LR_prob_c{c}"][val_idx] = probs_e5[:, c]
        m_e5 = calculate_comprehensive_metrics(y_val, pred_e5)
        m_e5.update({"experiment": "Legacy_Dev", "model": "E5_Soft_RF_LR", "fold": fold_number})
        fold_metric_rows.append(m_e5)

        # --- E6a: Soft RF-RF + Cost-Sensitive (Inner CV Tuned) ---
        best_name_a, best_mat_a, _ = optimize_cost_matrix_inner_cv(
            X_train, s_train, spec, inventory, model_type="RF_RF", n_splits=3
        )
        pred_e6a = clf_e4.predict_with_cost_matrix(X_val, best_mat_a)
        oof_predictions["E6a_Soft_RF_RF_CostSensitive_pred"][val_idx] = pred_e6a
        m_e6a = calculate_comprehensive_metrics(y_val, pred_e6a)
        m_e6a.update({
            "experiment": "Legacy_Dev",
            "model": "E6a_Soft_RF_RF_CostSensitive",
            "fold": fold_number,
            "selected_matrix": best_name_a,
        })
        fold_metric_rows.append(m_e6a)

        # --- E6b: Soft RF-LR + Cost-Sensitive (Inner CV Tuned) ---
        best_name_b, best_mat_b, _ = optimize_cost_matrix_inner_cv(
            X_train, s_train, spec, inventory, model_type="RF_LR", n_splits=3
        )
        pred_e6b = clf_e5.predict_with_cost_matrix(X_val, best_mat_b)
        oof_predictions["E6b_Soft_RF_LR_CostSensitive_pred"][val_idx] = pred_e6b
        m_e6b = calculate_comprehensive_metrics(y_val, pred_e6b)
        m_e6b.update({
            "experiment": "Legacy_Dev",
            "model": "E6b_Soft_RF_LR_CostSensitive",
            "fold": fold_number,
            "selected_matrix": best_name_b,
        })
        fold_metric_rows.append(m_e6b)

        # --- Calibration Sub-experiments (Sigmoid / Platt) ---
        clf_cal_e4 = SoftHierarchicalT3Classifier(make_rf(), make_rf(), calibrate=True, calibration_method="sigmoid")
        clf_cal_e4.fit(X_train, s_train)
        probs_cal_e4 = clf_cal_e4.predict_proba(X_val)
        pred_cal_e4 = np.argmax(probs_cal_e4, axis=1)
        oof_predictions["E4_Calibrated_Sigmoid_pred"][val_idx] = pred_cal_e4
        for c in range(3):
            prob_predictions[f"E4_Calibrated_Sigmoid_prob_c{c}"][val_idx] = probs_cal_e4[:, c]
        m_cal_e4 = calculate_comprehensive_metrics(y_val, pred_cal_e4)
        m_cal_e4.update({"experiment": "Legacy_Dev", "model": "E4_Calibrated_Sigmoid", "fold": fold_number})
        fold_metric_rows.append(m_cal_e4)

        clf_cal_e5 = SoftHierarchicalT3Classifier(make_rf(), make_lr(), calibrate=True, calibration_method="sigmoid")
        clf_cal_e5.fit(X_train, s_train)
        probs_cal_e5 = clf_cal_e5.predict_proba(X_val)
        pred_cal_e5 = np.argmax(probs_cal_e5, axis=1)
        oof_predictions["E5_Calibrated_Sigmoid_pred"][val_idx] = pred_cal_e5
        for c in range(3):
            prob_predictions[f"E5_Calibrated_Sigmoid_prob_c{c}"][val_idx] = probs_cal_e5[:, c]
        m_cal_e5 = calculate_comprehensive_metrics(y_val, pred_cal_e5)
        m_cal_e5.update({"experiment": "Legacy_Dev", "model": "E5_Calibrated_Sigmoid", "fold": fold_number})
        fold_metric_rows.append(m_cal_e5)

        # Evaluate all candidate cost matrices for sensitivity analysis
        for mat_name, K_mat in CANDIDATE_COST_MATRICES.items():
            for base_model_name, clf_obj in [("RF_RF", clf_e4), ("RF_LR", clf_e5)]:
                preds_mat = clf_obj.predict_with_cost_matrix(X_val, K_mat)
                m_mat = calculate_comprehensive_metrics(y_val, preds_mat)
                m_mat.update({
                    "fold": fold_number,
                    "base_model": base_model_name,
                    "cost_matrix": mat_name,
                    "c2_missed_as_c0": int(((y_val == 2) & (preds_mat == 0)).sum()),
                    "c1_missed_as_c0": int(((y_val == 1) & (preds_mat == 0)).sum()),
                    "c0_false_alarms": int(((y_val == 0) & (preds_mat != 0)).sum()),
                })
                sensitivity_records.append(m_mat)

    df_folds = pd.DataFrame(fold_metric_rows)
    df_oof = pd.DataFrame(oof_predictions)
    df_probs = pd.DataFrame(prob_predictions)
    df_sens = pd.DataFrame(sensitivity_records)

    # Compute Calibration Brier and Log Loss across entire OOF probabilities
    cal_metric_rows = []
    for pm in prob_models:
        p_mat = df_probs[[f"{pm}_prob_c{c}" for c in range(3)]].to_numpy()
        cal_m = compute_probability_calibration_metrics(df_probs["y_true"].to_numpy(), p_mat)
        cal_m["model"] = pm
        cal_metric_rows.append(cal_m)
    df_cal_summary = pd.DataFrame(cal_metric_rows)

    return df_folds, df_oof, df_probs, df_sens, df_cal_summary


def run_multicycle_experiments(
    frames: Dict[str, pd.DataFrame],
    policy: Dict[str, Any],
    old_spec: Dict[str, Any],
) -> pd.DataFrame:
    """Execute E0, E1, E4, E5, E6a, E6b across Multi-cycle Branch A, B, and D."""
    rows = []

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
        pred_flat = p_flat.predict(X_va)
        m = calculate_comprehensive_metrics(y_va, pred_flat)
        m.update({"branch": "A_SingleCycle", "model": "E0_Flat_RF", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E4 Soft RF-RF
        clf_e4 = SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_rf())
        clf_e4.fit(X_tr, s_tr)
        pred_e4 = clf_e4.predict(X_va)
        m = calculate_comprehensive_metrics(y_va, pred_e4)
        m.update({"branch": "A_SingleCycle", "model": "E4_Soft_RF_RF", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E5 Soft RF-LR
        clf_e5 = SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_lr())
        clf_e5.fit(X_tr, s_tr)
        pred_e5 = clf_e5.predict(X_va)
        m = calculate_comprehensive_metrics(y_va, pred_e5)
        m.update({"branch": "A_SingleCycle", "model": "E5_Soft_RF_LR", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E6a Soft RF-RF + Cost M2
        pred_e6a = clf_e4.predict_with_cost_matrix(X_va, CANDIDATE_COST_MATRICES["M2_High"])
        m = calculate_comprehensive_metrics(y_va, pred_e6a)
        m.update({"branch": "A_SingleCycle", "model": "E6a_Soft_RF_RF_CostM2", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m)

        # E6b Soft RF-LR + Cost M2
        pred_e6b = clf_e5.predict_with_cost_matrix(X_va, CANDIDATE_COST_MATRICES["M2_High"])
        m = calculate_comprehensive_metrics(y_va, pred_e6b)
        m.update({"branch": "A_SingleCycle", "model": "E6b_Soft_RF_LR_CostM2", "fold": fold, "n_samples": len(frame_a)})
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
        pred_flat = p_flat.predict(X_va)
        m = calculate_comprehensive_metrics(y_va, pred_flat)
        m.update({"branch": "B_Pooled", "model": "E0_Flat_RF", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        clf_e4 = SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_rf())
        clf_e4.fit(X_tr, s_tr)
        pred_e4 = clf_e4.predict(X_va)
        m = calculate_comprehensive_metrics(y_va, pred_e4)
        m.update({"branch": "B_Pooled", "model": "E4_Soft_RF_RF", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        clf_e5 = SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_lr())
        clf_e5.fit(X_tr, s_tr)
        pred_e5 = clf_e5.predict(X_va)
        m = calculate_comprehensive_metrics(y_va, pred_e5)
        m.update({"branch": "B_Pooled", "model": "E5_Soft_RF_LR", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        pred_e6a = clf_e4.predict_with_cost_matrix(X_va, CANDIDATE_COST_MATRICES["M2_High"])
        m = calculate_comprehensive_metrics(y_va, pred_e6a)
        m.update({"branch": "B_Pooled", "model": "E6a_Soft_RF_RF_CostM2", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m)

        pred_e6b = clf_e5.predict_with_cost_matrix(X_va, CANDIDATE_COST_MATRICES["M2_High"])
        m = calculate_comprehensive_metrics(y_va, pred_e6b)
        m.update({"branch": "B_Pooled", "model": "E6b_Soft_RF_LR_CostM2", "fold": fold, "n_samples": len(frame_b)})
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
    pred_flat_d = p_flat.predict(X_te_d)
    m = calculate_comprehensive_metrics(y_te_d, pred_flat_d)
    m.update({"branch": "D_Temporal", "model": "E0_Flat_RF", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E4
    clf_e4_d = SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_rf())
    clf_e4_d.fit(X_tr_d, s_tr_d)
    pred_e4_d = clf_e4_d.predict(X_te_d)
    m = calculate_comprehensive_metrics(y_te_d, pred_e4_d)
    m.update({"branch": "D_Temporal", "model": "E4_Soft_RF_RF", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E5
    clf_e5_d = SoftHierarchicalT3Classifier(make_mc_rf(), make_mc_lr())
    clf_e5_d.fit(X_tr_d, s_tr_d)
    pred_e5_d = clf_e5_d.predict(X_te_d)
    m = calculate_comprehensive_metrics(y_te_d, pred_e5_d)
    m.update({"branch": "D_Temporal", "model": "E5_Soft_RF_LR", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E6a
    pred_e6a_d = clf_e4_d.predict_with_cost_matrix(X_te_d, CANDIDATE_COST_MATRICES["M2_High"])
    m = calculate_comprehensive_metrics(y_te_d, pred_e6a_d)
    m.update({"branch": "D_Temporal", "model": "E6a_Soft_RF_RF_CostM2", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    # E6b
    pred_e6b_d = clf_e5_d.predict_with_cost_matrix(X_te_d, CANDIDATE_COST_MATRICES["M2_High"])
    m = calculate_comprehensive_metrics(y_te_d, pred_e6b_d)
    m.update({"branch": "D_Temporal", "model": "E6b_Soft_RF_LR_CostM2", "fold": 1, "n_samples": len(test_d)})
    rows.append(m)

    return pd.DataFrame(rows)


def summarize_fold_metrics(df_folds: pd.DataFrame, group_cols: List[str] = ["model"]) -> pd.DataFrame:
    """Aggregate fold metrics to mean and sample standard deviation (ddof=1)."""
    metric_cols = [c for c in df_folds.columns if c not in group_cols + ["fold", "experiment", "branch", "n_samples", "selected_matrix", "base_model", "cost_matrix"]]
    records = []
    for keys, sub in df_folds.groupby(group_cols, as_index=False):
        row = {}
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                row[col] = val
        else:
            row[group_cols[0]] = keys

        for col in metric_cols:
            if col in sub:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"] = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records)


def generate_visualizations(
    df_legacy_summary: pd.DataFrame,
    df_oof: pd.DataFrame,
    df_probs: pd.DataFrame,
    df_sens: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_mc_summary: pd.DataFrame,
):
    """Generate publication-ready figures for Soft Hierarchical & Cost-Sensitive Analysis."""
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})

    # Figure 1: Comprehensive Model Comparison (Macro-F1, BalAcc, Accuracy)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    models = ["E0_Flat_RF", "E1_Hierarchical_RF", "E4_Soft_RF_RF", "E5_Soft_RF_LR", "E6a_Soft_RF_RF_CostSensitive", "E6b_Soft_RF_LR_CostSensitive"]
    sub_df = df_legacy_summary[df_legacy_summary["model"].isin(models)].set_index("model").loc[models]

    x = np.arange(len(models))
    w = 0.25
    b1 = ax.bar(x - w, sub_df["macro_f1_mean"], w, yerr=sub_df["macro_f1_std"], label="Macro-F1", color="#2563EB", capsize=3)
    b2 = ax.bar(x, sub_df["balanced_accuracy_mean"], w, yerr=sub_df["balanced_accuracy_std"], label="Balanced Accuracy", color="#0D9488", capsize=3)
    b3 = ax.bar(x + w, sub_df["accuracy_mean"], w, yerr=sub_df["accuracy_std"], label="Accuracy", color="#64748B", capsize=3)

    ax.set_xticks(x)
    labels = ["E0: Flat RF", "E1: Hard RF", "E4: Soft RF\n(Argmax)", "E5: Soft RF-LR\n(Argmax)", "E6a: Soft RF-RF\n(Cost-Sens)", "E6b: Soft RF-LR\n(Cost-Sens)"]
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison: Flat vs Hard vs Soft vs Cost-Sensitive Decision Making")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(FIGURES_DIR / "legacy_model_comparison_e0_to_e6.png", dpi=200)
    plt.close(fig)

    # Figure 2: Cost-Sensitivity Trade-off Curve (C2+ Recall vs C0 False Alarm Rate)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    sens_agg = df_sens.groupby(["base_model", "cost_matrix"], as_index=False).agg({
        "c2_recall": "mean",
        "c0_false_alarms": "mean",
        "macro_f1": "mean",
    })
    # C0 false alarm rate
    sens_agg["c0_fa_rate"] = sens_agg["c0_false_alarms"] / (3016 / 5)

    matrix_order = ["M0_Uniform", "M1_Moderate", "M2_High", "M3_Extreme"]
    for bm, col, mk in [("RF_RF", "#2563EB", "o"), ("RF_LR", "#D97706", "s")]:
        sub = sens_agg[sens_agg["base_model"] == bm].set_index("cost_matrix").loc[matrix_order]
        ax.plot(sub["c0_fa_rate"] * 100, sub["c2_recall"] * 100, marker=mk, color=col, linewidth=2, label=f"Base {bm}")
        for i, txt in enumerate(["M0 (Argmax)", "M1 (Mod)", "M2 (High)", "M3 (Ext)"]):
            ax.annotate(txt, (sub["c0_fa_rate"].iloc[i] * 100 + 0.5, sub["c2_recall"].iloc[i] * 100 - 1.0), fontsize=9)

    ax.set_xlabel("C0 False Alarm Rate (%) [Cost of Over-screening]")
    ax.set_ylabel("C2+ Severe Recall (%) [Clinical Sensitivity]")
    ax.set_title("Cost-Sensitive Pareto Frontier: Sensitivity vs False Alarms")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.savefig(FIGURES_DIR / "cost_tradeoff_curve.png", dpi=200)
    plt.close(fig)

    # Figure 3: Fused Probability Density Distribution by True Class (E4 Soft Model)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    true_labels = df_probs["y_true"].to_numpy()
    p_c0 = df_probs["E4_Soft_RF_RF_prob_c0"].to_numpy()
    p_c1 = df_probs["E4_Soft_RF_RF_prob_c1"].to_numpy()
    p_c2 = df_probs["E4_Soft_RF_RF_prob_c2"].to_numpy()

    for idx, (p_vec, name) in enumerate([(p_c0, "P(C0 | X)"), (p_c1, "P(C1 | X)"), (p_c2, "P(C2+ | X)")]):
        ax = axes[idx]
        for c, color, label in [(0, "#2563EB", "True C0 (Minimal)"), (1, "#0D9488", "True C1 (Mild)"), (2, "#DC2626", "True C2+ (Severe)")]:
            vals = p_vec[true_labels == c]
            ax.hist(vals, bins=25, density=True, alpha=0.4, color=color, label=label)
        ax.set_title(name)
        ax.set_xlabel("Predicted Probability")
        if idx == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.savefig(FIGURES_DIR / "probability_distributions.png", dpi=200)
    plt.close(fig)

    # Figure 4: Multi-cycle Comparison (Branch A, B, D)
    if not df_mc_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        branches = ["A_SingleCycle", "B_Pooled", "D_Temporal"]
        x = np.arange(len(branches))
        width = 0.18

        target_models = ["E0_Flat_RF", "E4_Soft_RF_RF", "E5_Soft_RF_LR", "E6b_Soft_RF_LR_CostM2"]
        colors = ["#64748B", "#2563EB", "#0D9488", "#D97706"]
        names = ["Flat RF", "Soft RF-RF", "Soft RF-LR", "Cost-Sens RF-LR"]

        for idx, (tm, col, lbl) in enumerate(zip(target_models, colors, names)):
            vals = []
            errs = []
            for b in branches:
                sub = df_mc_summary[(df_mc_summary["branch"] == b) & (df_mc_summary["model"] == tm)]
                if not sub.empty:
                    vals.append(sub["macro_f1_mean"].iloc[0])
                    errs.append(sub["macro_f1_std"].iloc[0])
                else:
                    vals.append(0.0)
                    errs.append(0.0)
            ax.bar(x + (idx - 1.5) * width, vals, width, yerr=errs, capsize=3, label=lbl, color=col)

        ax.set_xticks(x)
        ax.set_xticklabels(["Branch A (2013-14)\nN=5,054", "Branch B (Pooled 11-16)\nN=14,581", "Branch D (Temporal 15-16)\nN=4,888"])
        ax.set_ylabel("Macro-F1")
        ax.set_title("Multi-cycle Evaluation: Soft Hierarchical and Cost-Sensitive Models")
        ax.set_ylim(0, 0.6)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.savefig(FIGURES_DIR / "multicycle_soft_comparison.png", dpi=200)
        plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path, class_labels: Sequence[str] = T3_LABELS):
    """Save normalized annotated confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    normalized = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    n = len(class_labels)
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            pct = normalized[i, j] * 100
            color = "white" if normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=15, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DepSev-ML: Soft Hierarchical Classification Runner")
    parser.add_argument("--skip-multicycle", action="store_true", help="Skip Multi-cycle evaluation")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc)
    print("=" * 60)
    print("DepSev-ML: Soft Hierarchical & Cost-Sensitive Experiment Runner")
    print(f"Timestamp UTC: {start_time.isoformat()}")
    print("=" * 60)

    # 1. Run Legacy Development Experiments (N=4,045)
    df_folds, df_oof, df_probs, df_sens, df_cal = run_legacy_experiments()

    df_legacy_summary = summarize_fold_metrics(df_folds, group_cols=["model"])
    df_folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    df_legacy_summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    df_oof.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    df_probs.to_csv(OUTPUT_DIR / "probability_predictions.csv", index=False)
    df_sens.to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)
    df_cal.to_csv(OUTPUT_DIR / "calibration_metrics.csv", index=False)

    # Generate Confusion Matrices for Legacy Models
    models_to_plot = [
        ("E4_Soft_RF_RF_pred", "E4_Soft_RF_RF"),
        ("E5_Soft_RF_LR_pred", "E5_Soft_RF_LR"),
        ("E6a_Soft_RF_RF_CostSensitive_pred", "E6a_Soft_RF_RF_CostSensitive"),
        ("E6b_Soft_RF_LR_CostSensitive_pred", "E6b_Soft_RF_LR_CostSensitive"),
    ]
    for col, name in models_to_plot:
        cm = confusion_matrix(df_oof["y_true"], df_oof[col], labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=T3_LABELS, columns=[f"Pred_{l}" for l in T3_LABELS])
        cm_df.to_csv(CONFUSION_DIR / f"{name}_confusion_matrix.csv")
        plot_confusion_matrix(cm, f"{name}: Out-of-Fold Confusion (N=4,045)", CONFUSION_DIR / f"{name}_confusion.png")

    print("\nLegacy development results complete:")
    for _, row in df_legacy_summary.iterrows():
        print(f"  {row['model']:32s} | Macro-F1: {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | BalAcc: {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | Acc: {row['accuracy_mean']:.4f}")

    # 2. Run Multi-cycle Cohort Experiments
    df_mc_summary = pd.DataFrame()
    if not args.skip_multicycle:
        frames, policy, old_spec = mp.load_inputs()
        df_mc_folds = run_multicycle_experiments(frames, policy, old_spec)
        df_mc_summary = summarize_fold_metrics(df_mc_folds, group_cols=["branch", "model"])
        df_mc_folds.to_csv(OUTPUT_DIR / "multicycle_fold_metrics.csv", index=False)
        df_mc_summary.to_csv(OUTPUT_DIR / "multicycle_comparison.csv", index=False)

        print("\nMulti-cycle results complete:")
        for _, row in df_mc_summary.iterrows():
            print(f"  Branch: {row['branch']:15s} | Model: {row['model']:25s} | Macro-F1: {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | BalAcc: {row['balanced_accuracy_mean']:.4f}")

    # 3. Generate Visualizations
    generate_visualizations(df_legacy_summary, df_oof, df_probs, df_sens, df_cal, df_mc_summary)
    print(f"\nGenerated publication figures in: {FIGURES_DIR}")

    # 4. Save Run Manifest & Metadata
    end_time = datetime.now(timezone.utc)
    manifest = {
        "experiment": "soft_hierarchical_t3_evaluation",
        "started_utc": start_time.isoformat(),
        "completed_utc": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "python_version": sys.version,
        "legacy_development_n": len(df_oof),
        "models_evaluated": list(df_legacy_summary["model"]),
        "multicycle_evaluated": not args.skip_multicycle,
        "files_generated": [str(p.relative_to(PROJECT_ROOT)) for p in OUTPUT_DIR.rglob("*") if p.is_file()],
    }
    with (EXPERIMENT_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    config_info = {
        "candidate_cost_matrices": {k: v.tolist() for k, v in CANDIDATE_COST_MATRICES.items()},
        "calibration_methods": ["sigmoid"],
        "target_classes": T3_CLASSES,
        "target_labels": T3_LABELS,
    }
    with (EXPERIMENT_DIR / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)

    print(f"\nExecution finished in {(end_time - start_time).total_seconds():.1f}s. Manifest written to {EXPERIMENT_DIR / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
