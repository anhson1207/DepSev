"""Hierarchical Classification (2-stage) for DepSev-ML T3 Depression Severity Prediction.

Architecture:
- Stage 1 (Screening): Binary classification Minimal (0-4 / C0) vs Symptomatic (5-27 / C1, C2+).
  Trained on all development training samples.
- Stage 2 (Severity): Binary classification Mild (5-9 / C1) vs "Severe" (10-27 / C2+).
  Trained ONLY on training samples with PHQ-9 >= 5.
- Inference:
  X -> Model 1 -> If 0 => C0.
               -> If 1 => Model 2 -> 0 => C1, 1 => C2+.

Experiments:
- E0: Flat Random Forest on 3 classes (Baseline Control).
- E1: Hierarchical Random Forest (Stage 1 RF + Stage 2 RF, default thresholds 0.5/0.5).
- E2: Hierarchical Random Forest with Threshold Optimization (Inner CV on train fold).
- E3: Hierarchical with Alternative Stage 2 models (E3a: RF+LR, E3b: RF+XGB).
- Multi-cycle Extension: Evaluated on Branch A (Single-cycle 2013-2014, N=5,054),
  Branch B (Pooled 2011-2016, N=14,581), Branch C (Matched-N), Branch D (Temporal).

Strict Scientific Safeguards:
- No data leakage: preprocessing fit strictly within training folds/subsets.
- No ground truth routing at inference time.
- Exact fixed folds and frozen Random Forest configurations preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# DepSev-ML internal imports
import final_protocol as frozen
from config import ARTIFACTS_DIR, PROJECT_ROOT, TABLES_DIR
import multicycle_protocol as mp
from utils import load_json

EXPERIMENT_DIR = PROJECT_ROOT / "experiments/hierarchical_t3"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Canonical T3 definition
T3_CLASSES = [0, 1, 2]
T3_LABELS = ["C0 (Minimal)", "C1 (Mild)", "C2+ (Severe conv.)"]
T3_RANGES = ["0-4", "5-9", "10-27"]


@dataclass
class StageMetrics:
    """Detailed breakdown of Stage 1, Stage 2, and end-to-end routing behavior."""
    fold: int
    model_name: str
    n_samples: int
    n_symptomatic_true: int
    n_symptomatic_predicted: int
    
    # Stage 1 Screening metrics
    stage1_accuracy: float
    stage1_balanced_accuracy: float
    stage1_macro_f1: float
    stage1_symptomatic_recall: float
    stage1_symptomatic_precision: float
    stage1_c0_specificity: float
    stage1_c1_recall: float
    stage1_c2_recall: float
    stage1_c1_miss_rate: float
    stage1_c2_miss_rate: float
    
    # Error propagation counts
    case1_mild_missed_as_c0: int
    case1_severe_missed_as_c0: int
    case1_total_symptomatic_missed: int
    case2_c0_false_alarms: int
    case2_c0_routed_to_c1: int
    case2_c0_routed_to_c2: int
    case3_mild_routed_to_c2: int
    case3_severe_routed_to_c1: int
    
    # Stage 2 metrics on routed true symptomatics
    stage2_routed_accuracy: float
    stage2_oracle_balanced_accuracy: float


class HierarchicalT3Classifier(BaseEstimator, ClassifierMixin):
    """Two-stage sequential classifier for depression severity."""

    def __init__(
        self,
        stage1_estimator: Pipeline,
        stage2_estimator: Pipeline,
        threshold1: float = 0.5,
        threshold2: float = 0.5,
    ):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: pd.DataFrame, s: pd.Series) -> HierarchicalT3Classifier:
        """Fit Stage 1 on all data and Stage 2 only on symptomatic samples (PHQ-9 >= 5)."""
        # Stage 1 binary target: 0 if PHQ9 < 5, 1 if PHQ9 >= 5
        y1 = (s >= 5).astype(int)
        self.stage1_ = clone(self.stage1_estimator)
        self.stage1_.fit(X, y1)

        # Stage 2 binary target: 0 if 5 <= PHQ9 <= 9, 1 if PHQ9 >= 10
        symptomatic_mask = (s >= 5)
        X2 = X.loc[symptomatic_mask]
        s2 = s.loc[symptomatic_mask]
        y2 = (s2 >= 10).astype(int)

        self.stage2_ = clone(self.stage2_estimator)
        self.stage2_.fit(X2, y2)
        return self

    def predict_proba_stages(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Obtain Stage 1 and Stage 2 positive-class probabilities."""
        p1 = self.stage1_.predict_proba(X)[:, 1]
        p2 = self.stage2_.predict_proba(X)[:, 1]
        return p1, p2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict 3 classes {0: C0, 1: C1, 2: C2+} using sequential routing."""
        p1, p2 = self.predict_proba_stages(X)
        pred = np.zeros(len(X), dtype=int)
        routed = p1 >= self.threshold1
        if np.any(routed):
            pred[routed] = np.where(p2[routed] >= self.threshold2, 2, 1)
        return pred

    def predict_detailed(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        p1, p2 = self.predict_proba_stages(X)
        routed = p1 >= self.threshold1
        pred = np.zeros(len(X), dtype=int)
        if np.any(routed):
            pred[routed] = np.where(p2[routed] >= self.threshold2, 2, 1)
        return {
            "y_pred": pred,
            "stage1_prob": p1,
            "stage1_pred": routed.astype(int),
            "routed": routed,
            "stage2_prob": p2,
            "stage2_pred": (p2 >= self.threshold2).astype(int),
        }


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


def analyze_error_propagation(
    fold: int,
    model_name: str,
    y_true_t3: np.ndarray,
    scores: np.ndarray,
    detailed: Dict[str, np.ndarray],
    stage2_model: Pipeline,
    X_val: pd.DataFrame,
) -> StageMetrics:
    """Quantify error propagation through the two-stage hierarchy."""
    y_pred = detailed["y_pred"]
    p1 = detailed["stage1_prob"]
    routed = detailed["routed"]
    
    true_symptomatic = (scores >= 5).astype(int)
    pred_symptomatic = routed.astype(int)
    
    s1_acc = float(accuracy_score(true_symptomatic, pred_symptomatic))
    s1_bal = float(balanced_accuracy_score(true_symptomatic, pred_symptomatic))
    s1_f1 = float(f1_score(true_symptomatic, pred_symptomatic, average="macro", zero_division=0))
    s1_rec = float(recall_score(true_symptomatic, pred_symptomatic, zero_division=0))
    s1_prec = float(precision_score(true_symptomatic, pred_symptomatic, zero_division=0))
    
    # C0 specificity (True Negative Rate for symptomatic)
    c0_mask = (y_true_t3 == 0)
    c0_spec = float((pred_symptomatic[c0_mask] == 0).mean())
    
    # Recall and miss rates for C1 and C2+ at Stage 1
    c1_mask = (y_true_t3 == 1)
    c2_mask = (y_true_t3 == 2)
    c1_rec = float((pred_symptomatic[c1_mask] == 1).mean()) if c1_mask.sum() > 0 else 0.0
    c2_rec = float((pred_symptomatic[c2_mask] == 1).mean()) if c2_mask.sum() > 0 else 0.0
    c1_miss = 1.0 - c1_rec
    c2_miss = 1.0 - c2_rec
    
    # Error Case counts
    c1_missed = int(((y_true_t3 == 1) & (pred_symptomatic == 0)).sum())
    c2_missed = int(((y_true_t3 == 2) & (pred_symptomatic == 0)).sum())
    tot_missed = c1_missed + c2_missed
    
    c0_false_alarms = int(((y_true_t3 == 0) & (pred_symptomatic == 1)).sum())
    c0_to_c1 = int(((y_true_t3 == 0) & (y_pred == 1)).sum())
    c0_to_c2 = int(((y_true_t3 == 0) & (y_pred == 2)).sum())
    
    c1_to_c2 = int(((y_true_t3 == 1) & (y_pred == 2)).sum())
    c2_to_c1 = int(((y_true_t3 == 2) & (y_pred == 1)).sum())
    
    # Stage 2 metrics on routed symptomatic
    routed_symptomatic_mask = (true_symptomatic == 1) & (pred_symptomatic == 1)
    if routed_symptomatic_mask.sum() > 0:
        y2_true_routed = (y_true_t3[routed_symptomatic_mask] == 2).astype(int)
        y2_pred_routed = (y_pred[routed_symptomatic_mask] == 2).astype(int)
        s2_routed_acc = float(accuracy_score(y2_true_routed, y2_pred_routed))
    else:
        s2_routed_acc = 0.0
        
    # Oracle Stage 2 performance (if all true symptomatics were passed to Stage 2)
    if true_symptomatic.sum() > 0:
        p2_oracle = stage2_model.predict_proba(X_val.loc[c1_mask | c2_mask])[:, 1] >= 0.5
        y2_oracle_true = (y_true_t3[c1_mask | c2_mask] == 2).astype(int)
        s2_oracle_bal = float(balanced_accuracy_score(y2_oracle_true, p2_oracle.astype(int)))
    else:
        s2_oracle_bal = 0.0
        
    return StageMetrics(
        fold=fold,
        model_name=model_name,
        n_samples=len(y_true_t3),
        n_symptomatic_true=int(true_symptomatic.sum()),
        n_symptomatic_predicted=int(pred_symptomatic.sum()),
        stage1_accuracy=s1_acc,
        stage1_balanced_accuracy=s1_bal,
        stage1_macro_f1=s1_f1,
        stage1_symptomatic_recall=s1_rec,
        stage1_symptomatic_precision=s1_prec,
        stage1_c0_specificity=c0_spec,
        stage1_c1_recall=c1_rec,
        stage1_c2_recall=c2_rec,
        stage1_c1_miss_rate=c1_miss,
        stage1_c2_miss_rate=c2_miss,
        case1_mild_missed_as_c0=c1_missed,
        case1_severe_missed_as_c0=c2_missed,
        case1_total_symptomatic_missed=tot_missed,
        case2_c0_false_alarms=c0_false_alarms,
        case2_c0_routed_to_c1=c0_to_c1,
        case2_c0_routed_to_c2=c0_to_c2,
        case3_mild_routed_to_c2=c1_to_c2,
        case3_severe_routed_to_c1=c2_to_c1,
        stage2_routed_accuracy=s2_routed_acc,
        stage2_oracle_balanced_accuracy=s2_oracle_bal,
    )


def optimize_thresholds_inner_cv(
    X_train: pd.DataFrame,
    s_train: pd.Series,
    spec: Dict[str, Any],
    inventory: pd.DataFrame,
    n_splits: int = 3,
) -> Tuple[float, float, pd.DataFrame]:
    """Inner CV grid search on training partition to maximize end-to-end Macro-F1."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_t3_train = np.searchsorted([5, 10], s_train, side="right")
    
    t1_grid = np.arange(0.25, 0.70, 0.05)
    t2_grid = np.arange(0.25, 0.70, 0.05)
    
    # Store out-of-fold probabilities for training partition
    inner_p1 = np.zeros(len(X_train), dtype=float)
    inner_p2 = np.zeros(len(X_train), dtype=float)
    
    for in_tr_idx, in_va_idx in splitter.split(X_train, y_t3_train):
        X_in_tr, s_in_tr = X_train.iloc[in_tr_idx], s_train.iloc[in_tr_idx]
        X_in_va = X_train.iloc[in_va_idx]
        
        # Inner Stage 1
        pipe1 = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", RandomForestClassifier(**spec["hyperparameters"])),
        ])
        pipe1.fit(X_in_tr, (s_in_tr >= 5).astype(int))
        inner_p1[in_va_idx] = pipe1.predict_proba(X_in_va)[:, 1]
        
        # Inner Stage 2
        symptomatic_in_tr = (s_in_tr >= 5)
        pipe2 = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", RandomForestClassifier(**spec["hyperparameters"])),
        ])
        pipe2.fit(X_in_tr[symptomatic_in_tr], (s_in_tr[symptomatic_in_tr] >= 10).astype(int))
        inner_p2[in_va_idx] = pipe2.predict_proba(X_in_va)[:, 1]
        
    grid_records = []
    best_macro_f1 = -1.0
    best_t1, best_t2 = 0.5, 0.5
    
    for t1 in t1_grid:
        for t2 in t2_grid:
            pred = np.zeros(len(X_train), dtype=int)
            routed = inner_p1 >= t1
            pred[routed] = np.where(inner_p2[routed] >= t2, 2, 1)
            score = float(f1_score(y_t3_train, pred, average="macro", zero_division=0))
            grid_records.append({"threshold1": round(t1, 2), "threshold2": round(t2, 2), "inner_macro_f1": score})
            if score > best_macro_f1:
                best_macro_f1 = score
                best_t1, best_t2 = float(t1), float(t2)
                
    return best_t1, best_t2, pd.DataFrame(grid_records)


def run_legacy_cohort_experiments() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute E0, E1, E2, E3a, E3b on Legacy Cohort Development (N=4,045)."""
    gran = sys.modules.get("09_granularity_experiment")
    if gran is None:
        import importlib
        gran = importlib.import_module("09_granularity_experiment")
        
    X, y_c0_c4, inventory, scores, folds, assignments, spec = gran.load_development()
    y_t3 = gran.map_target(y_c0_c4, "T3").to_numpy(dtype=int)
    
    fold_metric_rows = []
    stage_metric_rows = []
    oof_predictions = {
        "SEQN": X.index.to_numpy(),
        "fold": assignments,
        "PHQ9_total": scores.to_numpy(),
        "y_true": y_t3,
    }
    
    models = ["E0_Flat_RF", "E1_Hierarchical_RF", "E2_Hierarchical_RF_OptThresh", "E3a_Hierarchical_RF_LR", "E3b_Hierarchical_RF_XGB"]
    for m in models:
        oof_predictions[f"{m}_pred"] = np.zeros(len(X), dtype=int)
        
    threshold_records = []
    
    print("\n>>> Running Legacy Cohort (NHANES 2013-2014, N=4,045) across 5 fixed folds...")
    
    for fold_number, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"  Processing Fold {fold_number}/5...", flush=True)
        X_train, y_train, s_train = X.iloc[train_idx], y_t3[train_idx], scores.iloc[train_idx]
        X_val, y_val, s_val = X.iloc[val_idx], y_t3[val_idx], scores.iloc[val_idx]
        
        # 1. E0: Flat RF
        pipe_e0 = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", RandomForestClassifier(**spec["hyperparameters"])),
        ])
        pipe_e0.fit(X_train, y_train)
        pred_e0 = pipe_e0.predict(X_val)
        oof_predictions["E0_Flat_RF_pred"][val_idx] = pred_e0
        m_e0 = calculate_comprehensive_metrics(y_val, pred_e0)
        m_e0.update({"experiment": "Legacy_Dev", "model": "E0_Flat_RF", "fold": fold_number})
        fold_metric_rows.append(m_e0)
        
        # 2. E1: Hierarchical RF (Default 0.5/0.5)
        pipe_s1 = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", RandomForestClassifier(**spec["hyperparameters"])),
        ])
        pipe_s2_rf = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", RandomForestClassifier(**spec["hyperparameters"])),
        ])
        clf_e1 = HierarchicalT3Classifier(pipe_s1, pipe_s2_rf, threshold1=0.5, threshold2=0.5)
        clf_e1.fit(X_train, s_train)
        det_e1 = clf_e1.predict_detailed(X_val)
        pred_e1 = det_e1["y_pred"]
        oof_predictions["E1_Hierarchical_RF_pred"][val_idx] = pred_e1
        m_e1 = calculate_comprehensive_metrics(y_val, pred_e1)
        m_e1.update({"experiment": "Legacy_Dev", "model": "E1_Hierarchical_RF", "fold": fold_number})
        fold_metric_rows.append(m_e1)
        
        # Stage & Error propagation metrics for E1
        st_e1 = analyze_error_propagation(fold_number, "E1_Hierarchical_RF", y_val, s_val.to_numpy(), det_e1, clf_e1.stage2_, X_val)
        stage_metric_rows.append(asdict(st_e1))
        
        # 3. E2: Hierarchical RF with Threshold Optimization (Inner CV)
        t1_opt, t2_opt, grid_df = optimize_thresholds_inner_cv(X_train, s_train, spec, inventory, n_splits=3)
        threshold_records.append({
            "fold": fold_number,
            "best_threshold1": t1_opt,
            "best_threshold2": t2_opt,
        })
        clf_e2 = HierarchicalT3Classifier(pipe_s1, pipe_s2_rf, threshold1=t1_opt, threshold2=t2_opt)
        clf_e2.fit(X_train, s_train)
        det_e2 = clf_e2.predict_detailed(X_val)
        pred_e2 = det_e2["y_pred"]
        oof_predictions["E2_Hierarchical_RF_OptThresh_pred"][val_idx] = pred_e2
        m_e2 = calculate_comprehensive_metrics(y_val, pred_e2)
        m_e2.update({"experiment": "Legacy_Dev", "model": "E2_Hierarchical_RF_OptThresh", "fold": fold_number, "threshold1": t1_opt, "threshold2": t2_opt})
        fold_metric_rows.append(m_e2)
        
        # 4. E3a: Stage 1 RF + Stage 2 Logistic Regression
        pipe_s2_lr = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ])
        clf_e3a = HierarchicalT3Classifier(pipe_s1, pipe_s2_lr, threshold1=0.5, threshold2=0.5)
        clf_e3a.fit(X_train, s_train)
        pred_e3a = clf_e3a.predict(X_val)
        oof_predictions["E3a_Hierarchical_RF_LR_pred"][val_idx] = pred_e3a
        m_e3a = calculate_comprehensive_metrics(y_val, pred_e3a)
        m_e3a.update({"experiment": "Legacy_Dev", "model": "E3a_Hierarchical_RF_LR", "fold": fold_number})
        fold_metric_rows.append(m_e3a)
        
        # 5. E3b: Stage 1 RF + Stage 2 XGBoost
        symptomatic_tr = (s_train >= 5)
        pos_weight = float((s_train[symptomatic_tr] < 10).sum()) / max(float((s_train[symptomatic_tr] >= 10).sum()), 1.0)
        pipe_s2_xgb = Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(spec["features"], inventory)),
            ("model", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42, n_jobs=1, eval_metric="logloss")),
        ])
        clf_e3b = HierarchicalT3Classifier(pipe_s1, pipe_s2_xgb, threshold1=0.5, threshold2=0.5)
        clf_e3b.fit(X_train, s_train)
        pred_e3b = clf_e3b.predict(X_val)
        oof_predictions["E3b_Hierarchical_RF_XGB_pred"][val_idx] = pred_e3b
        m_e3b = calculate_comprehensive_metrics(y_val, pred_e3b)
        m_e3b.update({"experiment": "Legacy_Dev", "model": "E3b_Hierarchical_RF_XGB", "fold": fold_number})
        fold_metric_rows.append(m_e3b)
        
    df_folds = pd.DataFrame(fold_metric_rows)
    df_stages = pd.DataFrame(stage_metric_rows)
    df_oof = pd.DataFrame(oof_predictions)
    df_thresh = pd.DataFrame(threshold_records)
    
    return df_folds, df_stages, df_oof, df_thresh


def run_multicycle_cohort_experiments(
    frames: Dict[str, pd.DataFrame],
    policy: Dict[str, Any],
    old_spec: Dict[str, Any],
) -> pd.DataFrame:
    """Execute E0 vs E1 on Official Multi-cycle Cohorts (Branch A, B, C, D)."""
    rows = []
    
    # 1. Branch A: Official 2013-2014 (N=5,054)
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
        
        # Flat
        p_flat = mp.make_pipeline(policy, old_spec)
        p_flat.fit(X_tr, y_tr)
        pred_flat = p_flat.predict(X_va)
        m_flat = calculate_comprehensive_metrics(y_va, pred_flat)
        m_flat.update({"branch": "A_SingleCycle", "model": "E0_Flat_RF", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m_flat)
        
        # Hierarchical
        p1 = mp.make_pipeline(policy, old_spec)
        p2 = mp.make_pipeline(policy, old_spec)
        clf_h = HierarchicalT3Classifier(p1, p2, 0.5, 0.5)
        clf_h.fit(X_tr, s_tr)
        pred_h = clf_h.predict(X_va)
        m_h = calculate_comprehensive_metrics(y_va, pred_h)
        m_h.update({"branch": "A_SingleCycle", "model": "E1_Hierarchical_RF", "fold": fold, "n_samples": len(frame_a)})
        rows.append(m_h)
        
    # 2. Branch B: Official Pooled 2011-2016 (N=14,581)
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
        
        p_flat = mp.make_pipeline(policy, old_spec)
        p_flat.fit(X_tr, y_tr)
        pred_flat = p_flat.predict(X_va)
        m_flat = calculate_comprehensive_metrics(y_va, pred_flat)
        m_flat.update({"branch": "B_Pooled", "model": "E0_Flat_RF", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m_flat)
        
        p1 = mp.make_pipeline(policy, old_spec)
        p2 = mp.make_pipeline(policy, old_spec)
        clf_h = HierarchicalT3Classifier(p1, p2, 0.5, 0.5)
        clf_h.fit(X_tr, s_tr)
        pred_h = clf_h.predict(X_va)
        m_h = calculate_comprehensive_metrics(y_va, pred_h)
        m_h.update({"branch": "B_Pooled", "model": "E1_Hierarchical_RF", "fold": fold, "n_samples": len(frame_b)})
        rows.append(m_h)
        
    # 3. Branch D: Temporal Generalization (Train 2011-2014, Test 2015-2016)
    print(">>> Running Multi-cycle Branch D (Temporal Generalization)...")
    train_d = pd.concat([frames["2011_2012"], frames["2013_2014"]], ignore_index=True)
    test_d = frames["2015_2016"]
    X_tr_d = mp.predictor_frame(train_d, policy)
    s_tr_d = train_d.PHQ9_total
    y_tr_d = mp.target_values(s_tr_d, "T3")
    X_te_d = mp.predictor_frame(test_d, policy)
    s_te_d = test_d.PHQ9_total
    y_te_d = mp.target_values(s_te_d, "T3")
    
    p_flat = mp.make_pipeline(policy, old_spec)
    p_flat.fit(X_tr_d, y_tr_d)
    pred_flat_d = p_flat.predict(X_te_d)
    m_flat_d = calculate_comprehensive_metrics(y_te_d, pred_flat_d)
    m_flat_d.update({"branch": "D_Temporal", "model": "E0_Flat_RF", "fold": 1, "n_samples": len(test_d)})
    rows.append(m_flat_d)
    
    p1 = mp.make_pipeline(policy, old_spec)
    p2 = mp.make_pipeline(policy, old_spec)
    clf_h = HierarchicalT3Classifier(p1, p2, 0.5, 0.5)
    clf_h.fit(X_tr_d, s_tr_d)
    pred_h_d = clf_h.predict(X_te_d)
    m_h_d = calculate_comprehensive_metrics(y_te_d, pred_h_d)
    m_h_d.update({"branch": "D_Temporal", "model": "E1_Hierarchical_RF", "fold": 1, "n_samples": len(test_d)})
    rows.append(m_h_d)
    
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path, class_labels: Sequence[str] = T3_LABELS):
    """Save normalized and annotated confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    normalized = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    n = len(class_labels)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm[i, j]}\n{normalized[i, j]:.1%}",
                ha="center", va="center",
                color="white" if normalized[i, j] > 0.55 else "#1e293b",
                fontweight="bold" if i == j else "normal",
            )
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_labels, yticklabels=class_labels,
        xlabel="Predicted Severity Class", ylabel="True Severity Class",
        title=title,
    )
    fig.colorbar(im, ax=ax, label="Fraction of True Class", shrink=0.85)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_publication_figures(
    df_legacy_summary: pd.DataFrame,
    df_stages: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    df_multicycle_summary: pd.DataFrame,
):
    """Generate high-resolution charts for report and analysis."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Macro-F1 & Balanced Accuracy Comparison (Legacy)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    models = df_legacy_summary["model"].tolist()
    labels = [m.replace("_", " ") for m in models]
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, df_legacy_summary["macro_f1_mean"], width, yerr=df_legacy_summary["macro_f1_std"],
                    capsize=4, label="Macro-F1", color="#2563EB")
    rects2 = ax.bar(x + width/2, df_legacy_summary["balanced_accuracy_mean"], width, yerr=df_legacy_summary["balanced_accuracy_std"],
                    capsize=4, label="Balanced Accuracy", color="#0D9488")
    
    ax.set_ylabel("Score (Mean ± Fold SD)")
    ax.set_title("Performance Across Architectures on Legacy Development Cohort (N=4,045)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 0.65)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(rects1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(rects2, fmt="%.3f", padding=3, fontsize=8)
    fig.savefig(FIGURES_DIR / "legacy_model_comparison.png", dpi=200)
    plt.close(fig)
    
    # 2. Per-class metrics comparison (E0 vs E1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained", sharey=True)
    metrics_names = ["precision", "recall", "f1"]
    titles = ["Precision per Class", "Recall per Class", "F1-Score per Class"]
    
    e0_row = df_legacy_summary[df_legacy_summary["model"] == "E0_Flat_RF"].iloc[0]
    e1_row = df_legacy_summary[df_legacy_summary["model"] == "E1_Hierarchical_RF"].iloc[0]
    classes = ["C0", "C1", "C2+"]
    x = np.arange(len(classes))
    width = 0.35
    
    for ax, met, title in zip(axes, metrics_names, titles):
        e0_vals = [e0_row[f"c{i}_{met}_mean"] for i in range(3)]
        e1_vals = [e1_row[f"c{i}_{met}_mean"] for i in range(3)]
        e0_errs = [e0_row[f"c{i}_{met}_std"] for i in range(3)]
        e1_errs = [e1_row[f"c{i}_{met}_std"] for i in range(3)]
        
        b1 = ax.bar(x - width/2, e0_vals, width, yerr=e0_errs, capsize=4, label="E0 Flat RF", color="#64748B")
        b2 = ax.bar(x + width/2, e1_vals, width, yerr=e1_errs, capsize=4, label="E1 Hierarchical RF", color="#2563EB")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
        ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Score")
            ax.legend()
            
    fig.suptitle("Per-Class Trade-offs: E0 Flat RF vs E1 Hierarchical RF")
    fig.savefig(FIGURES_DIR / "per_class_tradeoffs.png", dpi=200)
    plt.close(fig)
    
    # 3. Error Breakdown Bar Chart
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    error_types = [
        "Case 1: Mild (C1) Missed as C0\n(Stage 1 False Negatives)",
        "Case 1: Severe (C2+) Missed as C0\n(Stage 1 False Negatives)",
        "Case 2: Minimal (C0) False Alarms\n(Stage 1 False Positives)",
        "Case 3: Mild (C1) Mistaken as C2+\n(Stage 2 Confusion)",
        "Case 3: Severe (C2+) Mistaken as C1\n(Stage 2 Confusion)",
    ]
    mean_errors = [
        df_stages["case1_mild_missed_as_c0"].sum(),
        df_stages["case1_severe_missed_as_c0"].sum(),
        df_stages["case2_c0_false_alarms"].sum(),
        df_stages["case3_mild_routed_to_c2"].sum(),
        df_stages["case3_severe_routed_to_c1"].sum(),
    ]
    colors = ["#F59E0B", "#DC2626", "#7C3AED", "#0D9488", "#2563EB"]
    bars = ax.barh(error_types, mean_errors, color=colors)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=9, fontweight="bold")
    ax.set_xlabel("Total Out-of-Fold Error Instances (N=4,045)")
    ax.set_title("Hierarchical Error Propagation: Where Do Predictions Fail?")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.savefig(FIGURES_DIR / "error_propagation_breakdown.png", dpi=200)
    plt.close(fig)
    
    # 4. Multi-cycle Branch Comparison (E0 vs E1)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    branches = ["A_SingleCycle", "B_Pooled", "D_Temporal"]
    x = np.arange(len(branches))
    width = 0.35
    
    flat_f1s = [df_multicycle_summary[(df_multicycle_summary["branch"] == b) & (df_multicycle_summary["model"] == "E0_Flat_RF")]["macro_f1_mean"].iloc[0] for b in branches]
    hier_f1s = [df_multicycle_summary[(df_multicycle_summary["branch"] == b) & (df_multicycle_summary["model"] == "E1_Hierarchical_RF")]["macro_f1_mean"].iloc[0] for b in branches]
    flat_errs = [df_multicycle_summary[(df_multicycle_summary["branch"] == b) & (df_multicycle_summary["model"] == "E0_Flat_RF")]["macro_f1_std"].iloc[0] for b in branches]
    hier_errs = [df_multicycle_summary[(df_multicycle_summary["branch"] == b) & (df_multicycle_summary["model"] == "E1_Hierarchical_RF")]["macro_f1_std"].iloc[0] for b in branches]
    
    b1 = ax.bar(x - width/2, flat_f1s, width, yerr=flat_errs, capsize=4, label="Flat RF (E0)", color="#64748B")
    b2 = ax.bar(x + width/2, hier_f1s, width, yerr=hier_errs, capsize=4, label="Hierarchical RF (E1)", color="#0D9488")
    
    ax.set_xticks(x)
    ax.set_xticklabels(["Branch A (Single 2013-14)\nN=5,054", "Branch B (Pooled 2011-16)\nN=14,581", "Branch D (Temporal)\nTrain: 9,693, Test: 4,888"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Multi-cycle Generalization: Flat RF vs Hierarchical RF (31 Common Features)")
    ax.set_ylim(0, 0.6)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=9)
    fig.savefig(FIGURES_DIR / "multicycle_e0_vs_e1.png", dpi=200)
    plt.close(fig)


def summarize_fold_metrics(df_folds: pd.DataFrame, group_cols: List[str] = ["model"]) -> pd.DataFrame:
    """Aggregate fold metrics to mean and sample standard deviation (ddof=1)."""
    metric_cols = [c for c in df_folds.columns if c not in group_cols + ["fold", "experiment", "branch", "n_samples"]]
    records = []
    for keys, group in df_folds.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        for col in metric_cols:
            vals = group[col].dropna()
            rec[f"{col}_mean"] = float(vals.mean())
            rec[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        records.append(rec)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-multicycle", action="store_true", help="Run only on Legacy Cohort")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now(timezone.utc)
    print(f"============================================================")
    print(f"DepSev-ML: Hierarchical Classification Experiment Runner")
    print(f"Timestamp UTC: {start_time.isoformat()}")
    print(f"============================================================")
    
    # 1. Run Legacy Cohort Experiments (E0, E1, E2, E3a, E3b)
    df_legacy_folds, df_stages, df_oof, df_thresh = run_legacy_cohort_experiments()
    df_legacy_summary = summarize_fold_metrics(df_legacy_folds, group_cols=["model"])
    
    # Save Legacy Tables
    df_legacy_folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    df_legacy_summary.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    df_stages.to_csv(OUTPUT_DIR / "stage_metrics.csv", index=False)
    df_oof.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    df_thresh.to_csv(OUTPUT_DIR / "threshold_optimization.csv", index=False)
    
    # Save Error Propagation Summary
    error_summary = {
        "total_development_n": len(df_oof),
        "true_c0": int((df_oof["y_true"] == 0).sum()),
        "true_c1": int((df_oof["y_true"] == 1).sum()),
        "true_c2": int((df_oof["y_true"] == 2).sum()),
        "case1_mild_missed_as_c0": int(df_stages["case1_mild_missed_as_c0"].sum()),
        "case1_mild_missed_rate": float(df_stages["case1_mild_missed_as_c0"].sum()) / int((df_oof["y_true"] == 1).sum()),
        "case1_severe_missed_as_c0": int(df_stages["case1_severe_missed_as_c0"].sum()),
        "case1_severe_missed_rate": float(df_stages["case1_severe_missed_as_c0"].sum()) / int((df_oof["y_true"] == 2).sum()),
        "case2_c0_false_alarms": int(df_stages["case2_c0_false_alarms"].sum()),
        "case2_c0_false_alarm_rate": float(df_stages["case2_c0_false_alarms"].sum()) / int((df_oof["y_true"] == 0).sum()),
        "case2_c0_routed_to_c1": int(df_stages["case2_c0_routed_to_c1"].sum()),
        "case2_c0_routed_to_c2": int(df_stages["case2_c0_routed_to_c2"].sum()),
        "case3_mild_routed_to_c2": int(df_stages["case3_mild_routed_to_c2"].sum()),
        "case3_severe_routed_to_c1": int(df_stages["case3_severe_routed_to_c1"].sum()),
    }
    pd.DataFrame([error_summary]).to_csv(OUTPUT_DIR / "error_propagation_analysis.csv", index=False)
    
    # Generate Confusion Matrices for Legacy Models
    for col, name in [
        ("E0_Flat_RF_pred", "E0_Flat_RF"),
        ("E1_Hierarchical_RF_pred", "E1_Hierarchical_RF"),
        ("E2_Hierarchical_RF_OptThresh_pred", "E2_Hierarchical_RF_OptThresh"),
        ("E3a_Hierarchical_RF_LR_pred", "E3a_Hierarchical_RF_LR"),
        ("E3b_Hierarchical_RF_XGB_pred", "E3b_Hierarchical_RF_XGB"),
    ]:
        cm = confusion_matrix(df_oof["y_true"], df_oof[col], labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=T3_LABELS, columns=[f"Pred_{l}" for l in T3_LABELS])
        cm_df.to_csv(CONFUSION_DIR / f"{name}_confusion_matrix.csv")
        plot_confusion_matrix(cm, f"{name}: Out-of-Fold Confusion (N=4,045)", CONFUSION_DIR / f"{name}_confusion.png")
        
    print("\nLegacy development results complete:")
    for _, row in df_legacy_summary.iterrows():
        print(f"  {row['model']:30s} | Macro-F1: {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | BalAcc: {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | Acc: {row['accuracy_mean']:.4f}")
        
    # 2. Run Multi-cycle Cohort Experiments
    df_multicycle_summary = pd.DataFrame()
    if not args.skip_multicycle:
        frames, policy, old_spec = mp.load_inputs()
        df_mc_folds = run_multicycle_cohort_experiments(frames, policy, old_spec)
        df_multicycle_summary = summarize_fold_metrics(df_mc_folds, group_cols=["branch", "model"])
        df_mc_folds.to_csv(OUTPUT_DIR / "multicycle_fold_metrics.csv", index=False)
        df_multicycle_summary.to_csv(OUTPUT_DIR / "multicycle_comparison.csv", index=False)
        
        print("\nMulti-cycle results complete:")
        for _, row in df_multicycle_summary.iterrows():
            print(f"  Branch: {row['branch']:15s} | Model: {row['model']:22s} | Macro-F1: {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | BalAcc: {row['balanced_accuracy_mean']:.4f}")
            
    # 3. Generate Visualizations
    generate_publication_figures(df_legacy_summary, df_stages, df_oof, df_multicycle_summary)
    print(f"\nGenerated all publication figures in: {FIGURES_DIR}")
    
    # 4. Save Run Manifest & Metadata
    end_time = datetime.now(timezone.utc)
    manifest = {
        "experiment": "hierarchical_t3_evaluation",
        "started_utc": start_time.isoformat(),
        "completed_utc": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "python_version": sys.version,
        "legacy_development_n": len(df_oof),
        "models_evaluated": [
            "E0_Flat_RF",
            "E1_Hierarchical_RF",
            "E2_Hierarchical_RF_OptThresh",
            "E3a_Hierarchical_RF_LR",
            "E3b_Hierarchical_RF_XGB",
        ],
        "multicycle_evaluated": not args.skip_multicycle,
        "files_generated": [str(p.relative_to(PROJECT_ROOT)) for p in OUTPUT_DIR.rglob("*") if p.is_file()],
    }
    with (EXPERIMENT_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nExecution finished in {(end_time - start_time).total_seconds():.1f}s. Manifest written to {EXPERIMENT_DIR / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
