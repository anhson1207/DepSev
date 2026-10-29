"""Frozen F4 learners, soft voting and cross-fitted stacking on development only.

Run with --audit-only to freeze configurations and verify existing folds without
fitting. The default run evaluates five outer folds, never fits a final model on
all development rows, and refuses to overwrite completed experiment results.
"""

import argparse
import ast
from datetime import datetime, timezone
from importlib import import_module, metadata
import inspect
from pathlib import Path
import tempfile
import textwrap
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ablation_features import prepare_features
from config import (ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR, PROJECT_ROOT,
                    PROCESSED_DATA_DIR, RAW_DATA_DIR, PROTOCOL_VERSION, experiment_path)
from protocol import FINAL_SPLIT, ORIGINAL_SPLIT, load_final_split, sha256_file
from tuning_estimators import FoldBalancedXGBClassifier, id_hash
from utils import load_json, save_json


ablation = import_module("05_feature_ablation")
CLASSES = np.arange(5)
BASES = ["LR", "RF", "XGB"]
VOTING = {"SoftVoting": [1, 1, 1], "SoftVoting_RF2": [1, 2, 1]}
MODELS = [*BASES, *VOTING, "Stacking"]
PROB_COLUMNS = [f"prob_C{c}" for c in CLASSES]
META_COLUMNS = [f"{model}_prob_C{c}" for model in BASES for c in CLASSES]
CONFIG_PATH = ARTIFACTS_DIR / "ensemble_model_configs.json"
MANIFEST_PATH = ARTIFACTS_DIR / "ensemble_protocol.json"
FOLDS_PATH = ARTIFACTS_DIR / "ensemble_cv_assignments.json"
WEIGHT_DIR = ARTIFACTS_DIR / "ensemble_weight_audit"
RESULT_PATH = TABLES_DIR / "ensemble_results.csv"


def verify_xgb_fixed(config):
    """Check the wrapper's actual constructor literals without fitting a model."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(FoldBalancedXGBClassifier.fit)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "XGBClassifier"]
    if len(calls) != 1:
        raise AssertionError("Review the XGB wrapper constructor before proceeding")
    fixed = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords
             if kw.arg in config["XGB_fixed"]}
    if fixed != config["XGB_fixed"]:
        raise AssertionError("Wrapper hard-coded XGB parameters differ from the frozen config")
    return fixed


def frozen_configuration(features):
    """Use only user-approved parameters and existing development evidence."""
    tuned = load_json(ARTIFACTS_DIR / "tuned_model_configs.json")
    if (tuned["selected_model"] != "Random Forest" or
            tuned["selected_feature_set"] != "F4" or tuned["features"] != features):
        raise ValueError("RF family/feature definition differs from the approved artifact")
    rf = tuned["estimator_params"]
    expected_rf = dict(n_estimators=150, max_depth=8, min_samples_split=10,
                       min_samples_leaf=8, max_features=0.5,
                       class_weight="balanced_subsample", random_state=42)
    if any(rf[key] != value for key, value in expected_rf.items()):
        raise ValueError("RF artifact differs from the approved tuned parameters")
    best = pd.read_csv(TABLES_DIR / "best_params_by_fold.csv")
    import json
    best_c = [json.loads(p)["model__C"] for p in best.loc[
        best.model.eq("Logistic Regression") & best.feature_set.eq("F4"), "best_params"]]
    if len(best_c) != 5 or np.median(best_c) != 0.3 or pd.Series(best_c).mode().tolist() != [0.3]:
        raise ValueError("LR consensus provenance no longer matches")
    baseline = pd.read_csv(experiment_path(TABLES_DIR, "feature_ablation_results.csv"))
    nested = pd.read_csv(TABLES_DIR / "nested_cv_summary.csv")
    old_xgb = baseline[baseline.model.eq("XGBoost") & baseline.feature_set.eq("F4")].iloc[0]
    tuned_xgb = nested[nested.model.eq("XGBoost") & nested.feature_set.eq("F4")].iloc[0]
    if old_xgb.macro_f1_mean <= tuned_xgb.macro_f1_mean:
        raise ValueError("Existing XGB development evidence changed; do not silently reselect")
    config = {
        "protocol": PROTOCOL_VERSION, "feature_set": "F4", "features": features,
        "base_order": BASES, "class_order": CLASSES.tolist(),
        "LR": dict(C=0.3, solver="newton-cg", l1_ratio=0.0,
                   class_weight="balanced", max_iter=3000, random_state=42),
        "LR_provenance": {"description": "consensus LR configuration for ensemble",
                          "globally_optimal_claim": False, "best_C_by_outer_fold": best_c,
                          "C_mode": 0.3, "C_median": 0.3, "user_confirmed": True,
                          "penalty": "l2", "API_note": "l1_ratio=0.0 is L2 in sklearn 1.9"},
        "RF": rf,
        "RF_provenance": "exact estimator_params in artifacts/tuned_model_configs.json",
        "XGB": dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                    min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                    gamma=0, reg_alpha=0, reg_lambda=1, random_state=42, n_jobs=1),
        "XGB_fixed": dict(objective="multi:softprob", num_class=5, eval_metric="mlogloss"),
        "XGB_provenance": {
            "choice": "F4 baseline; selected before ensemble using development only",
            "baseline_macro_f1_mean": float(old_xgb.macro_f1_mean),
            "baseline_macro_f1_std": float(old_xgb.macro_f1_std),
            "tuned_macro_f1_mean": float(tuned_xgb.macro_f1_mean),
            "tuned_macro_f1_std": float(tuned_xgb.macro_f1_std)},
        "voting_weights_LR_RF_XGB": VOTING,
        "meta_LR": dict(C=1.0, solver="lbfgs", l1_ratio=0.0,
                        class_weight="balanced", max_iter=3000, random_state=42),
        "meta_features": META_COLUMNS, "passthrough": False,
        "meta_preprocessing": "none; 15 probabilities on their original 0..1 scale",
        "meta_config_selected_before_results": True,
        "preprocessor": "unchanged ablation preprocessor, fresh fit in EVERY training partition",
        "XGB_weighting": "balanced weights recomputed INSIDE fit from that partition's y",
        "probability_calibration": "none; class-balanced probabilities are not population risks",
        "selection_rule": {"reference": "RF", "min_macro_f1_gain": 0.01,
                           "min_folds_improved": 4, "std_must_not_exceed_RF": True,
                           "interpretation": "practical conservative rule, not a significance test"},
        "tuning": False, "heldout_used": False, "full_development_refit": False,
        "historically_untouched": False,
    }
    ablation.lock_json(config, CONFIG_PATH)
    return config


def lock_partitions(y, outer_folds):
    """Verify all outer/inner assignments against the prior nested-CV artifact."""
    previous = load_json(ARTIFACTS_DIR / "nested_cv_assignments.json")
    records, inner_folds = [], {}
    for number, (train, valid) in enumerate(outer_folds, 1):
        inner_folds[number] = list(StratifiedKFold(3, shuffle=True, random_state=42).split(
            np.zeros(len(train)), y.iloc[train]))
        assignment = np.full(len(train), -1, dtype=int)
        for inner_number, (inner_train, inner_valid) in enumerate(inner_folds[number], 1):
            if set(train[inner_train]) & set(valid) or set(inner_train) & set(inner_valid):
                raise AssertionError("Inner/outer training leakage")
            assignment[inner_valid] = inner_number
        row = {"outer_fold": number, "outer_train_seqn": y.iloc[train].index.tolist(),
               "outer_validation_seqn": y.iloc[valid].index.tolist(),
               "inner_validation_fold": assignment.tolist()}
        if row != previous["outer_partitions"][number - 1]:
            raise AssertionError("Changed CV membership")
        records.append(row)
    payload = {"protocol": PROTOCOL_VERSION, "outer_n_splits": 5, "inner_n_splits": 3,
               "shuffle": True, "random_state": 42, "outer_partitions": records}
    ablation.lock_json(payload, FOLDS_PATH)
    return inner_folds


def lock_manifest(development, config):
    """Snapshot source/input/result fingerprints before any ensemble model fit."""
    protected = [FINAL_SPLIT, ORIGINAL_SPLIT, ARTIFACTS_DIR / "split_indices.json",
                 ARTIFACTS_DIR / "tuned_model_configs.json", CONFIG_PATH, FOLDS_PATH,
                 PROCESSED_DATA_DIR / "cohort.csv", RAW_DATA_DIR / "questionnaire.csv",
                 ARTIFACTS_DIR / "nested_cv_assignments.json",
                 ARTIFACTS_DIR / "nested_tuning_protocol.json",
                 experiment_path(ARTIFACTS_DIR, "cv_folds.json")]
    protected += [PROJECT_ROOT / "src" / name for name in [
        "06_ensemble.py", "05_feature_ablation.py", "ablation_features.py",
        "config.py", "protocol.py", "tuning_estimators.py", "utils.py"]]
    # Preserve all pre-existing experiment tables, not just headline scores.
    protected += sorted(p for p in TABLES_DIR.glob("*.csv") if not p.name.startswith("ensemble_"))
    payload = {
        "protocol": PROTOCOL_VERSION, "development_n": len(development),
        "development_class_counts": development.severity.value_counts().sort_index().to_dict(),
        "features": config["features"], "protected_sha256": {
            str(p): sha256_file(p) for p in protected},
        "versions": {p: metadata.version(p) for p in ["scikit-learn", "xgboost", "numpy", "pandas"]},
        "heldout_used": False, "historically_untouched": False,
        "full_development_refit": False,
        "evaluation_scope": "development comparison conditional on previously selected frozen configurations",
        "selection_bias_warning": "Earlier development tuning/config selection used these participants; "
                                  "cross-fitting prevents fit-time leakage, not all prior selection bias.",
        "std_ddof": 1, "per_class_aggregation": "pooled outer OOF, plus fold mean/SD",
    }
    ablation.lock_json(payload, MANIFEST_PATH)
    return payload


def verify_protected(manifest):
    for filename, expected in manifest["protected_sha256"].items():
        if sha256_file(Path(filename)) != expected:
            raise AssertionError(f"Protected input changed: {filename}")


def make_pipeline(name, config, inventory, context):
    """Construct fresh preprocessing and estimator; never reuse a fitted pipeline."""
    if name == "LR":
        model = LogisticRegression(**config[name])
    elif name == "RF":
        model = RandomForestClassifier(**config[name])
    else:
        model = FoldBalancedXGBClassifier(**config[name], audit_dir=str(WEIGHT_DIR),
                                         fit_context=context)
    return Pipeline([("preprocessor", ablation.build_preprocessor(config["features"], inventory)),
                     ("model", model)])


def checked_probabilities(estimator, X):
    np.testing.assert_array_equal(estimator.classes_, CLASSES)
    probabilities = estimator.predict_proba(X).astype(float)
    if probabilities.shape != (len(X), 5) or not np.isfinite(probabilities).all():
        raise AssertionError("Invalid probability matrix")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise AssertionError("Probabilities outside 0..1")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    return probabilities


def fit_base(name, X, y, train, validation, config, inventory, context, audits):
    """Audit actual train IDs and preprocessing statistics for every base fit."""
    if set(train) & set(validation) or len(train) >= len(y):
        raise AssertionError("Partition overlap or forbidden all-development fit")
    estimator = make_pipeline(name, config, inventory, context)
    estimator.fit(X.iloc[train], y.iloc[train])
    if name == "XGB":
        actual = estimator.named_steps["model"].estimator_.get_xgb_params()
        if any(actual[key] != value for key, value in config["XGB_fixed"].items()):
            raise AssertionError("Fitted XGB did not use the frozen objective/classes/metric")
    probabilities = checked_probabilities(estimator, X.iloc[validation])
    numerical = estimator.named_steps["preprocessor"].named_transformers_["numerical"]
    audits.append({
        "model": name, "context": context, "n_train": len(train),
        "training_ids_sha256": id_hash(y.iloc[train].index),
        "prediction_ids_sha256": id_hash(y.iloc[validation].index),
        "class_counts": y.iloc[train].value_counts().sort_index().tolist(),
        "numerical_imputer_medians": numerical.named_steps["imputer"].statistics_.tolist(),
        "scaler_n_samples_seen": int(numerical.named_steps["scaler"].n_samples_seen_),
        "preprocessor_freshly_fitted": True,
        "XGB_fixed_verified": config["XGB_fixed"] if name == "XGB" else None,
    })
    return probabilities


def evaluate(X, y, inventory, config, outer_folds, inner_folds):
    """Inner OOF -> meta fit -> outer-training refits -> outer validation only."""
    predictions, inner_tables, fit_audits, meta_audits = [], [], [], []
    for number, (train, valid) in enumerate(outer_folds, 1):
        print(f"Outer fold {number}/5: generating inner OOF probabilities", flush=True)
        meta_X = np.full((len(train), len(META_COLUMNS)), np.nan)
        visits = np.zeros((len(train), len(BASES)), dtype=int)
        assignments = np.full(len(train), -1, dtype=int)
        for inner_number, (inner_train, inner_valid) in enumerate(inner_folds[number], 1):
            assignments[inner_valid] = inner_number
            for position, name in enumerate(BASES):
                context = f"outer{number}_inner{inner_number}_{name}"
                probabilities = fit_base(name, X, y, train[inner_train], train[inner_valid],
                                         config, inventory, context, fit_audits)
                meta_X[inner_valid, position * 5:(position + 1) * 5] = probabilities
                visits[inner_valid, position] += 1
        if not (visits == 1).all() or not np.isfinite(meta_X).all():
            raise AssertionError("Meta features must be exactly one inner OOF prediction per base/row")
        inner_table = pd.DataFrame(meta_X, columns=META_COLUMNS)
        inner_table.insert(0, "y_true", y.iloc[train].to_numpy())
        inner_table.insert(0, "SEQN", y.iloc[train].index.to_numpy())
        inner_table.insert(0, "inner_fold", assignments)
        inner_table.insert(0, "outer_fold", number)
        inner_tables.append(inner_table)
        meta = LogisticRegression(**config["meta_LR"])
        meta.fit(meta_X, y.iloc[train])
        meta_audits.append({"outer_fold": number, "n_train": len(train),
                            "training_ids_sha256": id_hash(y.iloc[train].index),
                            "training_features": "inner OOF only", "n_meta_features": 15,
                            "params": config["meta_LR"], "coef": meta.coef_.tolist(),
                            "intercept": meta.intercept_.tolist(),
                            "class_counts": y.iloc[train].value_counts().sort_index().tolist()})
        # These SAME outer refits serve standalone, voting and stacking comparisons.
        outer_probabilities = {}
        for name in BASES:
            outer_probabilities[name] = fit_base(
                name, X, y, train, valid, config, inventory, f"outer{number}_refit_{name}", fit_audits)
        for name, weights in config["voting_weights_LR_RF_XGB"].items():
            outer_probabilities[name] = np.average(
                np.stack([outer_probabilities[b] for b in BASES]), axis=0, weights=weights)
        meta_valid = np.column_stack([outer_probabilities[b] for b in BASES])
        outer_probabilities["Stacking"] = checked_probabilities(meta, meta_valid)
        for name in MODELS:
            probabilities = outer_probabilities[name]
            table = pd.DataFrame(probabilities, columns=PROB_COLUMNS)
            table.insert(0, "y_pred", probabilities.argmax(axis=1))
            table.insert(0, "y_true", y.iloc[valid].to_numpy())
            table.insert(0, "SEQN", y.iloc[valid].index.to_numpy())
            table.insert(0, "fold", number)
            table.insert(0, "model", name)
            predictions.append(table)
        print(f"Outer fold {number}/5 complete; parameters remain frozen", flush=True)
    return (pd.concat(predictions, ignore_index=True), pd.concat(inner_tables, ignore_index=True),
            fit_audits, meta_audits)


def summarize(predictions):
    """Keep mean fold metrics distinct from pooled-OOF per-class metrics."""
    fold_rows, summaries, per_class = [], [], []
    for name in MODELS:
        rows = predictions[predictions.model.eq(name)]
        if rows.SEQN.duplicated().any():
            raise AssertionError("Repeated outer validation participant")
        model_folds = []
        for fold, values in rows.groupby("fold", sort=True):
            row = {"model": name, "feature_set": "F4", "fold": fold,
                   "n_validation": len(values),
                   **ablation.summarize_metrics(values.y_true, values.y_pred)}
            _, recall, f1, support = precision_recall_fscore_support(
                values.y_true, values.y_pred, labels=CLASSES, zero_division=0)
            for c in CLASSES:
                row.update({f"Recall_C{c}": recall[c], f"F1_C{c}": f1[c],
                            f"support_C{c}": int(support[c])})
            model_folds.append(row)
        fold_rows.extend(model_folds)
        scores = pd.DataFrame(model_folds)
        summary = {"model": name, "feature_set": "F4", "n_features_raw": 34}
        for metric in ablation.METRICS:
            summary[f"{metric}_mean"] = scores[metric].mean()
            summary[f"{metric}_std"] = scores[metric].std(ddof=1)
            if metric != "macro_f1":
                summary[metric] = summary[f"{metric}_mean"]
        summaries.append(summary)
        precision, recall, f1, support = precision_recall_fscore_support(
            rows.y_true, rows.y_pred, labels=CLASSES, zero_division=0)
        for c in CLASSES:
            per_class.append({"model": name, "feature_set": "F4", "class": f"C{c}",
                              "recall": recall[c], "f1": f1[c], "precision": precision[c],
                              "support": int(support[c]),
                              "true_positive": int(((rows.y_true == c) & (rows.y_pred == c)).sum()),
                              "predicted_count": int((rows.y_pred == c).sum()),
                              "recall_cv_mean": scores[f"Recall_C{c}"].mean(),
                              "recall_cv_std": scores[f"Recall_C{c}"].std(ddof=1),
                              "f1_cv_mean": scores[f"F1_C{c}"].mean(),
                              "f1_cv_std": scores[f"F1_C{c}"].std(ddof=1),
                              "aggregation": "pooled_development_outer_oof"})
    folds = pd.DataFrame(fold_rows)
    paired = folds.pivot(index="fold", columns="model", values="macro_f1")[MODELS].reset_index()
    for name in [*VOTING, "Stacking"]:
        paired[f"delta_{name}_minus_RF"] = paired[name] - paired.RF
    return pd.DataFrame(summaries), pd.DataFrame(per_class), folds, paired


def select_candidate(summary, paired, config):
    scores = summary.set_index("model")
    rule = config["selection_rule"]
    eligible, comparisons = [], []
    for name in [*VOTING, "Stacking"]:
        delta = paired[name] - paired.RF
        passed = (delta.mean() >= rule["min_macro_f1_gain"] and
                  (delta > 0).sum() >= rule["min_folds_improved"] and
                  scores.loc[name, "macro_f1_std"] <= scores.loc["RF", "macro_f1_std"])
        comparisons.append({"model": name, "mean_delta_vs_RF": float(delta.mean()),
                            "paired_delta_std": float(delta.std(ddof=1)),
                            "folds_improved": int((delta > 0).sum()),
                            "qualifies": bool(passed)})
        if passed:
            eligible.append(name)
    winner = max(eligible, key=lambda name: scores.loc[name, "macro_f1_mean"]) if eligible else "RF"
    return {"candidate": winner, "feature_set": "F4", "selection_rule": rule,
            "comparisons": comparisons, "requires_user_confirmation_before_final_refit": True,
            "full_development_model_fitted": False, "heldout_used": False,
            "historically_untouched": False, "config_sha256": sha256_file(CONFIG_PATH)}


def plot_results(summary, paired):
    import os
    with tempfile.TemporaryDirectory(prefix="depsev-ensemble-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = ["LR consensus", "RF tuned", "XGB baseline", "Voting 1:1:1", "Voting 1:2:1", "Stacking"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
        ordered = summary.set_index("model").loc[MODELS]
        axes[0].errorbar(ordered.macro_f1_mean, np.arange(6), xerr=ordered.macro_f1_std,
                         fmt="o", capsize=5, color="#24608a")
        axes[0].set_yticks(np.arange(6), labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Macro-F1 (mean ± sample SD across 5 folds)")
        axes[0].set_title("Frozen F4 configurations")
        for name, label in [("RF", "RF tuned"), ("SoftVoting", "Voting 1:1:1"),
                            ("SoftVoting_RF2", "Voting 1:2:1"), ("Stacking", "Stacking")]:
            axes[1].plot(paired.fold, paired[name], marker="o", label=label)
        axes[1].set_xticks(range(1, 6))
        axes[1].set_xlabel("Same locked outer fold")
        axes[1].set_ylabel("Macro-F1")
        axes[1].set_title("Paired outer-validation performance")
        axes[1].legend(fontsize=9)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.suptitle("Development only (N=4,045) — no held-out evaluation", fontsize=13)
        fig.savefig(FIGURES_DIR / "ensemble_comparison.png", dpi=180)
        plt.close(fig)


def main():
    """Freeze, evaluate, save evidence, then STOP before a final-model refit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    split = load_final_split()
    development, original_inventory = ablation.load_inputs()
    X, groups, inventory, _ = prepare_features(development, original_inventory)
    features = [f for group in "PCLD" for f in groups[group]]
    X = X[features]
    X.index = development.SEQN.astype(int)
    y = pd.Series(development.severity.to_numpy(dtype=int), index=X.index, name="severity")
    if set(X.index) != set(split["development_seqn"]) or set(X.index) & set(split["test_seqn"]):
        raise AssertionError("Only locked development participants are allowed")
    folds, _ = ablation.lock_folds(development)
    inner = lock_partitions(y, folds)
    config = frozen_configuration(features)
    fixed_xgb = verify_xgb_fixed(config)
    manifest = lock_manifest(development, config)
    verify_protected(manifest)
    print(f"Frozen F4 features ({len(features)}): {features}", flush=True)
    print(f"Development N={len(y)}; class counts={y.value_counts().sort_index().to_dict()}", flush=True)
    print("LR consensus; exact tuned RF; baseline XGB. Held-out excluded.", flush=True)
    print(f"Verified wrapper hard-coded XGB parameters: {fixed_xgb}", flush=True)
    if args.audit_only:
        print("Audit complete; no model fitted.")
        return
    if RESULT_PATH.exists() or (ARTIFACTS_DIR / "ensemble_candidate.json").exists():
        raise FileExistsError("Completed ensemble results exist; refusing to overwrite or rerun")
    start = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        predictions, inner_oof, fits, meta_fits = evaluate(X, y, inventory, config, folds, inner)
    verify_protected(manifest)
    summary, per_class, fold_results, paired = summarize(predictions)
    for frame, name in [(summary, "ensemble_results.csv"),
                        (per_class, "ensemble_per_class_results.csv"),
                        (paired, "ensemble_fold_comparison.csv"),
                        (fold_results, "ensemble_fold_results.csv"),
                        (predictions, "ensemble_oof_predictions.csv"),
                        (inner_oof, "ensemble_inner_oof_predictions.csv")]:
        frame.to_csv(TABLES_DIR / name, index=False)
    save_json({"base_fits": fits, "meta_fits": meta_fits}, ARTIFACTS_DIR / "ensemble_fit_audit.json")
    candidate = select_candidate(summary, paired, config)
    save_json(candidate, ARTIFACTS_DIR / "ensemble_candidate.json")
    plot_results(summary, paired)
    outputs = sorted(TABLES_DIR.glob("ensemble_*.csv")) + [
        FIGURES_DIR / "ensemble_comparison.png", ARTIFACTS_DIR / "ensemble_fit_audit.json",
        ARTIFACTS_DIR / "ensemble_candidate.json"]
    save_json({"started_utc": start, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "manifest_sha256": sha256_file(MANIFEST_PATH),
               "outputs_sha256": {str(p): sha256_file(p) for p in outputs},
               "n_base_fits": len(fits), "n_meta_fits": len(meta_fits),
               "heldout_used": False, "full_development_model_fitted": False},
              ARTIFACTS_DIR / "ensemble_run.json")
    print(summary[["model", "macro_f1_mean", "macro_f1_std", "balanced_accuracy"]].to_string(index=False))
    print(paired.to_string(index=False))
    print(f"Candidate: {candidate['candidate']}/F4. STOP: awaiting confirmation before final refit.")


if __name__ == "__main__":
    main()
