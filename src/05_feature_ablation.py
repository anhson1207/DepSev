"""Run Dummy and 16 weighted baselines on the same locked development folds.

Run: python src/05_feature_ablation.py
Audit and lock features/folds without fitting: add --audit-only.
No train/test splitting, tuning, or held-out evaluation occurs in this script.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ablation_features import CODEBOOKS, prepare_features, read_development_rows
from config import (ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, TABLES_DIR,
                    PRIMARY_MIN_AGE, PROTOCOL_VERSION, experiment_path)
from protocol import FINAL_SPLIT, ORIGINAL_SPLIT, load_final_split
from utils import ensure_output_directories, load_json, save_json


CLASSES = np.arange(5)
FEATURE_SETS = {"F1": "P", "F2": "PC", "F3": "PCL", "F4": "PCLD"}
METRICS = ["macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "weighted_f1"]


def sha256_file(path):
    """Fingerprint inputs without changing them."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_json(payload, path):
    """Reuse an identical lock; refuse to silently replace an existing protocol."""
    if path.exists():
        # XGBoost defaults include NaN, which is not equal to itself in Python.
        if json.dumps(load_json(path), sort_keys=True) != json.dumps(payload, sort_keys=True):
            raise RuntimeError(f"Existing lock differs: {path}. Review before creating a new run.")
    else:
        save_json(payload, path)


def load_inputs():
    """Load only development records, preserving previous F1 cohort row order."""
    split = load_final_split()
    if split.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Rebuild the authorized age20 cohort/split before this experiment")
    development_ids = split["development_seqn"]
    test_ids = split["test_seqn"]
    if (len(set(development_ids)) != len(development_ids)
            or len(set(test_ids)) != len(test_ids)
            or set(development_ids) & set(test_ids)):
        raise ValueError("Split IDs are duplicated or development and test overlap")
    inventory = pd.read_csv(TABLES_DIR / "feature_inventory.csv")
    development = read_development_rows(
        PROCESSED_DATA_DIR / "cohort.csv", development_ids,
        ["SEQN", *inventory.feature.tolist(), "severity"],
    )
    if not development.severity.isin(CLASSES).all():
        raise ValueError("Invalid development severity labels")
    if not development.RIDAGEYR.ge(PRIMARY_MIN_AGE).all():
        raise ValueError("Development cohort violates age20 eligibility")
    if development.severity.value_counts().reindex(CLASSES, fill_value=0).min() < 5:
        raise ValueError("A development class is too small for five-fold CV")
    return development, inventory


def lock_folds(development):
    """Persist one validation-fold assignment per ID, identical across experiments."""
    y = development.severity.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(development[["SEQN"]], y))
    assignments = np.full(len(y), -1, dtype=int)
    for number, (train, validation) in enumerate(folds, start=1):
        if set(train) & set(validation) or not (assignments[validation] == -1).all():
            raise RuntimeError("Invalid CV fold partition")
        assignments[validation] = number
    if (assignments < 1).any():
        raise RuntimeError("Development rows are missing a validation fold")
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": PROTOCOL_VERSION,
        "split_sha256": sha256_file(FINAL_SPLIT),
        "splitter": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42,
        "row_order": "original cohort order (matches 04_train_baselines.py)",
        "development_seqn": development.SEQN.astype(int).tolist(),
        "development_severity": y.tolist(),
        "validation_fold": assignments.tolist(),
    }
    lock_json(lock, experiment_path(ARTIFACTS_DIR, "cv_folds.json"))
    return folds, assignments


def build_preprocessor(features, inventory):
    """Each fresh Pipeline fits imputation, scaling and encoding on one train fold."""
    types = inventory.set_index("feature").feature_type
    numerical = [feature for feature in features if types[feature] == "numerical"]
    categorical = [feature for feature in features if types[feature] == "categorical"]
    return ColumnTransformer([
        ("numerical", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing_or_unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def make_models(n_classes=5):
    """Preserve the previous F1 model parameters; XGBoost is now fold-weighted."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=1),
        "XGBoost": XGBClassifier(
            objective="multi:softprob", num_class=n_classes, n_estimators=200,
            max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=1),
    }


def summarize_metrics(y_true, prediction, classes=CLASSES):
    """Unweighted validation scores; class balancing applies only during fitting."""
    return {
        "macro_f1": f1_score(y_true, prediction, labels=classes, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "macro_precision": precision_score(y_true, prediction, labels=classes,
                                           average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, prediction, labels=classes,
                                     average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, prediction, labels=classes,
                                average="weighted", zero_division=0),
    }


def evaluate_configuration(X, y, ids, folds, assignments, model, model_name,
                           feature_set, inventory, classes=CLASSES, label_prefix="C"):
    """Fit each fold once, collecting metrics, OOF predictions and weight evidence."""
    dummy = isinstance(model, DummyClassifier)
    oof = np.full(len(y), -1, dtype=int)
    visits = np.zeros(len(y), dtype=int)
    fold_rows, weight_rows = [], []
    for fold_number, (train, validation) in enumerate(folds, start=1):
        X_train, X_valid = X.iloc[train], X.iloc[validation]
        y_train, y_valid = y.iloc[train], y.iloc[validation]
        if set(train) & set(validation) or set(y_train) != set(classes):
            raise RuntimeError("Invalid training partition or missing training class")
        if dummy:
            estimator = clone(model)
        else:
            estimator = Pipeline([
                ("preprocessor", build_preprocessor(X.columns.tolist(), inventory)),
                ("model", clone(model)),
            ])
        fit_params = {}
        if model_name == "XGBoost":
            # This is intentionally INSIDE the fold, after selecting y_train.
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            counts = y_train.value_counts().reindex(classes)
            expected = np.array([len(y_train) / (len(classes) * counts[c]) for c in y_train])
            np.testing.assert_allclose(sample_weight, expected)
            fit_params["model__sample_weight"] = sample_weight
            for code in classes:
                weight_rows.append({
                    "feature_set": feature_set, "model": model_name, "fold": fold_number,
                    "class": f"{label_prefix}{code}", "n_train": len(train),
                    "class_count_train": int(counts[code]),
                    "sample_weight": float(sample_weight[np.asarray(y_train) == code][0]),
                    "class_weight_sum": float(sample_weight[np.asarray(y_train) == code].sum()),
                })
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(X_train, y_train, **fit_params)
        prediction = estimator.predict(X_valid).astype(int)
        oof[validation] = prediction
        visits[validation] += 1

        row = {"feature_set": feature_set, "model": model_name, "fold": fold_number,
               "n_train": len(train), "n_validation": len(validation),
               **summarize_metrics(y_valid, prediction, classes)}
        _, recalls, f1s, supports = precision_recall_fscore_support(
            y_valid, prediction, labels=classes, zero_division=0)
        for code in classes:
            row[f"Recall_{label_prefix}{code}"] = recalls[code]
            row[f"F1_{label_prefix}{code}"] = f1s[code]
            row[f"support_{label_prefix}{code}"] = int(supports[code])
        fold_rows.append(row)
    if (oof < 0).any() or not (visits == 1).all():
        raise RuntimeError("Each participant must have exactly one validation prediction per repeat")

    fold_table = pd.DataFrame(fold_rows)
    summary = {"feature_set": feature_set, "model": model_name,
               "n_features_raw": 0 if dummy else X.shape[1]}
    for metric in METRICS:
        summary[f"{metric}_mean"] = float(fold_table[metric].mean())
        summary[f"{metric}_std"] = float(fold_table[metric].std(ddof=1))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, oof, labels=classes, zero_division=0)
    class_rows = []
    for code in classes:
        class_rows.append({
            "feature_set": feature_set, "model": model_name, "class": f"{label_prefix}{code}",
            "recall": recall[code], "f1": f1[code], "precision": precision[code],
            "support": int(support[code]), "predicted_count": int((oof == code).sum()),
            "true_positive": int(((oof == code) & (y.to_numpy() == code)).sum()),
            "recall_cv_mean": float(fold_table[f"Recall_{label_prefix}{code}"].mean()),
            "recall_cv_std": float(fold_table[f"Recall_{label_prefix}{code}"].std(ddof=1)),
            "f1_cv_mean": float(fold_table[f"F1_{label_prefix}{code}"].mean()),
            "f1_cv_std": float(fold_table[f"F1_{label_prefix}{code}"].std(ddof=1)),
            "aggregation": "pooled_development_oof",
        })
    prediction_table = pd.DataFrame({
        "feature_set": feature_set, "model": model_name, "SEQN": ids.to_numpy(),
        "fold": assignments, "y_true": y.to_numpy(), "y_pred": oof,
    })
    matrix = confusion_matrix(y, oof, labels=classes)
    confusion_rows = [{"feature_set": feature_set, "model": model_name,
                       "true_class": f"{label_prefix}{a}", "predicted_class": f"{label_prefix}{b}",
                       "count": int(matrix[a, b])} for a in classes for b in classes]
    return summary, class_rows, prediction_table, fold_rows, weight_rows, confusion_rows


def incremental_results(fold_results):
    """Compute paired changes using the same five fold assignments."""
    rows = []
    real_models = fold_results[fold_results.feature_set.ne("Dummy")]
    for model, model_results in real_models.groupby("model", sort=False):
        pivot = model_results.pivot(index="fold", columns="feature_set", values="macro_f1")
        for before, after in [("F1", "F2"), ("F2", "F3"), ("F3", "F4")]:
            changes = pivot[after] - pivot[before]
            rows.append({"model": model, "comparison": f"{after}-{before}",
                         "macro_f1_delta": float(changes.mean()),
                         "paired_fold_delta_std": float(changes.std(ddof=1)),
                         "folds_improved": int((changes > 0).sum())})
    return pd.DataFrame(rows)


def plot_comparison(results, dummy_results):
    """Export a standalone research figure showing all fold mean ± SD scores."""
    cache = Path(tempfile.gettempdir()) / "depsev_ml_matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 6.5))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for offset, ((model, rows), color) in enumerate(zip(results.groupby("model", sort=False), colors)):
        rows = rows.set_index("feature_set").loc[list(FEATURE_SETS)]
        x = np.arange(4) + (offset - 1.5) * 0.045
        axis.errorbar(x, rows.macro_f1_mean, yerr=rows.macro_f1_std,
                      label=model, marker="o", capsize=4, linewidth=1.8, color=color)
    for row, style in zip(dummy_results.itertuples(index=False), ["--", ":"]):
        axis.axhline(row.macro_f1_mean, linestyle=style, color="#555555", label=row.model)
    axis.set_xticks(range(4), ["F1: P", "F2: P + C", "F3: P + C + L", "F4: P + C + L + D"])
    axis.set_ylabel("Macro-F1 (mean ± fold SD)")
    axis.set_title("Original split preserved · age >= 20 · development only\nSame 5 stratified folds · balanced training weights")
    axis.set_ylim(0, max(0.4, float((results.macro_f1_mean + results.macro_f1_std).max()) + 0.07))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(experiment_path(FIGURES_DIR, "feature_group_comparison.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="Lock inputs/features/folds without training")
    args = parser.parse_args()
    ensure_output_directories()
    protected_paths = [ARTIFACTS_DIR / "split_indices.json", FINAL_SPLIT, ORIGINAL_SPLIT,
                       PROCESSED_DATA_DIR / "cohort.csv",
                       ARTIFACTS_DIR / "feature_schema.json", TABLES_DIR / "feature_inventory.csv",
                       experiment_path(ARTIFACTS_DIR, "cohort_summary.json")]
    before_hashes = {str(path): sha256_file(path) for path in protected_paths}
    development, inventory = load_inputs()
    X, groups, final_inventory, audit = prepare_features(development, inventory)
    folds, assignments = lock_folds(development)
    y = development.severity.astype(int)
    ids = development.SEQN.astype(int)
    models = make_models()
    metadata = {
        "protocol": PROTOCOL_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protected_inputs_sha256": before_hashes,
        "cv_folds_sha256": sha256_file(experiment_path(ARTIFACTS_DIR, "cv_folds.json")),
        "groups": groups, "feature_sets": FEATURE_SETS,
        "preprocessing": {"numerical": "median then StandardScaler within training fold",
                          "categorical": "explicit missing_or_unknown then OneHotEncoder(handle_unknown=ignore)"},
        "weights": {"LR_DT_RF": "class_weight=balanced",
                    "XGBoost": "compute_sample_weight(balanced, y_train_fold) passed as model__sample_weight",
                    "validation": "unweighted", "Dummy": "unweighted"},
        "std_ddof": 1, "per_class_aggregation": "pooled development OOF; fold mean/std also saved",
        "python": sys.version,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "xgboost", "matplotlib"]},
        "model_params": {
            name: {key: "NaN" if isinstance(value, float) and np.isnan(value) else value
                   for key, value in model.get_params().items()}
            for name, model in models.items()
        },
        "sources": CODEBOOKS,
        "code_sha256": {path.name: sha256_file(path) for path in
                        [Path(__file__).resolve(), Path(__file__).with_name("ablation_features.py"),
                         Path(__file__).with_name("protocol.py"), Path(__file__).with_name("config.py")]},
    }
    # NaN parameters are logged as strings so the manifest is valid JSON.
    lock_json(metadata, experiment_path(ARTIFACTS_DIR, "ablation_config.json"))
    final_inventory.to_csv(experiment_path(TABLES_DIR, "ablation_feature_inventory.csv"), index=False)
    audit.to_csv(experiment_path(TABLES_DIR, "structural_missingness_audit.csv"), index=False)
    print(f"Development: {len(development)}; class counts: {y.value_counts().sort_index().to_dict()}", flush=True)
    for group, columns in groups.items():
        print(f"{group} ({len(columns)}): {', '.join(columns)}", flush=True)
    print("Fixed 5-fold assignments: artifacts/cv_folds_final_protocol.json", flush=True)
    if args.audit_only:
        print("Audit complete; no model fitted.")
        return

    runs = [("Dummy", name, model, X[groups["P"][:1]]) for name, model in {
        "Dummy most_frequent": DummyClassifier(strategy="most_frequent"),
        "Dummy stratified": DummyClassifier(strategy="stratified", random_state=42),
    }.items()]
    for feature_set, selected_groups in FEATURE_SETS.items():
        columns = [feature for group in selected_groups for feature in groups[group]]
        runs.extend((feature_set, name, model, X[columns]) for name, model in models.items())
    summaries, per_class, predictions, all_folds, weights, confusion = [], [], [], [], [], []
    for feature_set, model_name, model, features in runs:
        result = evaluate_configuration(features, y, ids, folds, assignments, model,
                                        model_name, feature_set, final_inventory)
        summary, class_rows, prediction_table, fold_rows, weight_rows, confusion_rows = result
        summaries.append(summary)
        per_class.extend(class_rows)
        predictions.append(prediction_table)
        all_folds.extend(fold_rows)
        weights.extend(weight_rows)
        confusion.extend(confusion_rows)
        print(f"{feature_set:5s} {model_name:22s} Macro-F1 "
              f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}; "
              f"BA {summary['balanced_accuracy_mean']:.4f}", flush=True)

    if before_hashes != {str(path): sha256_file(path) for path in protected_paths}:
        raise RuntimeError("Protected input changed during run; results not published")
    summary_table = pd.DataFrame(summaries)
    results = summary_table[summary_table.feature_set.ne("Dummy")].reset_index(drop=True)
    dummy_results = summary_table[summary_table.feature_set.eq("Dummy")]
    fold_table = pd.DataFrame(all_folds)
    tables = {
        "baseline_F1_results.csv": results[results.feature_set.eq("F1")],
        "feature_ablation_results.csv": results,
        "dummy_cv_results.csv": dummy_results,
        "per_class_cv_results.csv": pd.DataFrame(per_class),
        "ablation_fold_results.csv": fold_table,
        "ablation_oof_predictions.csv": pd.concat(predictions, ignore_index=True),
        "xgboost_fold_weights.csv": pd.DataFrame(weights),
        "ablation_confusion_matrices.csv": pd.DataFrame(confusion),
        "incremental_value.csv": incremental_results(fold_table),
    }
    for filename, table in tables.items():
        table.to_csv(experiment_path(TABLES_DIR, filename), index=False)
    plot_comparison(results, dummy_results)
    best = results.loc[results.macro_f1_mean.idxmax()]
    print(f"\nHighest development CV mean: {best.feature_set} / {best.model} = {best.macro_f1_mean:.4f}")
    print("Protected input hashes unchanged. Held-out test not evaluated.")


if __name__ == "__main__":
    main()
