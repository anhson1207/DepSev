"""Paired XGBoost robustness on original-membership age20 development.

Repeated CV: 5 folds x 10 repeats, same splits for F3 and F4, fixed parameters.
Historical sensitivity helper is retained but NOT called by the final protocol.
"""

from importlib import import_module
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from ablation_features import prepare_features
from config import ARTIFACTS_DIR, FIGURES_DIR, TABLES_DIR, PROTOCOL_VERSION, experiment_path
from protocol import FINAL_SPLIT
from utils import load_json, save_json


ablation = import_module("05_feature_ablation")


def paired_table(fold_table, indices):
    table = fold_table.pivot(index=indices, columns="feature_set", values="macro_f1")
    table = table[["F3", "F4"]].rename(
        columns={"F3": "F3_macro_f1", "F4": "F4_macro_f1"})
    table["delta_F4_minus_F3"] = table.F4_macro_f1 - table.F3_macro_f1
    return table.reset_index()


def distribution(values):
    """Descriptive spread, not an IID standard error or confidence interval."""
    return {"n": len(values), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)), "median": float(values.median()),
            "min": float(values.min()), "max": float(values.max()),
            "q025": float(values.quantile(0.025)), "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)), "q975": float(values.quantile(0.975))}


def repeated_experiment(X, y, ids, inventory, groups, repeated_folds):
    folds_out, classes_out, weights_out, predictions_out = [], [], [], []
    repeat_scores = []
    for repeat in range(1, 11):
        folds = repeated_folds[(repeat - 1) * 5: repeat * 5]
        assignments = np.full(len(y), -1, dtype=int)
        for number, (_, valid) in enumerate(folds, 1):
            assignments[valid] = number
        for feature_set in ["F3", "F4"]:
            features = [feature for group in ablation.FEATURE_SETS[feature_set]
                        for feature in groups[group]]
            result = ablation.evaluate_configuration(
                X[features], y, ids, folds, assignments,
                ablation.make_models()["XGBoost"], "XGBoost", feature_set, inventory)
            summary, per_class, predictions, fold_rows, weight_rows, _ = result
            repeat_scores.append({"repeat": repeat, **summary})
            for rows, collection in [(fold_rows, folds_out), (per_class, classes_out),
                                     (weight_rows, weights_out)]:
                collection.extend({"repeat": repeat, **row} for row in rows)
            predictions.insert(0, "repeat", repeat)
            predictions_out.append(predictions)
        print(f"Repeat {repeat:2d}/10: F3={repeat_scores[-2]['macro_f1_mean']:.4f}, "
              f"F4={repeat_scores[-1]['macro_f1_mean']:.4f}", flush=True)
    fold_table = pd.DataFrame(folds_out)
    repeat_table = pd.DataFrame(repeat_scores)
    paired = paired_table(fold_table, ["repeat", "fold"])
    repeat_deltas = paired.groupby("repeat", as_index=False)[
        ["F3_macro_f1", "F4_macro_f1", "delta_F4_minus_F3"]].mean()
    summary_rows = []
    for feature_set, rows in fold_table.groupby("feature_set", sort=False):
        row = {"feature_set": feature_set, "model": "XGBoost",
               **{f"macro_f1_{key}": value for key, value in distribution(rows.macro_f1).items()}}
        for metric in ablation.METRICS[1:]:
            row[f"{metric}_mean"] = float(rows[metric].mean())
            row[f"{metric}_std"] = float(rows[metric].std(ddof=1))
        summary_rows.append(row)
    delta_summary = []
    for scope, changes in [("50_paired_validation_folds", paired.delta_F4_minus_F3),
                           ("10_paired_repeat_means", repeat_deltas.delta_F4_minus_F3)]:
        delta_summary.append({"scope": scope, **distribution(changes),
                              "positive": int((changes > 0).sum()),
                              "negative": int((changes < 0).sum()),
                              "zero": int((changes == 0).sum())})
    return {
        "repeated_cv_fold_results.csv": fold_table,
        "repeated_cv_summary.csv": pd.DataFrame(summary_rows),
        "repeated_cv_paired_differences.csv": paired,
        "repeated_cv_difference_summary.csv": pd.DataFrame(delta_summary),
        "repeated_cv_repeat_means.csv": repeat_deltas,
        "repeated_cv_per_class.csv": pd.DataFrame(classes_out),
        "repeated_cv_weights.csv": pd.DataFrame(weights_out),
        "repeated_cv_oof_predictions.csv": pd.concat(predictions_out, ignore_index=True),
    }


def sensitivity_experiment(X, primary_y, ids, inventory, groups, folds, assignments):
    """Compare four-class training to post-hoc collapsed primary OOF predictions."""
    if PROTOCOL_VERSION == "original_split_preserved_age20":
        raise RuntimeError("Exploratory sensitivity is retired; final protocol runs five-class robustness only")
    y = primary_y.clip(upper=3).copy()
    summaries, classes, predictions, fold_rows, weights, matrices = [], [], [], [], [], []
    for feature_set in ["F3", "F4"]:
        features = [feature for group in ablation.FEATURE_SETS[feature_set]
                    for feature in groups[group]]
        result = ablation.evaluate_configuration(
            X[features], y, ids, folds, assignments,
            ablation.make_models(n_classes=4)["XGBoost"], "XGBoost", feature_set, inventory,
            classes=np.arange(4), label_prefix="S")
        summary, per_class, oof, metrics, weight_rows, confusion = result
        summaries.append(summary)
        classes.extend(per_class)
        predictions.append(oof)
        fold_rows.extend(metrics)
        weights.extend(weight_rows)
        matrices.extend(confusion)
        print(f"Sensitivity 4-class {feature_set}: "
              f"Macro-F1 {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}", flush=True)

    oof_four = pd.concat(predictions, ignore_index=True)
    primary = pd.read_csv(TABLES_DIR / "ablation_oof_predictions.csv")
    primary = primary[primary.model.eq("XGBoost") & primary.feature_set.isin(["F3", "F4"])]
    if not set(primary.SEQN).issubset(set(ids)):
        raise RuntimeError("Primary OOF includes non-development IDs")
    comparisons = []
    for feature_set in ["F3", "F4"]:
        primary_rows = primary[primary.feature_set.eq(feature_set)].set_index("SEQN").loc[ids]
        secondary_rows = oof_four[oof_four.feature_set.eq(feature_set)].set_index("SEQN").loc[ids]
        np.testing.assert_array_equal(primary_rows.y_true, primary_y)
        for name, output in [("five_class_predictions_collapsed_to_four", primary_rows.y_pred.clip(upper=3)),
                             ("four_class_training", secondary_rows.y_pred)]:
            precision, recall, f1, support = precision_recall_fscore_support(
                y, output, labels=np.arange(4), zero_division=0)
            predictions_array = output.to_numpy()
            comparisons.append({
                "feature_set": feature_set, "method": name,
                "macro_f1_pooled_oof_four_labels": f1_score(y, output, average="macro", labels=np.arange(4)),
                "S3_precision": precision[3], "S3_recall": recall[3], "S3_f1": f1[3],
                "S3_support": int(support[3]), "predicted_S3": int((predictions_array == 3).sum()),
                "S3_true_positive": int(((predictions_array == 3) & (y.to_numpy() == 3)).sum()),
                "original_C3_detected_as_S3": int(((primary_y.to_numpy() == 3) & (predictions_array == 3)).sum()),
                "original_C4_detected_as_S3": int(((primary_y.to_numpy() == 4) & (predictions_array == 3)).sum()),
            })
    counts = y.value_counts().reindex(range(4), fill_value=0)
    return {
        "sensitivity_4class_results.csv": pd.DataFrame(summaries),
        "sensitivity_4class_per_class.csv": pd.DataFrame(classes),
        "sensitivity_4class_oof_predictions.csv": oof_four,
        "sensitivity_4class_fold_results.csv": pd.DataFrame(fold_rows),
        "sensitivity_4class_weights.csv": pd.DataFrame(weights),
        "sensitivity_4class_confusion_matrices.csv": pd.DataFrame(matrices),
        "sensitivity_severe_comparison.csv": pd.DataFrame(comparisons),
        "sensitivity_class_distribution.csv": pd.DataFrame({
            "class": [f"S{code}" for code in range(4)], "development_count": counts.to_numpy()}),
    }


def plot_robustness(tables):
    cache = Path(tempfile.gettempdir()) / "depsev_ml_matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores = tables["repeated_cv_fold_results.csv"]
    paired = tables["repeated_cv_paired_differences.csv"]
    repeat = tables["repeated_cv_repeat_means.csv"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].boxplot([scores.loc[scores.feature_set.eq(f), "macro_f1"] for f in ["F3", "F4"]],
                    tick_labels=["F3", "F4"], showmeans=True)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("50 validation folds per candidate")
    axes[1].hist(paired.delta_F4_minus_F3, bins=12, color="#0072B2", edgecolor="white")
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Paired Macro-F1: F4 - F3")
    axes[1].set_ylabel("Validation folds")
    axes[1].set_title("Paired differences (same splits)")
    axes[2].plot(repeat["repeat"], repeat.delta_F4_minus_F3, "o-", color="#009E73")
    axes[2].axhline(0, color="black", linestyle="--")
    axes[2].set_xlabel("Repeat")
    axes[2].set_ylabel("Mean difference: F4 - F3")
    axes[2].set_title("Five-fold mean per repeat")
    fig.suptitle("Original split preserved | Age >= 20 | Development only | XGBoost\nDescriptive spread; overlapping CV folds are not independent experiments")
    fig.tight_layout()
    fig.savefig(experiment_path(FIGURES_DIR, "repeated_cv_F3_F4.png"), dpi=160)
    plt.close(fig)


def main():
    development, original_inventory = ablation.load_inputs()
    X, groups, inventory, _ = prepare_features(development, original_inventory)
    primary_folds, primary_assignments = ablation.lock_folds(development)
    y = development.severity.astype(int)
    ids = development.SEQN.astype(int)
    fixed = pd.read_csv(experiment_path(TABLES_DIR, "ablation_fold_results.csv"))
    fixed_pair = paired_table(fixed[fixed.model.eq("XGBoost") & fixed.feature_set.isin(["F3", "F4"])], ["fold"])
    print("Fixed 5-fold XGBoost comparison:\n" + fixed_pair.to_string(index=False), flush=True)

    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    repeated_folds = list(splitter.split(X, y))
    per_repeat_assignments = []
    for repeat in range(10):
        values = np.full(len(y), -1, dtype=int)
        for fold, (_, validation) in enumerate(repeated_folds[repeat * 5: (repeat + 1) * 5], 1):
            values[validation] = fold
        per_repeat_assignments.append(values.tolist())
    # With the same ordering and seed, repeat 1 must reproduce the fixed experiment.
    np.testing.assert_array_equal(per_repeat_assignments[0], primary_assignments)
    ablation.lock_json({"protocol": PROTOCOL_VERSION, "protocol_version": PROTOCOL_VERSION,
                       "splitter": "RepeatedStratifiedKFold", "n_splits": 5,
                       "n_repeats": 10, "random_state": 42,
                       "split_sha256": ablation.sha256_file(FINAL_SPLIT),
                       "development_seqn": ids.tolist(),
                       "validation_fold_by_repeat": per_repeat_assignments},
                      experiment_path(ARTIFACTS_DIR, "repeated_cv_folds.json"))
    paths = [experiment_path(ARTIFACTS_DIR, name) for name in ["split_indices.json", "cv_folds.json", "ablation_config.json"]]
    paths += [ARTIFACTS_DIR / "split_indices.json", Path(__file__).with_name("protocol.py"),
              Path(__file__).with_name("config.py")]
    paths += [ablation.PROCESSED_DATA_DIR / "cohort.csv",
              experiment_path(TABLES_DIR, "ablation_fold_results.csv"),
              experiment_path(TABLES_DIR, "ablation_oof_predictions.csv"),
              Path(__file__), Path(ablation.__file__), Path(__file__).with_name("ablation_features.py")]
    hashes = {str(path): ablation.sha256_file(path) for path in paths}
    ablation.lock_json({"protocol": PROTOCOL_VERSION, "protocol_version": PROTOCOL_VERSION,
                       "inputs_and_code_sha256": hashes,
                       "primary_target": "C0-C4 unchanged",
                       "robustness": "XGB F3/F4 paired 5x10 development CV; no tuning",
                       "sensitivity": "not run for final protocol; primary five-class only",
                       "weighting": "n_train/(n_classes*class_count_train), recalculated inside each fold",
                       "warning": "Fold SD and empirical quantiles are descriptive, not IID uncertainty intervals"},
                      experiment_path(ARTIFACTS_DIR, "robustness_config.json"))
    tables = {"xgb_fixed_F3_F4_comparison.csv": fixed_pair}
    tables.update(repeated_experiment(X, y, ids, inventory, groups, repeated_folds))
    if hashes != {str(path): ablation.sha256_file(path) for path in paths}:
        raise RuntimeError("Protected input/code changed during robustness run")
    for name, table in tables.items():
        table.to_csv(experiment_path(TABLES_DIR, name), index=False)
    plot_robustness(tables)
    print("\nRepeated CV summary:\n" + tables["repeated_cv_summary.csv"].to_string(index=False))
    print("\nPaired difference summary:\n" + tables["repeated_cv_difference_summary.csv"].to_string(index=False))
    print("Held-out test not evaluated; primary five-class target unchanged.")


if __name__ == "__main__":
    main()
