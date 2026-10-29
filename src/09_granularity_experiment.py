"""Compare frozen RF/F4 across four prespecified target resolutions.

Use only development records and the existing validation-fold assignments.
Run --audit-only before the experiment. Existing results are never overwritten.
All headline scores are fold means with sample SD (ddof=1); pooled OOF scores
are explicitly distinguished. No final model is refitted by this experiment.
"""

import argparse
from importlib import metadata
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import final_protocol as frozen
from ablation_features import read_development_rows
from config import ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, TABLES_DIR
from protocol import FINAL_SPLIT, load_final_split, sha256_file
from tuning_estimators import id_hash
from utils import load_json


FOLDS_PATH = ARTIFACTS_DIR / "cv_folds_final_protocol.json"
PROTOCOL_PATH = ARTIFACTS_DIR / "granularity_protocol.json"
RUN_PATH = ARTIFACTS_DIR / "granularity_run.json"
TASKS = {
    "T2": {"mapping": [0, 1, 1, 1, 1], "ranges": ["0-4", "5-27"],
           "labels": ["minimal symptoms", "symptomatic"]},
    "T3": {"mapping": [0, 1, 2, 2, 2], "ranges": ["0-4", "5-9", "10-27"],
           "labels": ["minimal", "mild", "moderate or higher"]},
    "T4": {"mapping": [0, 1, 2, 3, 3], "ranges": ["0-4", "5-9", "10-14", "15-27"],
           "labels": ["minimal", "mild", "moderate", "moderately severe or severe"]},
    "T5": {"mapping": [0, 1, 2, 3, 4], "ranges": ["0-4", "5-9", "10-14", "15-19", "20-27"],
           "labels": ["minimal", "mild", "moderate", "moderately severe", "severe"]},
}
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "macro_precision",
           "macro_recall", "weighted_f1", "mean_absolute_class_error", "within_one_class_accuracy"]
COLORS = ["#577590", "#43aa8b", "#e9c46a", "#f2844b", "#b73b52"]


def class_name(task, code):
    return f"C{code}" if task == "T5" else f"{task}_{code}"


def map_target(y, task):
    if not np.isin(y, np.arange(5)).all():
        raise ValueError("Granularity requires the original C0-C4 target")
    return pd.Series(np.asarray(TASKS[task]["mapping"])[np.asarray(y, dtype=int)],
                     index=y.index, name=task)


def load_development():
    """Never materialize held-out predictors or labels; read its ID list only."""
    spec = frozen.verify_specification()
    X, y, inventory = frozen.load_features("development")
    if X.shape != (4045, 34) or X.columns.tolist() != spec["features"]:
        raise AssertionError("Development membership or F4 schema changed")
    expected = dict(n_estimators=150, max_depth=8, min_samples_split=10,
                    min_samples_leaf=8, max_features=0.5,
                    class_weight="balanced_subsample", random_state=42, n_jobs=1)
    if any(spec["hyperparameters"][k] != value for k, value in expected.items()):
        raise AssertionError("RF is not the user-frozen configuration")
    split = load_final_split()
    if set(X.index) != set(split["development_seqn"]) or set(X.index) & set(split["test_seqn"]):
        raise AssertionError("Unexpected membership")
    scores = read_development_rows(PROCESSED_DATA_DIR / "cohort.csv", X.index,
                                   ["SEQN", "PHQ9_total"]).set_index("SEQN").loc[X.index, "PHQ9_total"]
    if not scores.between(0, 27).all() or not scores.eq(scores.round()).all():
        raise AssertionError("Invalid PHQ-9 total")
    np.testing.assert_array_equal(np.digitize(scores, [5, 10, 15, 20]), y)
    lock = load_json(FOLDS_PATH)
    np.testing.assert_array_equal(lock["development_seqn"], X.index)
    np.testing.assert_array_equal(lock["development_severity"], y)
    assert lock["split_sha256"] == sha256_file(FINAL_SPLIT)
    assert (lock["n_splits"], lock["shuffle"], lock["random_state"]) == (5, True, 42)
    assignments = np.asarray(lock["validation_fold"], dtype=int)
    assert set(assignments) == {1, 2, 3, 4, 5}
    # Consume the saved assignments directly. Do not create a new splitter.
    folds = [(np.flatnonzero(assignments != f), np.flatnonzero(assignments == f))
             for f in range(1, 6)]
    return X, y, inventory, scores, folds, assignments, spec


def verify_hashes(mapping):
    for filename, expected in mapping.items():
        if sha256_file(PROJECT_ROOT / filename) != expected:
            raise AssertionError(f"Protected file changed: {filename}")


def freeze_protocol(X, spec):
    """Lock the design before fitting and record byte hashes of existing work."""
    design = {
        "experiment": "target_granularity_RF_F4", "development_n": len(X),
        "tasks": TASKS, "features": X.columns.tolist(), "RF": spec["hyperparameters"],
        "preprocessing": {**spec["preprocessing"], "learned_only_from": "each development training fold during Pipeline.fit"},
        "preprocessing_fit_scope": "each development training fold independently",
        "folds_source": str(FOLDS_PATH.relative_to(PROJECT_ROOT)),
        "folds_sha256": sha256_file(FOLDS_PATH), "development_ids_sha256": id_hash(X.index),
        "final_spec_sha256": sha256_file(frozen.SPEC), "heldout_used": False,
        "new_split_created": False, "hyperparameter_search": False,
        "decision_rule": "argmax probabilities; no threshold adjustment",
        "sd_convention": "sample SD across the same 5 folds, ddof=1; not a confidence interval",
        "class_weight_note": "balanced_subsample policy stays fixed; effective bootstrap class weights change with merged targets",
        "ordinal_note": "T3-T5 only; class-index distance, not PHQ-9 score distance; units differ after merging",
        "interpretation": "within-development comparison after prior RF selection; no external generalization claim",
        "overlap_methods": {
            "numerical": "observed values: median/IQR, Spearman rho with PHQ9_total, Cliff delta for adjacent classes",
            "cliff_delta": "2*U/(n_lower*n_higher)-1; negative means higher severity tends to have larger values",
            "numerical_overlap": "sum min(histogram probabilities); shared pooled-development decile edges per feature; tied edges collapsed",
            "categorical": "within-class proportions including explicit unknown; uncorrected Cramer V; expected-cell sparsity reported",
            "categorical_overlap": "sum min(category probabilities) for adjacent classes",
            "PCA": "fresh existing preprocessing fit on development only for exploratory PCA(2); never reused for model CV",
            "adjacent_confusion": "(a-to-b + b-to-a)/(support_a+support_b); also report equal-direction mean rate",
            "representatives": "three numerical features with largest histogram overlap per adjacent pair; report all features separately",
        },
        "method_sources": [
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html"],
        "versions": {p: metadata.version(p) for p in ["scikit-learn", "numpy", "pandas", "scipy", "matplotlib"]},
        "source_sha256": {f"src/{name}": sha256_file(PROJECT_ROOT / "src" / name)
                          for name in ["09_granularity_experiment.py", "10_class_overlap_analysis.py"]},
    }
    if PROTOCOL_PATH.exists():
        saved = load_json(PROTOCOL_PATH)
        if saved["design"] != design:
            raise AssertionError("Granularity protocol differs from its frozen design")
        verify_hashes(saved["preserved_files_sha256"])
        return saved
    paths = []
    for folder in ["src", "tests", "results", "artifacts"]:
        paths.extend(p for p in (PROJECT_ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and not p.name.startswith(("granularity_", "overlap_", "09_", "10_", "test_granularity"))
                     and p.name != "adjacent_class_overlap.csv")
    paths += [PROJECT_ROOT / "README.md", PROJECT_ROOT / "requirements.txt"]
    saved = {"created_utc": frozen.now(), "design": design,
             "preserved_files_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in paths},
             "preservation_note": "Old files are fingerprinted as opaque bytes only; no held-out values/metrics are parsed or used"}
    frozen.write_new_json(saved, PROTOCOL_PATH)
    return saved


def score_predictions(y, pred, n_classes):
    result = frozen.ablation.summarize_metrics(y, pred, classes=np.arange(n_classes))
    result["accuracy"] = accuracy_score(y, pred)
    error = np.abs(np.asarray(y) - np.asarray(pred))
    result["mean_absolute_class_error"] = float(error.mean()) if n_classes > 2 else np.nan
    result["within_one_class_accuracy"] = float((error <= 1).mean()) if n_classes > 2 else np.nan
    return result


def class_distribution(y):
    rows = []
    for task, definition in TASKS.items():
        target = map_target(y, task)
        for code, interval in enumerate(definition["ranges"]):
            n = int(target.eq(code).sum())
            rows.append({"task": task, "class": class_name(task, code), "class_index": code,
                         "phq9_range": interval, "label": definition["labels"][code],
                         "n": n, "percentage": 100 * n / len(y)})
    return pd.DataFrame(rows)


def save_table(frame, name):
    path = TABLES_DIR / name
    frozen.write_new_csv(frame, path)
    return path


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "depsev-granularity-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save_figure(fig, name):
    path = FIGURES_DIR / name
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=180, bbox_inches="tight")
    pyplot().close(fig)
    return path


def plot_outputs(distribution, summary, confusion):
    plt = pyplot()
    outputs = []
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True, layout="constrained")
    for ax, task in zip(axes, TASKS):
        rows = distribution[distribution.task.eq(task)]
        bars = ax.bar(rows["class"], rows.n, color=COLORS[:len(rows)])
        ax.bar_label(bars, fontsize=9, padding=3)
        ax.set(title=task, ylim=(0, 3500))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Development participants")
    fig.suptitle("Class counts as severity resolution increases (N = 4,045)")
    outputs.append(save_figure(fig, "granularity_class_distribution.png"))
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for metric, label, color in [("accuracy", "Accuracy", COLORS[0]),
                                 ("macro_f1", "Macro-F1", COLORS[4]),
                                 ("balanced_accuracy", "Balanced accuracy", COLORS[1])]:
        ax.errorbar(summary.task, summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"],
                    marker="o", capsize=4, label=label, color=color)
    ax.set(ylim=(0, 1), ylabel="Mean ± sample SD across 5 fixed folds",
           xlabel="Target resolution", title="Frozen RF/F4 on development only")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    outputs.append(save_figure(fig, "granularity_performance.png"))
    for task in TASKS:
        n = len(TASKS[task]["ranges"])
        matrix = confusion[confusion.task.eq(task)].pivot(index="true_index", columns="predicted_index", values="count").to_numpy()
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        labels = [class_name(task, i) for i in range(n)]
        fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]}\n{normalized[i, j]:.1%}", ha="center", va="center",
                        color="white" if normalized[i, j] > 0.55 else "#202020")
        ax.set(xticks=range(n), yticks=range(n), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted class", ylabel="True class",
               title=f"{task}: development OOF confusion\nCounts and row percentages, N = 4,045")
        fig.colorbar(im, ax=ax, label="Fraction of true class", shrink=0.85)
        outputs.append(save_figure(fig, f"granularity_{task}_confusion.png"))
    return outputs


def run_experiment(data, protocol):
    X, original_y, inventory, _, folds, assignments, spec = data
    if RUN_PATH.exists() or (TABLES_DIR / "granularity_class_distribution.csv").exists():
        raise FileExistsError("Granularity output already exists; refusing to overwrite or rerun silently")
    distribution = class_distribution(original_y)
    outputs = [save_table(distribution, "granularity_class_distribution.csv")]
    fold_rows, per_fold_rows, oof_frames, fit_audit = [], [], [], []
    for task in TASKS:
        y = map_target(original_y, task)
        n_classes = len(TASKS[task]["ranges"])
        probabilities = np.full((len(y), n_classes), np.nan)
        visits = np.zeros(len(y), dtype=int)
        for fold_number, (train, valid) in enumerate(folds, 1):
            assert not set(train) & set(valid)
            assert set(y.iloc[train]) == set(y.iloc[valid]) == set(range(n_classes))
            estimator = frozen.make_final_pipeline(spec, inventory)
            estimator.fit(X.iloc[train], y.iloc[train])
            np.testing.assert_array_equal(estimator.classes_, np.arange(n_classes))
            proba = estimator.predict_proba(X.iloc[valid])
            np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-12)
            pred = proba.argmax(axis=1)
            probabilities[valid] = proba
            visits[valid] += 1
            scores = score_predictions(y.iloc[valid], pred, n_classes)
            fold_rows.append({"task": task, "n_classes": n_classes, "fold": fold_number,
                              "n_train": len(train), "n_validation": len(valid), **scores})
            precision, recall, f1, support = precision_recall_fscore_support(
                y.iloc[valid], pred, labels=np.arange(n_classes), zero_division=0)
            for code in range(n_classes):
                per_fold_rows.append({"task": task, "fold": fold_number, "class": class_name(task, code),
                                      "class_index": code, "precision": precision[code], "recall": recall[code],
                                      "f1": f1[code], "support": int(support[code]),
                                      "training_support": int(y.iloc[train].eq(code).sum())})
            numerical = estimator.named_steps["preprocessor"].named_transformers_["numerical"]
            assert int(numerical.named_steps["scaler"].n_samples_seen_) == len(train)
            numeric_names = spec["preprocessing"]["numerical"]["columns"]
            medians = numerical.named_steps["imputer"].statistics_
            np.testing.assert_allclose(medians, X.iloc[train][numeric_names].median())
            assert estimator.named_steps["model"].get_params() == spec["hyperparameters"]
            fit_audit.append({"task": task, "fold": fold_number, "n_train": len(train),
                              "training_ids_sha256": id_hash(X.index[train]),
                              "validation_ids_sha256": id_hash(X.index[valid]),
                              "numerical_imputer_medians": medians.tolist(),
                              "scaler_n_samples_seen": int(numerical.named_steps["scaler"].n_samples_seen_),
                              "training_class_counts": y.iloc[train].value_counts().sort_index().tolist(),
                              "RF_config_sha256": frozen.json_hash(estimator.named_steps["model"].get_params())})
            print(f"{task} fold {fold_number}/5: accuracy={scores['accuracy']:.4f}, macro_f1={scores['macro_f1']:.4f}", flush=True)
        assert (visits == 1).all() and np.isfinite(probabilities).all()
        frame = pd.DataFrame({"task": task, "SEQN": X.index, "fold": assignments,
                              "original_class": original_y.to_numpy(), "y_true": y.to_numpy(),
                              "y_pred": probabilities.argmax(axis=1)})
        for code in range(n_classes):
            frame[f"prob_{code}"] = probabilities[:, code]
        oof_frames.append(frame)
    oof = pd.concat(oof_frames, ignore_index=True)
    # This is development OOF from the earlier frozen-RF ensemble comparison.
    reference = pd.read_csv(TABLES_DIR / "ensemble_oof_predictions.csv")
    reference = reference[reference.model.eq("RF")].set_index("SEQN").loc[X.index]
    new = oof[oof.task.eq("T5")].set_index("SEQN").loc[X.index]
    np.testing.assert_array_equal(new[["fold", "y_true", "y_pred"]], reference[["fold", "y_true", "y_pred"]])
    np.testing.assert_allclose(new[[f"prob_{i}" for i in range(5)]],
                               reference[[f"prob_C{i}" for i in range(5)]], rtol=0, atol=1e-14)
    folds_table, per_fold = pd.DataFrame(fold_rows), pd.DataFrame(per_fold_rows)
    summary_rows, class_rows, confusion_rows = [], [], []
    for task in TASKS:
        rows = folds_table[folds_table.task.eq(task)]
        n_classes = len(TASKS[task]["ranges"])
        summary = {"task": task, "n_classes": n_classes, "model": "Random Forest", "feature_set": "F4"}
        for metric in METRICS:
            summary[f"{metric}_mean"] = rows[metric].mean()
            summary[f"{metric}_std"] = rows[metric].std(ddof=1)
        summary_rows.append(summary)
        pred = oof[oof.task.eq(task)]
        p, r, f, s = precision_recall_fscore_support(pred.y_true, pred.y_pred, labels=np.arange(n_classes), zero_division=0)
        matrix = confusion_matrix(pred.y_true, pred.y_pred, labels=np.arange(n_classes))
        for code in range(n_classes):
            cv = per_fold[per_fold.task.eq(task) & per_fold.class_index.eq(code)]
            row = {"task": task, "class": class_name(task, code), "class_index": code,
                   "precision_oof": p[code], "recall_oof": r[code], "f1_oof": f[code],
                   "support": int(s[code]), "true_positives": int(matrix[code, code]),
                   "predicted_count": int(matrix[:, code].sum())}
            for metric in ["precision", "recall", "f1"]:
                row[f"{metric}_mean"] = cv[metric].mean()
                row[f"{metric}_std"] = cv[metric].std(ddof=1)
            class_rows.append(row)
            for other in range(n_classes):
                confusion_rows.append({"task": task, "true_index": code, "predicted_index": other,
                                       "true_class": class_name(task, code), "predicted_class": class_name(task, other),
                                       "count": int(matrix[code, other]), "row_fraction": matrix[code, other] / s[code]})
    summary, confusion = pd.DataFrame(summary_rows), pd.DataFrame(confusion_rows)
    delta_rows, paired_rows = [], []
    for before, after in zip(list(TASKS)[:-1], list(TASKS)[1:]):
        for metric in ["accuracy", "macro_f1", "balanced_accuracy"]:
            paired = folds_table.pivot(index="fold", columns="task", values=metric)
            delta = paired[after] - paired[before]
            delta_rows.append({"comparison": f"{after}-{before}", "metric": metric,
                               "delta_mean": delta.mean(), "delta_std": delta.std(ddof=1),
                               "positive_folds": int(delta.gt(0).sum()), "negative_folds": int(delta.lt(0).sum())})
            paired_rows.extend({"comparison": f"{after}-{before}", "metric": metric, "fold": int(fold),
                                "before": paired.loc[fold, before], "after": paired.loc[fold, after], "delta": value}
                               for fold, value in delta.items())
    for frame, name in [(folds_table, "fold_results"), (summary, "results"),
                        (pd.DataFrame(class_rows), "per_class_results"), (per_fold, "per_class_fold_results"),
                        (oof, "oof_predictions"), (confusion, "confusion_matrices"),
                        (pd.DataFrame(delta_rows), "paired_deltas"), (pd.DataFrame(paired_rows), "paired_fold_deltas")]:
        outputs.append(save_table(frame, f"granularity_{name}.csv"))
    outputs.extend(plot_outputs(distribution, summary, confusion))
    audit_path = ARTIFACTS_DIR / "granularity_fit_audit.json"
    frozen.write_new_json(fit_audit, audit_path)
    outputs.append(audit_path)
    verify_hashes(protocol["preserved_files_sha256"])
    frozen.write_new_json({"completed_utc": frozen.now(), "heldout_used": False,
                           "new_split_created": False, "n_fits": len(fit_audit),
                           "T5_reproduces_prior_RF_OOF": True, "all_old_files_unchanged": True,
                           "protocol_sha256": sha256_file(PROTOCOL_PATH),
                           "outputs_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in outputs}}, RUN_PATH)
    print(summary.to_string(index=False), flush=True)
    print("PASS: T5 reproduces prior RF OOF; all previous files preserved; held-out not used.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    data = load_development()
    X, y, _, _, folds, _, spec = data
    protocol = freeze_protocol(X, spec)
    print("Frozen F4 features: " + ", ".join(X.columns), flush=True)
    print(class_distribution(y).to_string(index=False), flush=True)
    print("Existing fold sizes: " + str([len(valid) for _, valid in folds]), flush=True)
    print("AUDIT PASS: development only, original fold assignments, frozen RF/F4.", flush=True)
    if not args.audit_only:
        run_experiment(data, protocol)


if __name__ == "__main__":
    main()
