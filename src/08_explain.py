"""Development-only permutation importance and multiclass Tree SHAP for frozen RF.

Requires the saved supplementary held-out evaluation first. No fitting, feature
selection, threshold changes, or held-out predictions occur here. --stage can be
permutation, shap, or all. Completed outputs are verified and reused.
"""

import argparse
from importlib import metadata
from itertools import permutations
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, f1_score, recall_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits

from config import ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR, PROJECT_ROOT
from final_protocol import (SPEC, MODEL, EVALUATION_LOCK, CLASSES, now, load_features,
                            load_frozen_model, verify_model_lock, verify_evaluation,
                            write_new_json, write_new_csv)
from protocol import sha256_file
from tuning_estimators import id_hash
from utils import load_json


PROTOCOL = ARTIFACTS_DIR / "final_explainability_protocol.json"
PERM_LOCK = ARTIFACTS_DIR / "permutation_importance_run.json"
SHAP_LOCK = ARTIFACTS_DIR / "shap_run.json"
N_REPEATS = 10


def verify_saved(path):
    record = load_json(path)
    if record["model_sha256"] != sha256_file(MODEL):
        raise AssertionError("Explanation/model mismatch")
    for filename, expected in record["outputs_sha256"].items():
        if sha256_file(Path(filename)) != expected:
            raise AssertionError(f"Saved explanation changed: {filename}")
    return record


def freeze_explanation_protocol(X):
    payload = {"model_sha256": sha256_file(MODEL), "spec_sha256": sha256_file(SPEC),
               "saved_evaluation_sha256": sha256_file(EVALUATION_LOCK),
               "source_sha256": sha256_file(PROJECT_ROOT / "src/08_explain.py"),
               "partition": "development", "n": len(X), "IDs_sha256": id_hash(X.index),
               "permutation": {"n_repeats": N_REPEATS, "random_state": 42, "n_jobs": 1,
                               "evaluation": "in-sample development, explanatory only",
                               "permuted_unit": "one original F4 column before preprocessing",
                               "metrics": "macro_f1, balanced_accuracy, F1/Recall for C0-C4",
                               "std_ddof": 1},
               "shap": {"explainer": "TreeExplainer", "feature_perturbation": "tree_path_dependent",
                        "model_output": "raw", "approximate": False,
                        "background": "training path counts stored in the final RF; class-balanced tree weights",
                        "samples": "all 4045 development participants",
                        "class_dimension": "detect actual output shape; validate against model.classes_",
                        "grouping": "sum signed one-hot attributions per source feature BEFORE abs",
                        "global": "mean absolute grouped SHAP over all participants and equally over 5 outputs",
                        "per_class": "mean absolute grouped SHAP over all participants for each output class"},
               "causal_interpretation": False, "heldout_used_for_explanations": False,
               "refit_or_model_change": False, "shap_version": metadata.version("shap")}
    if PROTOCOL.exists():
        if load_json(PROTOCOL) != payload:
            raise AssertionError("Explainability protocol changed; do not silently overwrite")
    else:
        write_new_json(payload, PROTOCOL)


def single_class_f1(y_true, y_pred, class_code):
    return f1_score(y_true, y_pred, labels=[class_code], average="macro", zero_division=0)


def single_class_recall(y_true, y_pred, class_code):
    return recall_score(y_true, y_pred, labels=[class_code], average="macro", zero_division=0)


def run_permutation(pipeline, X, y, inventory):
    if PERM_LOCK.exists():
        verify_saved(PERM_LOCK)
        print("Saved permutation importance verified.", flush=True)
        return
    scorers = {"macro_f1": make_scorer(f1_score, labels=CLASSES, average="macro", zero_division=0),
               "balanced_accuracy": make_scorer(balanced_accuracy_score)}
    for code in CLASSES:
        scorers[f"F1_C{code}"] = make_scorer(single_class_f1, class_code=int(code))
        scorers[f"Recall_C{code}"] = make_scorer(single_class_recall, class_code=int(code))
    print("Permutation importance: final RF, development only, 34 features × 10 repeats", flush=True)
    with threadpool_limits(limits=1):
        results = permutation_importance(pipeline, X, y, scoring=scorers,
                                         n_repeats=N_REPEATS, random_state=42, n_jobs=1)
    group = inventory.set_index("feature").group
    records = []
    for metric, result in results.items():
        baseline = float(scorers[metric](pipeline, X, y))
        for i, feature in enumerate(X.columns):
            records.append({"feature": feature, "group": group[feature], "metric": metric,
                            "class": metric.split("_")[-1] if metric.startswith(("F1_", "Recall_")) else "global",
                            "importance_mean": float(result.importances[i].mean()),
                            "importance_std": float(result.importances[i].std(ddof=1)),
                            "baseline_score": baseline, "n_repeats": N_REPEATS,
                            "partition": "development_in_sample", "interpretation": "baseline minus permuted score"})
    output = TABLES_DIR / "permutation_importance.csv"
    write_new_csv(pd.DataFrame(records), output)
    raw_path = ARTIFACTS_DIR / "permutation_importance_raw.npz"
    with raw_path.open("xb") as stream:
        np.savez_compressed(stream, features=np.asarray(X.columns, dtype=str),
                            **{name: value.importances for name, value in results.items()})
    verify_model_lock()
    write_new_json({"completed_utc": now(), "model_sha256": sha256_file(MODEL),
                    "protocol_sha256": sha256_file(PROTOCOL), "n": len(X),
                    "n_repeats": N_REPEATS, "heldout_used": False,
                    "outputs_sha256": {str(p): sha256_file(p) for p in [output, raw_path]}}, PERM_LOCK)


def normalize_shap_values(raw, n_samples, n_features, classes):
    """Detect version-dependent shape; reject ambiguous/missing class dimensions."""
    if isinstance(raw, list):
        raw_shapes = [list(np.asarray(value).shape) for value in raw]
        if len(raw) != len(classes) or any(np.asarray(value).shape != (n_samples, n_features) for value in raw):
            raise ValueError(f"Unexpected SHAP list shapes: {raw_shapes}")
        return np.stack(raw, axis=-1), {"return_type": "list", "raw_shapes": raw_shapes,
                                      "normalization": "stack per-output matrices on last axis"}
    array = np.asarray(raw.values if hasattr(raw, "values") else raw)
    expected = (n_samples, n_features, len(classes))
    matches = [axes for axes in permutations(range(array.ndim))
               if tuple(array.shape[axis] for axis in axes) == expected]
    if array.ndim != 3 or len(matches) != 1:
        raise ValueError(f"Ambiguous SHAP shape {array.shape}; expected uniquely mappable to {expected}")
    return np.transpose(array, matches[0]), {"return_type": type(raw).__name__,
                                           "raw_shape": list(array.shape),
                                           "canonical_shape": list(expected),
                                           "axis_permutation": list(matches[0])}


def encoded_sources(preprocessor):
    """Map each transformed column to its original F4 feature using fitted encoder metadata."""
    numerical = preprocessor.transformers_[0][2]
    categorical = preprocessor.transformers_[1][2]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
        raise AssertionError("Unexpected encoder grouping; source mapping must be reviewed")
    sources = list(numerical)
    for feature, categories in zip(categorical, encoder.categories_):
        sources.extend([feature] * len(categories))
    names = preprocessor.get_feature_names_out().tolist()
    if len(names) != len(sources):
        raise AssertionError("One-hot source mapping does not match transformed columns")
    return names, sources


def plot_shap_importance(table, class_code=None):
    import matplotlib.pyplot as plt
    column = "mean_abs_shap_global" if class_code is None else f"mean_abs_shap_C{class_code}"
    top = table.nlargest(15, column).sort_values(column)
    fig, ax = plt.subplots(figsize=(9.3, 6.4), layout="constrained")
    ax.barh(top.feature, top[column], color="#2b658c")
    ax.set_xlabel("Mean |grouped SHAP| (class-probability units)")
    label = "global (equal weight across C0–C4 outputs)" if class_code is None else f"output C{class_code}"
    ax.set_title(f"Final RF/F4 SHAP: {label}\nAll development participants, N=4,045", fontsize=12)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    suffix = "global" if class_code is None else f"C{class_code}"
    path = FIGURES_DIR / f"shap_summary_{suffix}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_shap(pipeline, X, inventory):
    if SHAP_LOCK.exists():
        verify_saved(SHAP_LOCK)
        print("Saved SHAP outputs verified.", flush=True)
        return
    import shap
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)  # Never fit during explanations.
    names, sources = encoded_sources(preprocessor)
    print(f"TreeExplainer: SHAP {shap.__version__}; input shape={transformed.shape}; classes={model.classes_}", flush=True)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    with threadpool_limits(limits=1):
        raw = explainer.shap_values(transformed, approximate=False, check_additivity=True)
    values, shape_info = normalize_shap_values(raw, len(X), len(names), model.classes_)
    print(f"Observed SHAP shape audit: {shape_info}", flush=True)
    base_values = np.asarray(explainer.expected_value)
    if base_values.shape != (len(model.classes_),):
        raise AssertionError("Expected-value class shape differs from fitted RF classes")
    probabilities = model.predict_proba(transformed)  # Development only.
    reconstructed = values.sum(axis=1) + base_values
    np.testing.assert_allclose(reconstructed, probabilities, atol=1e-7, rtol=1e-6)
    # Preserve additivity when grouping one-hot columns: sum signed values first.
    grouped = np.stack([values[:, np.asarray(sources) == feature, :].sum(axis=1)
                        for feature in X.columns], axis=1)
    np.testing.assert_allclose(grouped.sum(axis=1) + base_values, probabilities, atol=1e-7, rtol=1e-6)
    group = inventory.set_index("feature").group
    importance = pd.DataFrame({"feature": X.columns, "group": [group[f] for f in X.columns],
                               "mean_abs_shap_global": np.abs(grouped).mean(axis=(0, 2))})
    class_rows = []
    outputs = []
    for position, code in enumerate(model.classes_):
        importance[f"mean_abs_shap_C{code}"] = np.abs(grouped[:, :, position]).mean(axis=0)
        rows = pd.DataFrame({"class": f"C{code}", "feature": X.columns,
                             "mean_abs_shap": importance[f"mean_abs_shap_C{code}"],
                             "mean_signed_shap": grouped[:, :, position].mean(axis=0),
                             "explained_n": len(X), "scope": "all development participants for this output class"})
        rows = rows.sort_values("mean_abs_shap", ascending=False)
        class_path = TABLES_DIR / f"shap_importance_C{code}.csv"
        write_new_csv(rows, class_path)
        outputs.append(class_path)
        class_rows.append(rows)
    global_path = TABLES_DIR / "shap_global_importance.csv"
    write_new_csv(importance.sort_values("mean_abs_shap_global", ascending=False), global_path)
    per_class_path = TABLES_DIR / "shap_per_class_importance.csv"
    write_new_csv(pd.concat(class_rows, ignore_index=True), per_class_path)
    mapping_path = ARTIFACTS_DIR / "shap_encoded_feature_mapping.json"
    write_new_json({"encoded_feature_names": names, "original_features": sources,
                    "aggregation": "signed sum within each source feature before absolute value"}, mapping_path)
    values_path = ARTIFACTS_DIR / "shap_values_final_rf.npz"
    with values_path.open("xb") as stream:
        np.savez_compressed(stream, SEQN=X.index.to_numpy(), features=np.asarray(X.columns, dtype=str),
                            classes=model.classes_, grouped_values=grouped, encoded_values=values,
                            base_values=base_values, development_probabilities=probabilities)
    outputs.extend([global_path, per_class_path, mapping_path, values_path])
    with tempfile.TemporaryDirectory(prefix="depsev-shap-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        outputs.append(plot_shap_importance(importance))
        for code in model.classes_:
            outputs.append(plot_shap_importance(importance, code))
    verify_model_lock()
    write_new_json({"completed_utc": now(), "model_sha256": sha256_file(MODEL),
                    "protocol_sha256": sha256_file(PROTOCOL), "shap_version": shap.__version__,
                    "shape_audit": shape_info, "class_order": model.classes_.tolist(),
                    "additivity_max_absolute_error": float(np.max(np.abs(reconstructed - probabilities))),
                    "verified_output_space": "RandomForestClassifier.predict_proba probabilities",
                    "partition": "development", "explained_n": len(X), "heldout_used": False,
                    "causal_interpretation": False,
                    "outputs_sha256": {str(p): sha256_file(p) for p in outputs}}, SHAP_LOCK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["permutation", "shap", "all"], default="all")
    args = parser.parse_args()
    verify_evaluation()  # Explanations are allowed only AFTER saved supplementary evaluation.
    pipeline, _ = load_frozen_model()
    X, y, inventory = load_features("development")
    freeze_explanation_protocol(X)
    if args.stage in ["permutation", "all"]:
        run_permutation(pipeline, X, y, inventory)
    if args.stage in ["shap", "all"]:
        run_shap(pipeline, X, inventory)
    verify_evaluation()
    verify_model_lock()
    print("Explainability complete; frozen RF artifact unchanged.", flush=True)


if __name__ == "__main__":
    main()
