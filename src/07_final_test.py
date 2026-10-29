"""Freeze/refit RF once, then perform one supplementary held-out evaluation.

Stages: --stage freeze, --stage refit, --stage evaluate, or --stage all (default).
Existing completed stages are verified and reused, never refitted/re-predicted.
"""

import argparse
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from config import ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR
from final_protocol import (
    SPEC, SCHEMA, MAPPING, MODEL, MODEL_LOCK, PREDICTIONS, EVALUATION_LOCK,
    EVALUATION_NAME, CLASSES, ablation, now, freeze_specification,
    verify_specification, load_features, make_final_pipeline, load_frozen_model,
    verify_model_lock, verify_evaluation, write_new_json, write_new_csv,
)
from protocol import sha256_file
from tuning_estimators import id_hash


FIT_CLAIM = ARTIFACTS_DIR / "final_refit_started.json"
EVALUATION_CLAIM = ARTIFACTS_DIR / "supplementary_heldout_started.json"


def refit():
    """Fit imputer, scaler/encoder and RF together, using 4,045 development rows."""
    spec = verify_specification()
    if MODEL_LOCK.exists():
        verify_model_lock()
        print("Existing final model verified; no refit.", flush=True)
        return
    if FIT_CLAIM.exists() or MODEL.exists():
        raise RuntimeError("Incomplete prior refit exists; refusing to silently refit/overwrite")
    X, y, inventory = load_features("development")
    if X.columns.tolist() != spec["features"] or id_hash(X.index) != spec["hashes"]["development_row_order_sha256"]:
        raise AssertionError("Development input/order differs from frozen specification")
    pipeline = make_final_pipeline(spec, inventory)
    write_new_json({"started_utc": now(), "spec_sha256": sha256_file(SPEC),
                    "development_n": len(y), "heldout_used_for_fit": False}, FIT_CLAIM)
    print("Fitting final preprocessing + RF pipeline on development N=4045", flush=True)
    with threadpool_limits(limits=1):
        pipeline.fit(X, y)
    numerical = pipeline.named_steps["preprocessor"].named_transformers_["numerical"]
    np.testing.assert_allclose(numerical["imputer"].statistics_, X[spec["preprocessing"]["numerical"]["columns"]].median())
    if numerical["scaler"].n_samples_seen_ != 4045:
        raise AssertionError("Unexpected preprocessor fit size")
    with MODEL.open("xb") as stream:
        joblib.dump(pipeline, stream, compress=3)
    # Serialization check uses development only; no held-out access here.
    restored = joblib.load(MODEL)
    np.testing.assert_allclose(pipeline.predict_proba(X.iloc[:20]), restored.predict_proba(X.iloc[:20]), atol=0, rtol=0)
    fitted_metadata = {
        "fit_completed_utc": now(), "training_n": len(y),
        "training_ids_sha256": id_hash(X.index),
        "class_counts": y.value_counts().sort_index().tolist(),
        "numerical_medians": numerical["imputer"].statistics_.tolist(),
        "scaler_n_samples_seen": int(numerical["scaler"].n_samples_seen_),
        "encoded_feature_names": pipeline.named_steps["preprocessor"].get_feature_names_out().tolist(),
        "heldout_used_for_fit": False,
    }
    metadata_path = ARTIFACTS_DIR / "final_preprocessor_metadata.json"
    write_new_json(fitted_metadata, metadata_path)
    verify_specification()
    write_new_json({**fitted_metadata, "spec_sha256": sha256_file(SPEC),
                    "model_sha256": sha256_file(MODEL), "locked_utc": now(),
                    "schema_sha256": sha256_file(SCHEMA), "mapping_sha256": sha256_file(MAPPING),
                    "metadata_sha256": sha256_file(metadata_path),
                    "selected_before_heldout": True, "heldout_used_for_selection": False,
                    "historically_untouched": False}, MODEL_LOCK)
    print(f"Final model fitted and locked: {MODEL}", flush=True)


def plot_confusion(matrix):
    with tempfile.TemporaryDirectory(prefix="depsev-final-mpl-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), layout="constrained")
        labels = [f"C{i}" for i in CLASSES]
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
            ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(normalized, display_labels=labels).plot(
            ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f", im_kw={"vmin": 0, "vmax": 1})
        axes[0].set_title("Counts (N=1,009)")
        axes[1].set_title("Row-normalized recall")
        fig.suptitle("Supplementary held-out evaluation — frozen RF/F4", fontsize=13)
        fig.savefig(FIGURES_DIR / "final_confusion_matrix.png", dpi=180)
        plt.close(fig)


def supplementary_evaluation():
    """One predict_proba call; class decisions and metrics reuse that same output."""
    if EVALUATION_LOCK.exists():
        verify_evaluation()
        print("Saved supplementary held-out evaluation verified; no new prediction.", flush=True)
        return
    if EVALUATION_CLAIM.exists() or PREDICTIONS.exists():
        raise RuntimeError("Prior evaluation attempt exists; NEVER automatically predict held-out again")
    pipeline, lock = load_frozen_model()
    # This is the first stage allowed to materialize held-out predictors/labels.
    X, y, _ = load_features("heldout")
    if X.columns.tolist() != verify_specification()["features"]:
        raise AssertionError("Held-out feature schema differs from frozen F4")
    write_new_json({"started_utc": now(), "evaluation": EVALUATION_NAME,
                    "model_sha256": lock["model_sha256"], "spec_sha256": sha256_file(SPEC),
                    "prediction_call_limit": 1, "heldout_row_order_sha256": id_hash(X.index),
                    "historically_untouched": False}, EVALUATION_CLAIM)
    with threadpool_limits(limits=1):
        probabilities = pipeline.predict_proba(X)  # The ONLY held-out prediction call.
    if probabilities.shape != (1009, 5) or not np.isfinite(probabilities).all():
        raise AssertionError("Invalid held-out probabilities; do not rerun prediction")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    predicted = pipeline.classes_[probabilities.argmax(axis=1)]
    predictions = pd.DataFrame({"SEQN": X.index.to_numpy(), "y_true": y.to_numpy(), "y_pred": predicted})
    for i, code in enumerate(pipeline.classes_):
        predictions[f"prob_C{code}"] = probabilities[:, i]
    write_new_csv(predictions, PREDICTIONS)
    metrics = {"evaluation": EVALUATION_NAME, "model": "Random Forest", "feature_set": "F4",
               "n": len(y), **ablation.summarize_metrics(y, predicted),
               "accuracy": accuracy_score(y, predicted), "historically_untouched": False}
    metrics_path = TABLES_DIR / "final_heldout_metrics.csv"
    write_new_csv(pd.DataFrame([metrics]), metrics_path)
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=CLASSES, zero_division=0)
    per_class = pd.DataFrame([{
        "evaluation": EVALUATION_NAME, "class": f"C{code}", "precision": precision[code],
        "recall": recall[code], "f1": f1[code], "support": int(support[code]),
        "true_positive": int(((y.to_numpy() == code) & (predicted == code)).sum()),
        "predicted_count": int((predicted == code).sum())} for code in CLASSES])
    class_path = TABLES_DIR / "final_heldout_per_class_results.csv"
    write_new_csv(per_class, class_path)
    matrix = confusion_matrix(y, predicted, labels=CLASSES)
    matrix_path = TABLES_DIR / "final_confusion_matrix.csv"
    matrix_table = pd.DataFrame(matrix, columns=[f"pred_C{c}" for c in CLASSES])
    matrix_table.insert(0, "true_class", [f"C{c}" for c in CLASSES])
    write_new_csv(matrix_table, matrix_path)
    plot_confusion(matrix)
    verify_model_lock()
    outputs = [PREDICTIONS, metrics_path, class_path, matrix_path, FIGURES_DIR / "final_confusion_matrix.png"]
    write_new_json({"completed_utc": now(), "evaluation": EVALUATION_NAME,
                    "model_sha256": lock["model_sha256"], "spec_sha256": sha256_file(SPEC),
                    "prediction_calls": 1, "prediction_method": "predict_proba; y_pred=classes[argmax]",
                    "heldout_n": len(y), "historically_untouched": False,
                    "heldout_used_for_selection": False, "model_changed_after_evaluation": False,
                    "outputs_sha256": {str(path): sha256_file(path) for path in outputs}}, EVALUATION_LOCK)
    print(EVALUATION_NAME, flush=True)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(per_class.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["freeze", "refit", "evaluate", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ["freeze", "all"]:
        spec = freeze_specification()
        print(f"Frozen final specification: {SPEC}; features={len(spec['features'])}; N=4045", flush=True)
    if args.stage in ["refit", "all"]:
        refit()
    if args.stage in ["evaluate", "all"]:
        supplementary_evaluation()


if __name__ == "__main__":
    main()
