"""Frozen RF specification, provenance checks and explicitly scoped data loading."""

from datetime import datetime, timezone
import hashlib
from importlib import import_module, metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ablation_features import prepare_features, read_development_rows
from config import (ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR, PROJECT_ROOT,
                    PROCESSED_DATA_DIR, RAW_DATA_DIR, PHQ9_COLUMNS, PROTOCOL_VERSION)
from protocol import FINAL_SPLIT, load_final_split, sha256_file
from tuning_estimators import id_hash
from utils import load_json


ablation = import_module("05_feature_ablation")
SPEC = ARTIFACTS_DIR / "final_model_spec.json"
SCHEMA = ARTIFACTS_DIR / "final_feature_schema.json"
MAPPING = ARTIFACTS_DIR / "class_mapping.json"
MODEL = ARTIFACTS_DIR / "final_model.pkl"
MODEL_LOCK = ARTIFACTS_DIR / "final_model_lock.json"
PREDICTIONS = TABLES_DIR / "final_heldout_predictions.csv"
EVALUATION_LOCK = ARTIFACTS_DIR / "supplementary_heldout_evaluation.json"
EVALUATION_NAME = "supplementary held-out evaluation"
CLASSES = np.arange(5)


def now():
    return datetime.now(timezone.utc).isoformat()


def json_hash(value):
    """Canonical JSON SHA-256; sorted keys and compact UTF-8 representation."""
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def membership_hash(ids):
    return json_hash(sorted(int(i) for i in ids))


def write_new_json(value, path):
    """Exclusive creation prevents silent replacement of a frozen artifact."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def write_new_csv(frame, path):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def load_features(partition="development"):
    """Apply the unchanged deterministic feature rules to an explicit membership.

    The legacy helper is named read_development_rows/prepare_features but filters
    solely by the supplied IDs. Its rules fit no statistics and use no labels.
    Held-out IDs may be supplied ONLY after the final model lock has been verified.
    """
    split = load_final_split()
    if partition == "development":
        ids, expected_n = split["development_seqn"], 4045
    elif partition == "heldout":
        verify_model_lock()
        ids, expected_n = split["test_seqn"], 1009
    else:
        raise ValueError("Partition must be development or heldout")
    inventory = pd.read_csv(TABLES_DIR / "feature_inventory.csv")
    frame = read_development_rows(PROCESSED_DATA_DIR / "cohort.csv", ids,
                                  ["SEQN", *inventory.feature.tolist(), "severity"])
    if len(frame) != expected_n or not frame.RIDAGEYR.ge(20).all():
        raise AssertionError("Frozen membership/age eligibility changed")
    if not frame.severity.isin(CLASSES).all():
        raise AssertionError("Invalid class label")
    X, groups, final_inventory, _ = prepare_features(frame, inventory)
    features = [name for group in "PCLD" for name in groups[group]]
    X = X[features]
    X.index = frame.SEQN.astype(int)
    y = pd.Series(frame.severity.to_numpy(dtype=int), index=X.index, name="severity")
    if set(X.index) != set(ids):
        raise AssertionError("Requested partition IDs differ")
    return X, y, final_inventory


def freeze_specification():
    """Freeze inputs and choices using development and split IDs only, before fit."""
    if SPEC.exists():
        return verify_specification()
    if MODEL.exists() or MODEL_LOCK.exists() or PREDICTIONS.exists():
        raise RuntimeError("An unfrozen final artifact already exists; do not overwrite")
    ensemble = import_module("06_ensemble")
    ensemble.verify_protected(load_json(ensemble.MANIFEST_PATH))
    config = load_json(ensemble.CONFIG_PATH)
    candidate = load_json(ARTIFACTS_DIR / "ensemble_candidate.json")
    if candidate["candidate"] != "RF" or candidate["feature_set"] != "F4":
        raise AssertionError("Candidate differs from user-confirmed RF/F4")
    if candidate["config_sha256"] != sha256_file(ensemble.CONFIG_PATH):
        raise AssertionError("Candidate configuration provenance changed")
    split = load_final_split()
    X, y, inventory = load_features()
    if X.columns.tolist() != config["features"] or X.shape != (4045, 34):
        raise AssertionError("Frozen F4 schema changed")
    rf = config["RF"]
    tuned = load_json(ARTIFACTS_DIR / "tuned_model_configs.json")
    if rf != tuned["estimator_params"]:
        raise AssertionError("RF differs from tuned artifact")
    schema_fields = ["feature", "group", "feature_type", "source_file", "source_columns", "description"]
    schema = {"feature_set": "F4", "n_features": 34, "ordered_features": X.columns.tolist(),
              "columns": inventory[schema_fields].to_dict("records"),
              "input_contract": "34 F4 columns AFTER existing deterministic questionnaire "
                                "code/routing rules, BEFORE imputation/scaling/one-hot encoding",
              "categorical_values": "object strings; missing=np.nan",
              "numerical_values": "numeric; missing=np.nan",
              "deterministic_feature_source_sha256": sha256_file(PROJECT_ROOT / "src/ablation_features.py")}
    mapping = {"target": "PHQ-9 symptom severity", "source_items": PHQ9_COLUMNS,
               "requires_all_nine_items": True, "valid_item_scores": [0, 1, 2, 3],
               "class_order": CLASSES.tolist(), "classes": [
                   {"value": i, "class": f"C{i}", "label": label,
                    "phq9_min": lower, "phq9_max": upper}
                   for i, (lower, upper, label) in enumerate([
                       (0, 4, "Minimal"), (5, 9, "Mild"), (10, 14, "Moderate"),
                       (15, 19, "Moderately severe"), (20, 27, "Severe")])]}
    numerical = inventory.loc[inventory.feature_type.eq("numerical"), "feature"].tolist()
    categorical = inventory.loc[inventory.feature_type.eq("categorical"), "feature"].tolist()
    preprocessor = {
        "numerical": {"columns": numerical, "imputation": "median",
                      "scaling": "StandardScaler(with_mean=True, with_std=True)"},
        "categorical": {"columns": categorical, "imputation": "constant",
                        "fill_value": "missing_or_unknown",
                        "encoding": "OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None)"},
        "learned_only_from": "full development_age20 during Pipeline.fit",
        "structural_missingness": "existing smoking/alcohol routing rules unchanged",
    }
    protected = [FINAL_SPLIT, ARTIFACTS_DIR / "split_indices.json",
                 ARTIFACTS_DIR / "ensemble_candidate.json", ensemble.CONFIG_PATH,
                 ARTIFACTS_DIR / "tuned_model_configs.json", PROCESSED_DATA_DIR / "cohort.csv",
                 RAW_DATA_DIR / "questionnaire.csv", TABLES_DIR / "feature_inventory.csv"]
    protected += [PROJECT_ROOT / "src" / name for name in [
        "final_protocol.py", "07_final_test.py", "05_feature_ablation.py",
        "ablation_features.py", "protocol.py", "config.py", "utils.py"]]
    protected += [TABLES_DIR / name for name in ["ensemble_results.csv", "nested_cv_summary.csv",
                                                "ensemble_per_class_results.csv"]]
    spec = {
        "created_utc": now(), "protocol": PROTOCOL_VERSION, "model": "Random Forest",
        "feature_set": "F4", "features": X.columns.tolist(), "preprocessing": preprocessor,
        "hyperparameters": rf, "target_mapping": mapping,
        "development_n": 4045, "heldout_n": 1009,
        "development_class_counts": {f"C{i}": int((y == i).sum()) for i in CLASSES},
        "selection_rule": candidate["selection_rule"], "user_confirmed_final_candidate": True,
        "selected_before_heldout": True, "heldout_used_for_selection": False,
        "historically_untouched": False,
        "selected_before_heldout_scope": "before the CURRENT supplementary held-out evaluation; "
                                         "does not erase prior age20_v2 development exposure",
        "evaluation_name": EVALUATION_NAME,
        "primary_performance_evidence": "development CV / nested-CV, with prior selection limitations",
        "decision_rule": "argmax probabilities in C0,C1,C2,C3,C4 order; no threshold adjustment",
        "hashes": {"cohort_sha256": sha256_file(PROCESSED_DATA_DIR / "cohort.csv"),
                   "development_ids_sha256": membership_hash(split["development_seqn"]),
                   "development_row_order_sha256": id_hash(X.index),
                   "heldout_ids_sha256": membership_hash(split["test_seqn"]),
                   "feature_schema_sha256": json_hash(schema), "RF_config_sha256": json_hash(rf)},
        "hash_conventions": {"membership": "SHA-256 of compact canonical JSON sorted integer IDs",
                             "schema_and_config": "SHA-256 of compact canonical JSON with sorted keys",
                             "row_order": "tuning_estimators.id_hash; order-sensitive JSON integer list",
                             "files": "SHA-256 of exact bytes"},
        "protected_sha256": {str(path): sha256_file(path) for path in protected},
        "versions": {name: metadata.version(name) for name in ["scikit-learn", "numpy", "pandas", "joblib", "shap"]},
    }
    write_new_json(schema, SCHEMA)
    write_new_json(mapping, MAPPING)
    spec["feature_schema_file_sha256"] = sha256_file(SCHEMA)
    spec["class_mapping_file_sha256"] = sha256_file(MAPPING)
    write_new_json(spec, SPEC)
    return verify_specification()


def verify_specification():
    spec = load_json(SPEC)
    for path, expected in spec["protected_sha256"].items():
        if sha256_file(Path(path)) != expected:
            raise AssertionError(f"Frozen specification input changed: {path}")
    if sha256_file(SCHEMA) != spec["feature_schema_file_sha256"]:
        raise AssertionError("Feature schema changed")
    if sha256_file(MAPPING) != spec["class_mapping_file_sha256"]:
        raise AssertionError("Class mapping changed")
    split = load_final_split()
    for field, ids in [("development_ids_sha256", split["development_seqn"]),
                       ("heldout_ids_sha256", split["test_seqn"])]:
        if spec["hashes"][field] != membership_hash(ids):
            raise AssertionError("Split membership hash changed")
    if json_hash(load_json(SCHEMA)) != spec["hashes"]["feature_schema_sha256"]:
        raise AssertionError("Schema fingerprint changed")
    if json_hash(spec["hyperparameters"]) != spec["hashes"]["RF_config_sha256"]:
        raise AssertionError("RF configuration fingerprint changed")
    return spec


def make_final_pipeline(spec, inventory):
    return Pipeline([
        ("preprocessor", ablation.build_preprocessor(spec["features"], inventory)),
        ("model", RandomForestClassifier(**spec["hyperparameters"])),
    ])


def verify_model_lock():
    spec = verify_specification()
    lock = load_json(MODEL_LOCK)
    if lock["spec_sha256"] != sha256_file(SPEC) or lock["model_sha256"] != sha256_file(MODEL):
        raise AssertionError("Locked specification or fitted model changed")
    if lock["training_n"] != spec["development_n"] or lock["heldout_used_for_fit"]:
        raise AssertionError("Invalid final fit provenance")
    return lock


def load_frozen_model():
    lock = verify_model_lock()
    model = joblib.load(MODEL)  # This project's own verified artifact, not an untrusted pickle.
    if model.named_steps["model"].get_params() != load_json(SPEC)["hyperparameters"]:
        raise AssertionError("Serialized RF parameters differ from specification")
    np.testing.assert_array_equal(model.classes_, CLASSES)
    return model, lock


def verify_evaluation():
    """Read the saved one-time evaluation; never run another held-out prediction."""
    verify_model_lock()
    evaluation = load_json(EVALUATION_LOCK)
    if evaluation["model_sha256"] != sha256_file(MODEL):
        raise AssertionError("Evaluation/model mismatch")
    for path, expected in evaluation["outputs_sha256"].items():
        if sha256_file(Path(path)) != expected:
            raise AssertionError(f"Saved supplementary evaluation changed: {path}")
    return evaluation
