"""Frozen common31 experiment primitives, with no legacy experiment execution."""

import hashlib
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/multicycle"
OUT = ROOT / "results/multicycle"
TABLES, FIGURES = OUT / "tables", OUT / "figures"
DATA = ROOT / "data/processed_multicycle"
CYCLES = ["2011_2012", "2013_2014", "2015_2016"]
SEEDS = list(range(101, 111))
RF = dict(n_estimators=150, max_depth=8, min_samples_split=10, min_samples_leaf=8,
          max_features=0.5, class_weight="balanced_subsample", random_state=42, n_jobs=1)
TASKS = {
    "T2": {"cutoffs": [5], "ranges": ["0-4", "5-27"], "labels": ["minimal symptoms", "symptomatic"]},
    "T3": {"cutoffs": [5, 10], "ranges": ["0-4", "5-9", "10-27"], "labels": ["minimal", "mild", "moderate or higher"]},
    "T4": {"cutoffs": [5, 10, 15], "ranges": ["0-4", "5-9", "10-14", "15-27"], "labels": ["minimal", "mild", "moderate", "moderately severe or severe"]},
    "T5": {"cutoffs": [5, 10, 15, 20], "ranges": ["0-4", "5-9", "10-14", "15-19", "20-27"], "labels": ["minimal", "mild", "moderate", "moderately severe", "severe"]},
}
METRICS = ["macro_f1", "balanced_accuracy", "macro_precision", "macro_recall",
           "weighted_f1", "accuracy", "mean_absolute_class_error", "within_one_class_accuracy"]
CODE = ["src/multicycle_protocol.py", "src/multicycle_outputs.py",
        "src/13_multicycle_controlled_experiment.py", "src/14_temporal_generalization.py",
        "tests/test_multicycle.py"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def read_json(path):
    return json.loads(path.read_text())


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if not (ROOT / name).is_file() or digest(ROOT / name) != expected:
            raise AssertionError(f"Protected file changed: {name}")


def preserve_legacy():
    path = ART / "legacy_preservation.json"
    if path.exists():
        saved = read_json(path)
    else:
        files = []
        for directory in ["data", "artifacts", "results", "src", "tests"]:
            files.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.name != ".DS_Store"
                         and not p.is_relative_to(ART) and not p.is_relative_to(OUT)
                         and str(p.relative_to(ROOT)) not in CODE)
        files.extend(ROOT / name for name in ["README.md", "requirements.txt", ".gitignore", "final_report.md"] if (ROOT / name).exists())
        saved = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}
        write_json(path, saved)
    verify_hashes(saved)
    return len(saved)


def target_values(scores, task):
    values = np.asarray(scores)
    if not np.isfinite(values).all() or not np.equal(values, np.round(values)).all() or not ((values >= 0) & (values <= 27)).all():
        raise ValueError("PHQ9 totals must be integers in 0..27")
    return np.searchsorted(TASKS[task]["cutoffs"], values, side="right")


def participant_keys(frame):
    return frame.cycle.astype(str) + ":" + frame.SEQN.astype(str)


def membership_hash(frame):
    return hashlib.sha256("\n".join(participant_keys(frame)).encode()).hexdigest()


def load_inputs():
    """Read policy typing explicitly, and verify all four existing cohorts."""
    policy = read_json(ROOT / "artifacts/common31_harmonization_policy.json")
    build = read_json(ROOT / "artifacts/common31_build_audit.json")
    verify_hashes(build["output_and_code_sha256"])
    if digest(ROOT / "artifacts/common31_harmonization_policy.json") != build["policy_sha256"]:
        raise AssertionError("Harmonization policy changed")
    schema = policy["columns"]
    features = policy["ordered_features"]
    if len(features) != 31 or features != [row["feature"] for row in schema]:
        raise AssertionError("Expected the frozen common31 schema")
    excluded = set(policy["metadata_not_predictors"]) | set(policy["excluded_features"]) | {"SLD012"}
    if set(features) & excluded or any(name.startswith("DPQ") for name in features):
        raise AssertionError("Leakage or excluded predictor in schema")
    dtype = {row["feature"]: "object" if row["feature_type"] == "categorical" else "float64" for row in schema}
    frames = {}
    for cycle in [*CYCLES, "2011_2016_common31"]:
        frame = pd.read_csv(DATA / f"cohort_{cycle}.csv", dtype=dtype, float_precision="round_trip")
        if frame[["cycle", "SEQN"]].isna().any().any() or frame.duplicated(["cycle", "SEQN"]).any():
            raise AssertionError("Duplicate or missing participant-within-cycle")
        if not frame.RIDAGEYR.ge(20).all():
            raise AssertionError("Nonadult participant")
        phq = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
        if not frame[phq].isin([0, 1, 2, 3]).all().all():
            raise AssertionError("Invalid PHQ items")
        np.testing.assert_array_equal(frame.PHQ9_total, frame[phq].sum(axis=1))
        np.testing.assert_array_equal(frame.severity, target_values(frame.PHQ9_total, "T5"))
        if cycle in CYCLES and not frame.cycle.eq(cycle).all():
            raise AssertionError("Cycle provenance mismatch")
        frames[cycle] = frame.sort_values(["cycle", "SEQN"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frames["2011_2016_common31"], pd.concat([frames[c] for c in CYCLES], ignore_index=True))
    if len(frames["2013_2014"]) != 5054 or len(frames["2011_2016_common31"]) != 14581:
        raise AssertionError("Unexpected A/B sizes")
    old = read_json(ROOT / "artifacts/final_model_spec.json")
    if any(old["hyperparameters"][key] != value for key, value in RF.items()):
        raise AssertionError("RF differs from previously frozen configuration")
    expected = old["preprocessing"]
    assert expected["numerical"]["imputation"] == "median"
    assert expected["numerical"]["scaling"] == "StandardScaler(with_mean=True, with_std=True)"
    assert expected["categorical"]["imputation"] == "constant"
    assert expected["categorical"]["fill_value"] == "missing_or_unknown"
    for row in schema:
        original = "alcohol_quantity_code" if row["feature"] == "alcohol_quantity_code_common" else row["feature"]
        assert original in expected[row["feature_type"]]["columns"]
    return frames, policy, old


def predictor_frame(frame, policy):
    X = frame.loc[:, policy["ordered_features"]].copy()
    for row in policy["columns"]:
        if row["feature_type"] == "categorical":
            X[row["feature"]] = X[row["feature"]].astype(object).where(X[row["feature"]].notna(), np.nan)
    assert X.shape[1] == 31
    assert not set(X) & {"cycle", "SEQN", "PHQ9_total", "severity", "severity_class", "alcohol_quantity_state", "alcohol_quantity_reason"}
    assert not any(column.startswith("DPQ") for column in X)
    return X


def make_pipeline(policy, old):
    """Same preprocessing as the legacy pipeline; no fitted object is reused."""
    numerical = [r["feature"] for r in policy["columns"] if r["feature_type"] == "numerical"]
    categorical = [r["feature"] for r in policy["columns"] if r["feature_type"] == "categorical"]
    preprocessor = ColumnTransformer([
        ("numerical", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing_or_unknown")),
                                   ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(**old["hyperparameters"]))])


def fixed_folds(frame):
    assignments = np.zeros(len(frame), dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(frame)), frame.severity), 1):
        assignments[valid] = fold
    assert set(assignments) == {1, 2, 3, 4, 5}
    return assignments


def matched_sample(pool, counts, seed):
    """Exact T5 counts, proportional largest-remainder cycle allocation, no replacement.

    This is cohort sampling for the requested control, NOT class balancing of
    training folds. A participant can recur across different sampling seeds.
    """
    if pool.cycle.eq("2013_2014").any():
        raise AssertionError("Matched-N pool contains control-cycle participants")
    rng, chosen = np.random.default_rng(seed), []
    for label, total in enumerate(counts):
        group = pool.loc[pool.severity.eq(label)]
        cycles = sorted(group.cycle.unique())
        available = np.asarray([group.cycle.eq(c).sum() for c in cycles])
        if total > available.sum():
            raise ValueError("Insufficient class members; replacement is not allowed")
        quotas = total * available / available.sum()
        allocations = np.floor(quotas).astype(int)
        order = np.argsort(-(quotas - allocations), kind="stable")
        allocations[order[:int(total - allocations.sum())]] += 1
        assert (allocations <= available).all() and (allocations > 0).all()
        for cycle, amount in zip(cycles, allocations):
            candidates = group.loc[group.cycle.eq(cycle)].sort_values("SEQN").index.to_numpy()
            chosen.extend(rng.choice(candidates, size=int(amount), replace=False).tolist())
    sample = pool.loc[chosen].sort_values(["cycle", "SEQN"]).reset_index(drop=True)
    np.testing.assert_array_equal(sample.severity.value_counts().sort_index(), counts)
    assert not sample.duplicated(["cycle", "SEQN"]).any()
    return sample


def freeze_config(frames, policy, old):
    preserve_legacy()
    inputs = [DATA / f"cohort_{c}.csv" for c in [*CYCLES, "2011_2016_common31"]]
    design = {
        "experiment": "controlled_official_common31_multicycle", "feature_set": "F4_common31",
        "features": policy["ordered_features"], "feature_typing": {r["feature"]: r["feature_type"] for r in policy["columns"]},
        "targets": TASKS, "RF": old["hyperparameters"],
        "preprocessing": {"numerical": "median -> StandardScaler", "categorical": "constant missing_or_unknown -> OneHotEncoder(handle_unknown=ignore,sparse_output=False,drop=None)",
                          "fit_scope": "fresh Pipeline fitted only on each training partition"},
        "n_splits": 5, "fold_seed": 42, "shuffle": True,
        "fold_stratification": "T5 once per dataset/sample; same assignments reused across T2-T5",
        "row_order": "cycle then SEQN ascending", "participant_key": "cycle + ':' + SEQN",
        "global_seqn_unique": bool(frames["2011_2016_common31"].SEQN.is_unique),
        "matched_sampling_seeds": SEEDS,
        "matched_sampling": "without replacement within sample; exact A T5 counts; cycle quotas proportional to class availability, largest-remainder allocation; seeds may overlap",
        "matched_counts": frames["2013_2014"].severity.value_counts().sort_index().tolist(),
        "matched_counts_user_reference": [3764, 800, 303, 132, 55],
        "A_n": 5054, "B_n": 14581, "C_n": 5054,
        "D": {"train_cycles": CYCLES[:2], "test_cycle": CYCLES[2], "train_n": 9693, "test_n": 4888},
        "metrics": METRICS, "primary_metric": "macro_f1", "decision": "argmax class probabilities, no adjusted thresholds",
        "aggregation": {"A_B": "mean and sample SD (ddof=1) of 5 folds; OOF scores reported separately",
                        "C": "mean and sample SD across 10 sample-level CV means, NOT independent 50-fold inference",
                        "per_class": "pooled OOF for A/B; mean and SD of 10 sample-level OOF results for C; test for D",
                        "D": "single temporal test, no fold SD or confidence interval"},
        "no_paired_A_B_significance_test": True, "no_tuning": True,
        "survey_weights": "not used; descriptive/predictive cohort analysis, not population estimates",
        "input_dataset_hashes": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "policy_sha256": digest(ROOT / "artifacts/common31_harmonization_policy.json"),
        "legacy_spec_sha256": digest(ROOT / "artifacts/final_model_spec.json"),
        "script_hashes": {p: digest(ROOT / p) for p in CODE},
        "versions": {p: metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "matplotlib"]},
    }
    path = ART / "experiment_config.json"
    if path.exists():
        saved = read_json(path)
        if {k: v for k, v in saved.items() if k != "timestamp"} != design:
            raise AssertionError("Frozen experiment configuration differs; refusing to change")
    else:
        write_json(path, {"timestamp": datetime.now(timezone.utc).isoformat(), **design})
    return read_json(path)


def score(y, pred, task):
    k = len(TASKS[task]["ranges"])
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(k), zero_division=0)
    matrix = confusion_matrix(y, pred, labels=np.arange(k))
    error = np.abs(np.asarray(y) - np.asarray(pred))
    result = {"macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
              "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
              "weighted_f1": float(np.average(f1, weights=support)), "accuracy": float(accuracy_score(y, pred)),
              "mean_absolute_class_error": float(error.mean()) if k > 2 else np.nan,
              "within_one_class_accuracy": float((error <= 1).mean()) if k > 2 else np.nan}
    classes = [{"class_index": i, "class": f"C{i}" if task == "T5" else f"{task}_{i}",
                "label": TASKS[task]["labels"][i], "phq9_range": TASKS[task]["ranges"][i],
                "precision": precision[i], "recall": recall[i], "f1": f1[i],
                "support": int(support[i]), "predicted_count": int(matrix[:, i].sum()), "true_positives": int(matrix[i, i])} for i in range(k)]
    confusion = [{"true_class": i, "predicted_class": j, "count": int(matrix[i, j]),
                  "row_proportion": float(matrix[i, j] / support[i]) if support[i] else np.nan} for i in range(k) for j in range(k)]
    return result, classes, confusion


def fit_audit(pipeline, train, validation, policy):
    """Verify learned preprocessing statistics against only the training inputs."""
    assert not set(participant_keys(train)) & set(participant_keys(validation))
    X = predictor_frame(train, policy)
    pre = pipeline.named_steps["preprocessor"]
    numerical = [r["feature"] for r in policy["columns"] if r["feature_type"] == "numerical"]
    categorical = [r["feature"] for r in policy["columns"] if r["feature_type"] == "categorical"]
    num, cat = pre.named_transformers_["numerical"], pre.named_transformers_["categorical"]
    medians = X[numerical].median()
    assert medians.notna().all(), "All-empty numerical feature needs explicit protocol review"
    np.testing.assert_allclose(num["imputer"].statistics_, medians)
    filled = X[numerical].fillna(medians).to_numpy()
    np.testing.assert_allclose(num["scaler"].mean_, filled.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(num["scaler"].var_, filled.var(axis=0), rtol=1e-12, atol=1e-12)
    assert num["scaler"].n_samples_seen_ == len(train)
    categories = {}
    for column, learned in zip(categorical, cat["encoder"].categories_):
        expected = sorted(X[column].fillna("missing_or_unknown").unique())
        assert list(learned) == expected
        categories[column] = list(learned)
    return {"train_n": len(train), "validation_n": len(validation),
            "train_membership_hash": membership_hash(train), "validation_membership_hash": membership_hash(validation),
            "numerical_columns": numerical, "imputer_medians": medians.tolist(),
            "scaler_mean": num["scaler"].mean_.tolist(), "scaler_variance": num["scaler"].var_.tolist(),
            "scaler_n_samples_seen": int(num["scaler"].n_samples_seen_), "encoder_categories": categories,
            "RF_config": pipeline.named_steps["model"].get_params(), "disjoint": True}


def run_task(experiment, frame, task, policy, old, seed=0, train_frame=None):
    """One fixed-CV task, or one D temporal fit; checkpoint exact completed tasks."""
    name = f"{experiment}_{seed}_{task}"
    checkpoint = ART / "checkpoints" / f"{name}.json"
    predictions_path = ART / "checkpoints" / f"{name}_predictions.csv"
    config_hash = digest(ART / "experiment_config.json")
    if checkpoint.exists():
        saved = read_json(checkpoint)
        assert saved["config_sha256"] == config_hash
        assert saved["predictions_sha256"] == digest(predictions_path)
        print(f"REUSE verified checkpoint {name}", flush=True)
        return saved, pd.read_csv(predictions_path, float_precision="round_trip")
    if predictions_path.exists():
        raise FileExistsError(f"Incomplete checkpoint requires review: {name}")
    X, y = predictor_frame(frame, policy), target_values(frame.PHQ9_total, task)
    k = len(TASKS[task]["ranges"])
    assignments = fixed_folds(frame) if train_frame is None else np.zeros(len(frame), dtype=int)
    probabilities = np.full((len(frame), k), np.nan)
    folds, audits = [], []
    for fold in (range(1, 6) if train_frame is None else [0]):
        valid_indices = np.flatnonzero(assignments == fold)
        train = frame.loc[assignments != fold] if train_frame is None else train_frame
        valid = frame.iloc[valid_indices]
        pipeline = make_pipeline(policy, old)
        pipeline.fit(predictor_frame(train, policy), target_values(train.PHQ9_total, task))
        audit = fit_audit(pipeline, train, valid, policy)
        np.testing.assert_array_equal(pipeline.classes_, np.arange(k))
        probabilities[valid_indices] = pipeline.predict_proba(X.iloc[valid_indices])
        pred = probabilities[valid_indices].argmax(axis=1)
        metrics, _, _ = score(y[valid_indices], pred, task)
        folds.append({"experiment": experiment, "sampling_seed": seed, "target": task, "fold": fold, **metrics})
        audits.append({"experiment": experiment, "sampling_seed": seed, "target": task, "fold": fold, **audit})
        print(f"{name} fold={fold} n_train={len(train)} n_valid={len(valid)} Macro-F1={metrics['macro_f1']:.4f}", flush=True)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    pred = probabilities.argmax(axis=1)
    oof, classes, confusion = score(y, pred, task)
    rows = frame[["SEQN", "cycle"]].copy()
    rows["experiment"], rows["sampling_seed"], rows["target"] = experiment, seed, task
    rows["fold"], rows["y_true"], rows["y_pred"] = assignments, y, pred
    for i in range(k):
        rows[f"prob_{i}"] = probabilities[:, i]
    if train_frame is not None:
        rows["train_cycles"], rows["test_cycle"] = ";".join(CYCLES[:2]), CYCLES[2]
    summary = {"experiment": experiment, "sampling_seed": seed, "target": task, "n": len(frame),
               "evaluation": "temporal_test" if train_frame is not None else "fixed_5fold_CV"}
    for metric in METRICS:
        values = pd.Series([row[metric] for row in folds], dtype=float)
        summary[f"{metric}_mean"] = values.mean()
        summary[f"{metric}_std"] = values.std(ddof=1) if train_frame is None else np.nan
        summary[f"{metric}_oof_or_test"] = oof[metric]
    context = {"experiment": experiment, "sampling_seed": seed, "target": task,
               "scope": "temporal_test" if train_frame is not None else "pooled_OOF"}
    write_csv(predictions_path, rows)
    saved = {"config_sha256": config_hash, "predictions_sha256": digest(predictions_path),
             "summary": summary, "folds": folds, "fit_audits": audits,
             "per_class": [{**context, **row} for row in classes],
             "confusion": [{**context, **row} for row in confusion]}
    write_json(checkpoint, saved)
    return saved, rows
