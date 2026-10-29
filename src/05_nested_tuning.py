"""Modest, locked 5x3 nested tuning; final search only after candidate selection.

Only original-membership development_age20 is loaded. No held-out prediction,
new cohort/split, feature changes, ensemble, SMOTE, or full-development refit.
"""

from datetime import datetime, timezone
from importlib import import_module, metadata
import json
from pathlib import Path
import warnings

from joblib import parallel_backend
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ablation_features import prepare_features
from config import (ARTIFACTS_DIR, TABLES_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                    PROTOCOL_VERSION, experiment_path)
from protocol import FINAL_SPLIT, ORIGINAL_SPLIT, load_final_split, sha256_file
from tuning_estimators import FoldBalancedXGBClassifier
from utils import load_json, save_json


ablation = import_module("05_feature_ablation")
N_ITER = 12
SEARCH_JOBS = 2
CANDIDATES = [("Logistic Regression", "F4"), ("Random Forest", "F4"),
              ("XGBoost", "F3"), ("XGBoost", "F4")]
SHORT_NAMES = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB"}
WEIGHT_DIR = ARTIFACTS_DIR / "nested_tuning_weight_audit"
CHECKPOINT_DIR = ARTIFACTS_DIR / "nested_tuning_checkpoints"
MANIFEST = ARTIFACTS_DIR / "nested_tuning_protocol.json"
FOLDS_PATH = ARTIFACTS_DIR / "nested_cv_assignments.json"
SCORER = make_scorer(f1_score, labels=np.arange(5), average="macro", zero_division=0)

# Discrete, modest ranges, locked before looking at any outer result.
SEARCH_SPACES = {
    "Logistic Regression": {
        "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        "model__solver": ["lbfgs", "newton-cg"],
    },
    "Random Forest": {
        "model__n_estimators": [150, 300, 450], "model__max_depth": [None, 8, 16],
        "model__min_samples_split": [2, 10], "model__min_samples_leaf": [1, 3, 8],
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": ["balanced", "balanced_subsample"],
    },
    "XGBoost": {
        "model__n_estimators": [100, 200, 350], "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.1], "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.8, 1.0], "model__colsample_bytree": [0.8, 1.0],
        "model__gamma": [0, 0.2], "model__reg_alpha": [0, 0.1], "model__reg_lambda": [1, 5],
    },
}


def make_pipeline(model_name, features, inventory, context):
    if model_name == "Logistic Regression":
        # sklearn >=1.8 uses l1_ratio=0 to express L2 without deprecated penalty.
        model = LogisticRegression(l1_ratio=0.0, class_weight="balanced", max_iter=3000, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_jobs=1)
    else:
        model = FoldBalancedXGBClassifier(audit_dir=str(WEIGHT_DIR), fit_context=context)
    return Pipeline([("preprocessor", ablation.build_preprocessor(features, inventory)), ("model", model)])


def make_search(pipeline, model_name, folds):
    return RandomizedSearchCV(
        pipeline, param_distributions=SEARCH_SPACES[model_name], n_iter=N_ITER,
        scoring=SCORER, cv=folds, random_state=42, refit=False,
        n_jobs=SEARCH_JOBS, pre_dispatch=SEARCH_JOBS, error_score="raise", return_train_score=False)


def search_rows(search, model, feature_set, outer_fold):
    rows = []
    for index, parameters in enumerate(search.cv_results_["params"]):
        row = {"model": model, "feature_set": feature_set, "outer_fold": outer_fold,
               "candidate_index": index, "params": json.dumps(parameters, sort_keys=True),
               "mean_inner_macro_f1": float(search.cv_results_["mean_test_score"][index]),
               "std_inner_macro_f1": float(search.cv_results_["std_test_score"][index]),
               "rank": int(search.cv_results_["rank_test_score"][index])}
        for fold in range(3):
            row[f"inner_fold_{fold + 1}_macro_f1"] = float(search.cv_results_[f"split{fold}_test_score"][index])
        rows.append(row)
    return rows


def build_inner_folds(y, outer_folds):
    inner = {}
    locked_rows = []
    for outer_number, (train, valid) in enumerate(outer_folds, 1):
        inner[outer_number] = list(StratifiedKFold(3, shuffle=True, random_state=42).split(np.zeros(len(train)), y.iloc[train]))
        assignments = np.full(len(train), -1, dtype=int)
        for inner_number, (inner_train, inner_valid) in enumerate(inner[outer_number], 1):
            assert not set(train[inner_train]) & set(valid)
            assert not set(train[inner_valid]) & set(valid)
            assignments[inner_valid] = inner_number
        locked_rows.append({"outer_fold": outer_number, "outer_train_seqn": y.iloc[train].index.tolist(),
                            "outer_validation_seqn": y.iloc[valid].index.tolist(),
                            "inner_validation_fold": assignments.tolist()})
    ablation.lock_json({"protocol": PROTOCOL_VERSION, "outer_n_splits": 5, "inner_n_splits": 3,
                        "shuffle": True, "random_state": 42, "outer_partitions": locked_rows,
                        "outer_lock_sha256": sha256_file(experiment_path(ARTIFACTS_DIR, "cv_folds.json"))}, FOLDS_PATH)
    return inner


def outer_experiment(X, y, groups, inventory, outer_folds, inner_folds):
    scores, predictions, trials, per_class, matrices = [], [], [], [], []
    for model_name, feature_set in CANDIDATES:
        features = [f for group in ablation.FEATURE_SETS[feature_set] for f in groups[group]]
        candidate_predictions = []
        for number, (train, validation) in enumerate(outer_folds, 1):
            context = f"{SHORT_NAMES[model_name]}_{feature_set}_outer{number}"
            checkpoint = CHECKPOINT_DIR / f"{context}.json"
            if checkpoint.exists():
                payload = load_json(checkpoint)
                if payload["manifest_sha256"] != sha256_file(MANIFEST):
                    raise RuntimeError("Checkpoint belongs to another protocol")
            else:
                print(f"Searching {context}: {N_ITER} candidates x 3 inner folds", flush=True)
                pipeline = make_pipeline(model_name, features, inventory, context + "_inner")
                search = make_search(pipeline, model_name, inner_folds[number])
                search.fit(X.iloc[train][features], y.iloc[train])
                best = clone(pipeline).set_params(**search.best_params_)
                if model_name == "XGBoost":
                    best.set_params(model__fit_context=context + "_refit")
                # The outer validation is not supplied to search or this refit.
                best.fit(X.iloc[train][features], y.iloc[train])
                predicted = best.predict(X.iloc[validation][features]).astype(int)
                metrics = ablation.summarize_metrics(y.iloc[validation], predicted)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y.iloc[validation], predicted, labels=np.arange(5), zero_division=0)
                row = {"model": model_name, "feature_set": feature_set, "outer_fold": number,
                       "n_train": len(train), "n_validation": len(validation),
                       "best_inner_macro_f1": float(search.best_score_),
                       "best_params": json.dumps(search.best_params_, sort_keys=True),
                       **{f"outer_{key}": value for key, value in metrics.items()}}
                for code in range(5):
                    row[f"outer_F1_C{code}"] = float(f1[code])
                    row[f"outer_Recall_C{code}"] = float(recall[code])
                    row[f"outer_support_C{code}"] = int(support[code])
                prediction_rows = [{"model": model_name, "feature_set": feature_set, "outer_fold": number,
                                    "SEQN": int(seqn), "y_true": int(truth), "y_pred": int(pred)}
                                   for seqn, truth, pred in zip(y.iloc[validation].index, y.iloc[validation], predicted)]
                payload = {"manifest_sha256": sha256_file(MANIFEST), "score": row,
                           "predictions": prediction_rows,
                           "inner_trials": search_rows(search, model_name, feature_set, number)}
                save_json(payload, checkpoint)
            scores.append(payload["score"])
            trials.extend(payload["inner_trials"])
            candidate_predictions.extend(payload["predictions"])
            print(f"{context}: inner={payload['score']['best_inner_macro_f1']:.4f}; "
                  f"outer={payload['score']['outer_macro_f1']:.4f}", flush=True)
        oof = pd.DataFrame(candidate_predictions).set_index("SEQN").loc[y.index]
        if not oof.index.is_unique or len(oof) != len(y):
            raise RuntimeError("Invalid nested OOF coverage")
        np.testing.assert_array_equal(oof.y_true, y)
        precision, recall, f1, support = precision_recall_fscore_support(
            oof.y_true, oof.y_pred, labels=np.arange(5), zero_division=0)
        for code in range(5):
            per_class.append({"model": model_name, "feature_set": feature_set, "class": f"C{code}",
                              "f1": f1[code], "recall": recall[code], "precision": precision[code],
                              "support": int(support[code]),
                              "true_positive": int((oof.y_true.eq(code) & oof.y_pred.eq(code)).sum()),
                              "predicted_count": int(oof.y_pred.eq(code).sum()),
                              "aggregation": "pooled_development_nested_oof"})
        matrix = confusion_matrix(oof.y_true, oof.y_pred, labels=np.arange(5))
        matrices.extend({"model": model_name, "feature_set": feature_set, "true_class": f"C{a}",
                         "predicted_class": f"C{b}", "count": int(matrix[a, b])} for a in range(5) for b in range(5))
        predictions.extend(candidate_predictions)
    return pd.DataFrame(scores), pd.DataFrame(predictions), pd.DataFrame(trials), pd.DataFrame(per_class), pd.DataFrame(matrices)


def summarize_and_compare(scores, per_class):
    summary = []
    baseline = pd.read_csv(experiment_path(TABLES_DIR, "feature_ablation_results.csv"))
    for (model, feature_set), rows in scores.groupby(["model", "feature_set"], sort=False):
        row = {"model": model, "feature_set": feature_set}
        for metric in ablation.METRICS:
            row[f"{metric}_mean"] = float(rows[f"outer_{metric}"].mean())
            row[f"{metric}_std"] = float(rows[f"outer_{metric}"].std(ddof=1))
        previous = baseline[baseline.model.eq(model) & baseline.feature_set.eq(feature_set)].iloc[0]
        row["baseline_macro_f1_mean"] = previous.macro_f1_mean
        row["delta_vs_baseline"] = row["macro_f1_mean"] - previous.macro_f1_mean
        summary.append(row)
    comparison = scores[scores.model.eq("XGBoost")].pivot(index="outer_fold", columns="feature_set", values="outer_macro_f1")
    comparison = comparison.rename(columns={"F3": "F3_macro_f1", "F4": "F4_macro_f1"})
    comparison["delta_F4_minus_F3"] = comparison.F4_macro_f1 - comparison.F3_macro_f1
    old_folds = pd.read_csv(experiment_path(TABLES_DIR, "ablation_fold_results.csv"))
    paired = scores.merge(old_folds[["model", "feature_set", "fold", "macro_f1"]],
                          left_on=["model", "feature_set", "outer_fold"], right_on=["model", "feature_set", "fold"],
                          validate="one_to_one")
    paired["delta_tuned_minus_baseline"] = paired.outer_macro_f1 - paired.macro_f1
    old_classes = pd.read_csv(experiment_path(TABLES_DIR, "per_class_cv_results.csv"))
    class_changes = per_class.merge(old_classes[["model", "feature_set", "class", "recall", "f1"]],
                                   on=["model", "feature_set", "class"], suffixes=("_tuned", "_baseline"), validate="one_to_one")
    for metric in ["recall", "f1"]:
        class_changes[f"delta_{metric}"] = class_changes[f"{metric}_tuned"] - class_changes[f"{metric}_baseline"]
    return pd.DataFrame(summary), comparison.reset_index(), paired, class_changes


def final_parameter_search(summary, X, y, groups, inventory):
    # Candidate selection is locked BEFORE the one full-development search.
    ranked = summary.sort_values(["macro_f1_mean", "macro_f1_std", "feature_set", "model"],
                                 ascending=[False, True, True, True])
    winner = ranked.iloc[0]
    selection = {"protocol": PROTOCOL_VERSION, "selection_source": "mean outer Macro-F1 across five folds",
                 "tie_break": "lower outer SD, then feature_set/model lexical order",
                 "model": winner.model, "feature_set": winner.feature_set,
                 "nested_macro_f1_mean": float(winner.macro_f1_mean), "nested_macro_f1_std": float(winner.macro_f1_std),
                 "nested_results_sha256": sha256_file(TABLES_DIR / "nested_cv_tuning_results.csv"),
                 "manifest_sha256": sha256_file(MANIFEST)}
    selection_path = ARTIFACTS_DIR / "nested_cv_selection.json"
    ablation.lock_json(selection, selection_path)
    final_path = ARTIFACTS_DIR / "tuned_model_configs.json"
    if final_path.exists():
        existing = load_json(final_path)
        if existing["selection_sha256"] != sha256_file(selection_path):
            raise RuntimeError("Existing final config belongs to another selection")
        return existing
    features = [f for group in ablation.FEATURE_SETS[winner.feature_set] for f in groups[group]]
    final_folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))
    assignments = np.full(len(y), -1, dtype=int)
    for number, (_, valid) in enumerate(final_folds, 1):
        assignments[valid] = number
    ablation.lock_json({"protocol": PROTOCOL_VERSION, "selected_model": winner.model,
                       "selected_feature_set": winner.feature_set, "selection_sha256": sha256_file(selection_path),
                       "development_seqn": y.index.tolist(), "validation_fold": assignments.tolist(),
                       "n_splits": 3, "shuffle": True, "random_state": 42},
                      ARTIFACTS_DIR / "final_tuning_cv_assignments.json")
    context = f"{SHORT_NAMES[winner.model]}_{winner.feature_set}_full_development_inner"
    pipeline = make_pipeline(winner.model, features, inventory, context)
    search = make_search(pipeline, winner.model, final_folds)
    print(f"Candidate locked: {winner.model}/{winner.feature_set}. Final parameter search (no full-data refit).", flush=True)
    search.fit(X[features], y)
    pd.DataFrame(search_rows(search, winner.model, winner.feature_set, "full_development")).to_csv(
        TABLES_DIR / "final_development_search_results.csv", index=False)
    configured = clone(pipeline).set_params(**search.best_params_).named_steps["model"].get_params()
    configured = {k: v for k, v in configured.items() if k not in {"audit_dir", "fit_context"}}
    output = {"protocol": PROTOCOL_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection_sha256": sha256_file(selection_path), "selected_model": winner.model,
              "selected_feature_set": winner.feature_set, "features": features,
              "best_params": search.best_params_, "estimator_params": configured,
              "xgboost_fixed_params": ({"objective": "multi:softprob", "num_class": 5, "eval_metric": "mlogloss"}
                                       if winner.model == "XGBoost" else None),
              "full_development_best_inner_macro_f1": float(search.best_score_),
              "score_warning": "Parameter-selection CV score, NOT independent model performance",
              "nested_macro_f1_mean": float(winner.macro_f1_mean), "nested_macro_f1_std": float(winner.macro_f1_std),
              "search_space": SEARCH_SPACES[winner.model], "n_iter": N_ITER, "random_state": 42,
              "final_search_refit": False, "full_development_model_fitted": False,
              "heldout_used_for_selection": False, "historically_untouched": False,
              "base_learners_retained": {"LR": "F4 outer-fold configs in best_params_by_fold.csv",
                                         "RF": "F4 outer-fold configs in best_params_by_fold.csv"},
              "weighting": "XGBoost weights recomputed inside fit from that training y only"}
    save_json(output, final_path)
    return output


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    split = load_final_split()
    development, original_inventory = ablation.load_inputs()
    X, groups, inventory, _ = prepare_features(development, original_inventory)
    outer_folds, _ = ablation.lock_folds(development)
    y = development.severity.astype(int).copy()
    y.index = pd.Index(development.SEQN.astype(int), name="SEQN")
    X.index = y.index
    assert not set(y.index) & set(split["test_seqn"])
    inner_folds = build_inner_folds(y, outer_folds)
    paths = [FINAL_SPLIT, ORIGINAL_SPLIT, ARTIFACTS_DIR / "split_indices.json",
             PROCESSED_DATA_DIR / "cohort.csv", ARTIFACTS_DIR / "feature_schema.json",
             RAW_DATA_DIR / "questionnaire.csv", TABLES_DIR / "feature_inventory.csv",
             experiment_path(ARTIFACTS_DIR, "cv_folds.json"), FOLDS_PATH]
    paths += [experiment_path(TABLES_DIR, name) for name in ["feature_ablation_results.csv", "ablation_fold_results.csv",
                                                           "per_class_cv_results.csv", "ablation_feature_inventory.csv"]]
    paths += [Path(__file__), Path(__file__).with_name("tuning_estimators.py"), Path(ablation.__file__),
              Path(__file__).with_name("ablation_features.py"), Path(__file__).with_name("protocol.py"),
              Path(__file__).with_name("config.py")]
    hashes = {str(path): sha256_file(path) for path in paths}
    manifest = {"protocol": PROTOCOL_VERSION, "stage": "nested_tuning", "development_n": len(y),
                "candidates": CANDIDATES, "n_iter": N_ITER, "outer_splits": 5, "inner_splits": 3,
                "random_state": 42, "search_jobs": SEARCH_JOBS, "search_spaces": SEARCH_SPACES,
                "sampled_parameters": {name: list(ParameterSampler(space, n_iter=N_ITER, random_state=42))
                                       for name, space in SEARCH_SPACES.items()},
                "fixed_lr": {"penalty_semantics": "L2", "l1_ratio": 0.0, "class_weight": "balanced", "max_iter": 3000},
                "preprocessing": "unchanged fold-fitted pipeline from final-protocol ablation",
                "primary_metric": "macro_f1", "secondary_metrics": ablation.METRICS[1:],
                "std_ddof": 1, "protected_sha256": hashes,
                "versions": {name: metadata.version(name) for name in ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]},
                "heldout_used": False, "historically_untouched": False,
                "selection_caveat": "Nested CV isolates hyperparameter tuning, not all prior exploratory decisions or selecting the best of multiple outer-CV candidates"}
    ablation.lock_json(manifest, MANIFEST)
    with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=SEARCH_JOBS), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scores, oof, trials, per_class, confusion = outer_experiment(X, y, groups, inventory, outer_folds, inner_folds)
        summary, paired_xgb, baseline_changes, class_changes = summarize_and_compare(scores, per_class)
        tables = {"nested_cv_tuning_results.csv": scores, "nested_cv_summary.csv": summary,
                  "nested_cv_per_class_results.csv": per_class, "nested_cv_oof_predictions.csv": oof,
                  "nested_cv_confusion_matrices.csv": confusion, "nested_cv_inner_search_results.csv": trials,
                  "best_params_by_fold.csv": scores[["model", "feature_set", "outer_fold", "best_inner_macro_f1", "best_params"]],
                  "nested_cv_xgb_F3_F4_comparison.csv": paired_xgb,
                  "nested_cv_baseline_comparison.csv": baseline_changes,
                  "nested_cv_per_class_baseline_comparison.csv": class_changes}
        if hashes != {str(path): sha256_file(path) for path in paths}:
            raise RuntimeError("Protected protocol changed during nested CV")
        for name, table in tables.items():
            table.to_csv(TABLES_DIR / name, index=False)
        print("\nNested CV summary:\n" + summary.to_string(index=False), flush=True)
        final = final_parameter_search(summary, X, y, groups, inventory)
    if hashes != {str(path): sha256_file(path) for path in paths}:
        raise RuntimeError("Protected protocol changed during final parameter search")
    print("\nSelected config:\n" + json.dumps(final, indent=2), flush=True)
    print("No held-out evaluation, ensemble, SHAP, or full-development model refit performed.")


if __name__ == "__main__":
    main()
