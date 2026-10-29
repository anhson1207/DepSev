"""Legacy F1 baseline; use 05_feature_ablation.py for the corrected age20 protocol."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (
    ARTIFACTS_DIR,
    N_CV_SPLITS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TABLES_DIR,
)
from utils import ensure_output_directories, load_json


def build_preprocessor(numerical_features, categorical_features):
    """Build fold-fitted numerical and categorical preprocessing."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def main():
    """Run five-fold CV on the locked development subset using F1=P only."""
    ensure_output_directories()
    cohort_path = PROCESSED_DATA_DIR / "cohort.csv"
    split_path = ARTIFACTS_DIR / "split_indices.json"
    schema_path = ARTIFACTS_DIR / "feature_schema.json"
    if not all(path.exists() for path in [cohort_path, split_path, schema_path]):
        raise FileNotFoundError("Run src/02_build_cohort.py before baseline training.")

    split = load_json(split_path)
    if split.get("protocol_version") in {"age20_v2", "original_split_preserved_age20"}:
        raise RuntimeError(
            "Legacy baseline is retired for age20_v2. Run src/05_feature_ablation.py "
            "for the locked folds and fold-balanced XGBoost F1 results."
        )
    cohort = pd.read_csv(cohort_path)
    schema = load_json(schema_path)
    physiological_features = schema["groups"]["P"]
    physiological_specs = {
        spec["feature"]: spec for spec in schema["features"] if spec["group"] == "P"
    }
    numerical_features = [
        feature
        for feature in physiological_features
        if physiological_specs[feature]["feature_type"] == "numerical"
    ]
    categorical_features = [
        feature
        for feature in physiological_features
        if physiological_specs[feature]["feature_type"] == "categorical"
    ]

    development_ids = set(split["development_seqn"])
    development = cohort[cohort["SEQN"].isin(development_ids)].copy()
    if len(development) != len(development_ids):
        raise RuntimeError("Development split IDs do not match the processed cohort.")

    X_development = development[physiological_features]
    y_development = development["severity"]

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
        "macro_precision": make_scorer(
            precision_score, average="macro", zero_division=0
        ),
        "macro_recall": make_scorer(
            recall_score, average="macro", zero_division=0
        ),
        "weighted_f1": make_scorer(
            f1_score, average="weighted", zero_division=0
        ),
    }
    cross_validator = StratifiedKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    shared_folds = list(cross_validator.split(X_development, y_development))

    result_rows = []
    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numerical_features, categorical_features),
                ),
                ("model", model),
            ]
        )
        scores = cross_validate(
            pipeline,
            X_development,
            y_development,
            cv=shared_folds,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        row = {"model": model_name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            if metric == "macro_f1":
                row["macro_f1_std"] = float(np.std(values, ddof=1))
        result_rows.append(row)
        print(
            f"{model_name}: Macro-F1 "
            f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
        )

    results = pd.DataFrame(result_rows)[
        [
            "model",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "macro_precision_mean",
            "macro_recall_mean",
            "weighted_f1_mean",
        ]
    ].sort_values("macro_f1_mean", ascending=False)
    results.to_csv(TABLES_DIR / "baseline_F1_results.csv", index=False)

    print("\nF1=P physiological features:")
    print(physiological_features)
    print("\nDevelopment-only 5-fold CV results:")
    print(results.to_string(index=False))
    print("\nHeld-out test set was not evaluated.")


if __name__ == "__main__":
    main()
