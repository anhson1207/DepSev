"""Create reproducible EDA tables and figures for the processed cohort."""

import math
import os
import tempfile
from pathlib import Path

from config import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)

MATPLOTLIB_CACHE_DIR = Path(tempfile.gettempdir()) / "depsev_ml_matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_output_directories, load_json
from ablation_features import read_development_rows


def main():
    """Summarize class balance, missingness, and feature distributions."""
    ensure_output_directories()
    cohort_path = PROCESSED_DATA_DIR / "cohort.csv"
    schema_path = ARTIFACTS_DIR / "feature_schema.json"
    split_path = ARTIFACTS_DIR / "split_indices.json"
    if not all(path.exists() for path in [cohort_path, schema_path, split_path]):
        raise FileNotFoundError("Run src/02_build_cohort.py before EDA.")

    schema = load_json(schema_path)
    split = load_json(split_path)
    if split.get("protocol") == "original_split_preserved_age20":
        raise RuntimeError("Legacy EDA would overwrite exploratory results. Use final-protocol feature inventory and report.")
    development_ids = set(split["development_seqn"])
    analysis_data = read_development_rows(
        cohort_path, development_ids,
        ["SEQN", *[spec["feature"] for spec in schema["features"]], "PHQ9_total", "severity"],
    )
    if len(analysis_data) != len(development_ids):
        raise RuntimeError("Development split IDs do not match the processed cohort.")

    feature_specs = schema["features"]
    feature_columns = [spec["feature"] for spec in feature_specs]
    numerical_features = [
        spec["feature"]
        for spec in feature_specs
        if spec["feature_type"] == "numerical"
    ]
    categorical_features = [
        spec["feature"]
        for spec in feature_specs
        if spec["feature_type"] == "categorical"
    ]

    class_distribution = pd.read_csv(TABLES_DIR / "class_distribution.csv")

    missingness = pd.DataFrame(
        {
            "feature": feature_columns,
            "missing_count": analysis_data[feature_columns].isna().sum().values,
            "missing_percent": (
                analysis_data[feature_columns].isna().mean().values * 100
            ).round(4),
        }
    ).sort_values("missing_percent", ascending=False)
    missingness.to_csv(TABLES_DIR / "missingness.csv", index=False)

    descriptive_columns = [*numerical_features, "PHQ9_total"]
    descriptive_statistics = analysis_data[descriptive_columns].describe().T
    descriptive_statistics.index.name = "feature"
    descriptive_statistics.to_csv(TABLES_DIR / "descriptive_statistics.csv")

    categorical_rows = []
    for feature in categorical_features:
        counts = analysis_data[feature].value_counts(dropna=False)
        for value, count in counts.items():
            categorical_rows.append(
                {
                    "feature": feature,
                    "value": "Missing" if pd.isna(value) else value,
                    "count": int(count),
                    "percent": round(count / len(analysis_data) * 100, 4),
                }
            )
    pd.DataFrame(categorical_rows).to_csv(
        TABLES_DIR / "categorical_value_counts.csv", index=False
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(class_distribution["class"], class_distribution["count"])
    axis.bar_label(bars)
    axis.set_title("PHQ-9 symptom severity class distribution")
    axis.set_xlabel("Severity class")
    axis.set_ylabel("Participants")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    fig_height = max(6, len(missingness) * 0.28)
    fig, axis = plt.subplots(figsize=(9, fig_height))
    axis.barh(missingness["feature"], missingness["missing_percent"])
    axis.invert_yaxis()
    axis.set_title("Predictor missingness")
    axis.set_xlabel("Missing (%)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missingness.png", dpi=150)
    plt.close(fig)

    plot_columns = [*numerical_features, "PHQ9_total"]
    n_columns = 3
    n_rows = math.ceil(len(plot_columns) / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 3.5 * n_rows))
    for axis, feature in zip(axes.flat, plot_columns):
        axis.hist(analysis_data[feature].dropna(), bins=30)
        axis.set_title(feature)
        axis.set_ylabel("Count")
    for axis in axes.flat[len(plot_columns) :]:
        axis.set_visible(False)
    fig.suptitle("Numerical feature distributions", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numerical_distributions.png", dpi=150)
    plt.close(fig)

    correlation = analysis_data[numerical_features].corr()
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=90)
    axis.set_yticks(range(len(correlation.index)))
    axis.set_yticklabels(correlation.index)
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Numerical predictor correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Cohort N (split metadata): {len(development_ids) + len(split['test_seqn'])}")
    print(f"Development EDA shape: {analysis_data.shape}")
    print("Pre-split class distribution:")
    print(class_distribution.to_string(index=False))
    print("\nHighest predictor missingness:")
    print(missingness.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
