"""Build a one-row-per-participant cohort and leakage-safe PHQ-9 target."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ARTIFACTS_DIR,
    FEATURE_SPECS,
    LEAKAGE_COLUMNS,
    PHQ9_COLUMNS,
    PRIMARY_MIN_AGE,
    PROTOCOL_VERSION,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    SEVERITY_LABELS,
    TABLES_DIR,
    TEST_SIZE,
)
from utils import ensure_output_directories, load_json, read_csv_with_fallback, save_json


def find_column_sources(csv_files):
    """Map each observed column to the raw files that contain it."""
    sources = defaultdict(list)
    for csv_file in csv_files:
        header, _ = read_csv_with_fallback(csv_file, nrows=0)
        for column in header.columns:
            sources[column].append(csv_file)
    return sources


def find_single_source(columns, column_sources):
    """Return the one file containing every requested source column."""
    candidate_sets = [set(column_sources.get(column, [])) for column in columns]
    common_sources = set.intersection(*candidate_sets) if candidate_sets else set()
    if len(common_sources) != 1:
        return None
    return next(iter(common_sources))


def audit_seqn_cardinality(csv_files):
    """Report one-to-many files before any feature merge is attempted."""
    one_to_many_files = []
    for csv_file in csv_files:
        data, _ = read_csv_with_fallback(
            csv_file, usecols=["SEQN"], low_memory=False
        )
        duplicate_ids = data.loc[
            data["SEQN"].duplicated(keep=False), "SEQN"
        ].nunique()
        if duplicate_ids:
            one_to_many_files.append(csv_file.name)
            print(
                f"One-to-many file excluded without an aggregation rule: "
                f"{csv_file.name} ({duplicate_ids} duplicated SEQN values)"
            )
    return one_to_many_files


def main():
    """Create the cohort, feature inventory, class table, and locked split."""
    if PROTOCOL_VERSION == "original_split_preserved_age20":
        raise RuntimeError("Random re-splitting is retired. Use src/02_restore_original_split.py.")
    ensure_output_directories()
    previous_split_path = ARTIFACTS_DIR / "archive/pre_age20_v1/artifacts/split_indices.json"
    if not previous_split_path.exists():
        raise RuntimeError("Run archive_protocol.py before applying the age20 correction.")
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    audit_seqn_cardinality(csv_files)
    column_sources = find_column_sources(csv_files)
    missing_phq9 = [column for column in PHQ9_COLUMNS if not column_sources[column]]
    if missing_phq9:
        raise RuntimeError(f"Missing PHQ-9 columns: {missing_phq9}")

    phq_source = find_single_source(PHQ9_COLUMNS, column_sources)
    if phq_source is None:
        raise RuntimeError("The nine PHQ-9 items are not located together in one file.")

    questionnaire, _ = read_csv_with_fallback(phq_source, low_memory=False)
    if questionnaire["SEQN"].duplicated().any():
        raise RuntimeError(f"Duplicate SEQN found in target file: {phq_source.name}")

    clean_phq9 = questionnaire[PHQ9_COLUMNS].where(
        questionnaire[PHQ9_COLUMNS].isin([0, 1, 2, 3])
    )
    valid_phq9 = clean_phq9.notna().all(axis=1)
    target = questionnaire.loc[valid_phq9, ["SEQN"]].copy()
    target["PHQ9_total"] = clean_phq9.loc[valid_phq9].sum(axis=1).astype(int)
    target["severity"] = pd.cut(
        target["PHQ9_total"],
        bins=[-1, 4, 9, 14, 19, 27],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # Eligibility is fixed by protocol, before splitting or inspecting performance.
    age_source = find_single_source(["RIDAGEYR"], column_sources)
    if age_source is None:
        raise RuntimeError("Cannot identify the real age source for eligibility")
    ages, _ = read_csv_with_fallback(age_source, usecols=["SEQN", "RIDAGEYR"])
    if ages.SEQN.isna().any() or ages.SEQN.duplicated().any():
        raise RuntimeError("Missing/duplicate IDs in the age source")
    eligible_age = target.SEQN.map(ages.set_index("SEQN").RIDAGEYR)
    valid_phq9_before_age = len(target)
    excluded_under20 = int(eligible_age.lt(PRIMARY_MIN_AGE).sum())
    excluded_missing_age = int(eligible_age.isna().sum())
    target = target.loc[eligible_age.ge(PRIMARY_MIN_AGE)].copy()
    if target.empty:
        raise RuntimeError("No eligible age20 participants with complete PHQ-9")

    class_counts = target["severity"].value_counts().reindex(range(5), fill_value=0)
    class_distribution = pd.DataFrame(
        {
            "class": [f"C{code}" for code in range(5)],
            "severity": range(5),
            "label": [SEVERITY_LABELS[code] for code in range(5)],
            "count": class_counts.values,
            "percent": (class_counts.values / len(target) * 100).round(4),
        }
    )
    class_distribution.to_csv(TABLES_DIR / "class_distribution.csv", index=False)

    print(f"PHQ-9 source: {phq_source.name}")
    print(f"Total participants: {len(questionnaire)}")
    print(f"Participants with valid 9-item PHQ-9 before age eligibility: {valid_phq9_before_age}")
    print(f"Excluded for age < {PRIMARY_MIN_AGE}: {excluded_under20}")
    print(f"Excluded for missing age: {excluded_missing_age}")
    print(f"Eligible age20 cohort: {len(target)}")
    for row in class_distribution.to_dict(orient="records"):
        print(
            f"{row['class']} count: {row['count']} "
            f"({row['percent']:.2f}%)"
        )

    feature_sources = {}
    unavailable_features = []
    required_columns_by_file = defaultdict(set)

    for spec in FEATURE_SPECS:
        source_columns = spec.get("source_columns", [spec["feature"]])
        source_file = find_single_source(source_columns, column_sources)
        if source_file is None:
            unavailable_features.append(spec["feature"])
            continue
        feature_sources[spec["feature"]] = source_file
        required_columns_by_file[source_file].update(source_columns)

    missing_physiological = [
        spec["feature"]
        for spec in FEATURE_SPECS
        if spec["group"] == "P" and spec["feature"] in unavailable_features
    ]
    if missing_physiological:
        raise RuntimeError(
            f"Required physiological features are unavailable: {missing_physiological}"
        )

    cohort = target.copy()
    for source_file, required_columns in required_columns_by_file.items():
        usecols = ["SEQN", *sorted(required_columns)]
        source_data, _ = read_csv_with_fallback(
            source_file, usecols=usecols, low_memory=False
        )
        duplicate_rows = int(source_data["SEQN"].duplicated(keep=False).sum())
        if duplicate_rows:
            raise RuntimeError(
                f"Unsafe one-to-many source {source_file.name}: "
                f"{duplicate_rows} rows have duplicate SEQN"
            )
        cohort = cohort.merge(
            source_data,
            on="SEQN",
            how="left",
            validate="one_to_one",
        )

    for spec in FEATURE_SPECS:
        source_columns = spec.get("source_columns")
        if not source_columns or spec["feature"] in unavailable_features:
            continue
        cohort[spec["feature"]] = cohort[source_columns].mean(axis=1, skipna=True)

    derived_source_columns = {
        column
        for spec in FEATURE_SPECS
        for column in spec.get("source_columns", [])
    }
    cohort = cohort.drop(columns=list(derived_source_columns), errors="ignore")

    available_specs = [
        spec for spec in FEATURE_SPECS if spec["feature"] not in unavailable_features
    ]
    predictor_columns = [spec["feature"] for spec in available_specs]
    leaked_predictors = sorted(set(predictor_columns) & LEAKAGE_COLUMNS)
    if leaked_predictors:
        raise RuntimeError(f"Leakage columns found in predictors: {leaked_predictors}")

    cohort = cohort[["SEQN", *predictor_columns, "PHQ9_total", "severity"]]
    if cohort.SEQN.isna().any() or cohort["SEQN"].duplicated().any():
        raise RuntimeError("Cohort construction created duplicate participants.")
    if not cohort.RIDAGEYR.ge(PRIMARY_MIN_AGE).all():
        raise RuntimeError("Primary cohort violates age eligibility")
    cohort.to_csv(PROCESSED_DATA_DIR / "cohort.csv", index=False)

    inventory_rows = []
    for spec in FEATURE_SPECS:
        feature = spec["feature"]
        if feature in unavailable_features:
            inventory_rows.append(
                {
                    **spec,
                    "source_file": "not available",
                    "dtype": "not available",
                    "missing_count": pd.NA,
                    "missing_percent": pd.NA,
                    "unique_values": pd.NA,
                }
            )
            continue
        values = cohort[feature]
        inventory_rows.append(
            {
                **spec,
                "source_file": feature_sources[feature].name,
                "dtype": str(values.dtype),
                "missing_count": int(values.isna().sum()),
                "missing_percent": round(values.isna().mean() * 100, 4),
                "unique_values": int(values.nunique(dropna=True)),
            }
        )

    feature_inventory = pd.DataFrame(inventory_rows)
    feature_inventory = feature_inventory.drop(columns=["source_columns"], errors="ignore")
    feature_inventory.to_csv(TABLES_DIR / "feature_inventory.csv", index=False)
    feature_inventory[["feature", "group", "missing_count", "missing_percent"]].to_csv(
        TABLES_DIR / "cohort_missingness.csv", index=False)

    feature_schema = {
        "features": available_specs,
        "groups": {
            group: [
                spec["feature"] for spec in available_specs if spec["group"] == group
            ]
            for group in ["P", "C", "L", "D"]
        },
        "excluded_for_leakage": sorted(LEAKAGE_COLUMNS),
    }
    save_json(feature_schema, ARTIFACTS_DIR / "feature_schema.json")

    development_seqn, test_seqn = train_test_split(
        cohort["SEQN"],
        test_size=TEST_SIZE,
        stratify=cohort["severity"],
        random_state=RANDOM_STATE,
    )
    previous_split = load_json(previous_split_path)
    old_dev = set(previous_split["development_seqn"])
    old_test = set(previous_split["test_seqn"])
    role_migration = {
        "old_development_to_new_test": len(old_dev & set(test_seqn)),
        "old_test_to_new_development": len(old_test & set(development_seqn)),
        "old_test_retained_in_new_test": len(old_test & set(test_seqn)),
        "note": "ID overlap only; no old/new held-out performance evaluated. New test includes previously analyzed development participants.",
    }
    cohort_summary = {
        "protocol_version": PROTOCOL_VERSION,
        "eligibility": "RIDAGEYR >= 20 and all nine PHQ items in {0,1,2,3}",
        "reason": "Survey age eligibility for clinical questions and DMDEDUC2; correction is not based on model scores.",
        "total_raw_questionnaire_participants": len(questionnaire),
        "valid_phq9_before_age": valid_phq9_before_age,
        "excluded_valid_phq9_age_under20": excluded_under20,
        "excluded_valid_phq9_missing_age": excluded_missing_age,
        "cohort_n": len(cohort), "development_n": len(development_seqn),
        "held_out_n": len(test_seqn),
        "class_counts": {f"C{code}": int(class_counts[code]) for code in range(5)},
        "cohort_sha256": hashlib.sha256((PROCESSED_DATA_DIR / "cohort.csv").read_bytes()).hexdigest(),
        "role_migration": role_migration,
    }
    split_indices = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility": cohort_summary["eligibility"],
        "correction_reason": cohort_summary["reason"],
        "previous_split_sha256": hashlib.sha256(previous_split_path.read_bytes()).hexdigest(),
        "cohort_sha256": cohort_summary["cohort_sha256"],
        "role_migration": role_migration,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "development_seqn": [int(value) for value in development_seqn],
        "test_seqn": [int(value) for value in test_seqn],
    }
    split_path = ARTIFACTS_DIR / "split_indices.json"
    if split_path.exists() and load_json(split_path).get("protocol_version") == PROTOCOL_VERSION:
        existing = load_json(split_path)
        for key in ["development_seqn", "test_seqn", "cohort_sha256"]:
            if existing[key] != split_indices[key]:
                raise RuntimeError("Age20 split already locked but rebuild differs; review required")
        print("Reusing identical locked age20 split.")
    else:
        save_json(split_indices, split_path)
    save_json(cohort_summary, ARTIFACTS_DIR / "cohort_summary.json")

    print(f"Unavailable candidate features: {unavailable_features or 'None'}")
    print(f"Final cohort shape: {cohort.shape}")
    print(f"Development shape: ({len(development_seqn)}, {cohort.shape[1]})")
    print(f"Held-out test shape: ({len(test_seqn)}, {cohort.shape[1]})")
    print(f"Split role migration (IDs only): {role_migration}")


if __name__ == "__main__":
    main()
