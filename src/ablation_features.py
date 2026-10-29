"""Development-only features with documented NHANES questionnaire skip rules.

These deterministic rules do not fit statistics or use PHQ answers. Original
cohort and feature inventory are left unchanged. See results/ablation_report.md
and the CDC codebook URLs below for the interpretation of each special code.
"""

import csv

import numpy as np
import pandas as pd

from config import LEAKAGE_COLUMNS, RAW_DATA_DIR


CODEBOOKS = {
    name: f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/{name}_H.htm"
    for name in ["SMQ", "ALQ", "PAQ", "SLQ", "BPQ", "DIQ", "MCQ", "HUQ", "DEMO"]
}

# Domains are variable-specific: e.g. DIQ010=3 (borderline) remains valid.
CATEGORY_DOMAINS = {
    "BPQ020": [1, 2],
    "DIQ010": [1, 2, 3],
    "MCQ160B": [1, 2],
    "MCQ160C": [1, 2],
    "MCQ160D": [1, 2],
    "MCQ160E": [1, 2],
    "MCQ160F": [1, 2],
    "MCQ220": [1, 2],
    "HUQ010": [1, 2, 3, 4, 5],
    "PAQ605": [1, 2],
    "PAQ620": [1, 2],
    "PAQ650": [1, 2],
    "PAQ665": [1, 2],
    "ALQ101": [1, 2],
    "RIAGENDR": [1, 2],
    "RIDRETH1": [1, 2, 3, 4, 5],
    "DMDEDUC2": [1, 2, 3, 4, 5],
}


def read_development_rows(path, development_ids, columns):
    """Scan IDs in a CSV but materialize requested values only for development.

    The development row order remains the original cohort order, matching the
    previous F1 baseline. Test predictors/targets are never parsed into a frame.
    """
    requested_ids = set(development_ids)
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate column names in {path.name}")
        missing = set(["SEQN", *columns]) - set(header)
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        positions = [header.index(column) for column in columns]
        id_position = header.index("SEQN")
        for row in reader:
            if int(row[id_position]) in requested_ids:
                rows.append([row[position] for position in positions])
    data = pd.DataFrame(rows, columns=columns).replace("", np.nan)
    data = data.apply(pd.to_numeric, errors="raise")
    if data.SEQN.isna().any() or data.SEQN.duplicated().any():
        raise ValueError(f"Missing/duplicated development IDs in {path.name}")
    if set(data.SEQN) != requested_ids:
        raise ValueError(f"Development IDs do not match {path.name}")
    return data


def clean_codes(values, valid_codes, missing_codes=(7, 9)):
    """Clean documented nonresponses; reject unknown codes for human review."""
    unexpected = values.notna() & ~values.isin([*valid_codes, *missing_codes])
    if unexpected.any():
        raise ValueError(
            f"Unrecognized coding in {values.name}: "
            f"{sorted(values[unexpected].unique().tolist())}"
        )
    return values.where(values.isin(valid_codes))


def numeric_response(values, minimum, maximum, missing_codes):
    """Keep documented numerical values, including genuine top-coded values."""
    valid = values.between(minimum, maximum) & values.eq(values.round())
    unexpected = values.notna() & ~valid & ~values.isin(missing_codes)
    if unexpected.any():
        raise ValueError(f"Unrecognized coding in {values.name}")
    return values.where(valid)


def prepare_features(development, inventory):
    """Validate inventory and construct the fixed P/C/L/D representation."""
    required = {"feature", "group", "feature_type", "source_file", "description"}
    if not required.issubset(inventory.columns) or inventory.feature.duplicated().any():
        raise ValueError("Feature inventory is missing fields or has duplicate entries")
    if set(inventory.group) != {"P", "C", "L", "D"}:
        raise ValueError("Expected exactly the P/C/L/D groups in the inventory")
    if not set(inventory.feature_type).issubset({"numerical", "categorical"}):
        raise ValueError("Unknown feature type in inventory")
    features = inventory.feature.tolist()
    if set(features) & LEAKAGE_COLUMNS or any(c.startswith("DPQ") for c in features):
        raise ValueError("Target leakage column in inventory")
    if not set(features).issubset(development.columns):
        raise ValueError("Inventory contains features absent from the cohort")
    for row in inventory.itertuples(index=False):
        if not (RAW_DATA_DIR / row.source_file).is_file():
            raise FileNotFoundError(f"Inventory source not found: {row.source_file}")

    clean = development[features].copy()
    audits = []

    def record(feature, reason, mask):
        count = int(mask.sum())
        audits.append({"feature": feature, "reason": reason, "count": count,
                       "percent_development": 100 * count / len(development)})

    for column, valid_codes in CATEGORY_DOMAINS.items():
        if column not in clean:
            continue
        values = clean_codes(clean[column], valid_codes,
                             () if column in ["RIAGENDR", "RIDRETH1"] else (7, 9))
        record(column, "special_nonresponse_to_missing",
               clean[column].notna() & values.isna())
        # Explicit object strings + np.nan work with sklearn's categorical imputer.
        clean[column] = values.map(lambda x: str(int(x)) if pd.notna(x) else np.nan)
        clean[column] = clean[column].astype(object)

    for column, minimum, maximum, codes in [
        ("PAD680", 0, 1440, [7777, 9999]),
        ("SLD010H", 2, 12, [77, 99]),
    ]:
        values = numeric_response(clean[column], minimum, maximum, codes)
        record(column, "special_nonresponse_to_missing",
               clean[column].notna() & values.isna())
        clean[column] = values

    smoking_lifetime = clean_codes(development.SMQ020, [1, 2])
    smoking_now = clean_codes(development.SMQ040, [1, 2, 3])
    if (smoking_lifetime.ne(1) & smoking_now.notna()).any():
        raise ValueError("Inconsistent SMQ020/SMQ040 skip pattern; review before fitting")
    smoking = pd.Series(np.nan, index=development.index, dtype=object)
    smoking.loc[smoking_lifetime.eq(2)] = "under_100_lifetime"
    for code, label in [(1, "current_daily"), (2, "current_some_days"), (3, "former")]:
        smoking.loc[smoking_lifetime.eq(1) & smoking_now.eq(code)] = label
    clean["smoking_status"] = smoking
    record("SMQ040", "raw_missing", development.SMQ040.isna())
    record("SMQ040", "structural_under_100_lifetime",
           development.SMQ040.isna() & smoking_lifetime.eq(2))
    record("SMQ040", "unresolved_smoking_status", smoking.isna())

    # These routing columns were verified in the same real source as ALQ101.
    source_names = inventory.set_index("feature").loc[
        ["ALQ101", "ALQ130"], "source_file"
    ].unique()
    if len(source_names) != 1:
        raise ValueError("Alcohol items do not share a single verified source")
    auxiliary = read_development_rows(
        RAW_DATA_DIR / source_names[0], development.SEQN,
        ["SEQN", "ALQ101", "ALQ110", "ALQ120Q", "ALQ130"],
    ).set_index("SEQN").loc[development.SEQN].reset_index(drop=True)
    for column in ["ALQ101", "ALQ130"]:
        pd.testing.assert_series_equal(
            auxiliary[column], development[column], check_names=False, check_dtype=False
        )
    lifetime = clean_codes(auxiliary.ALQ110, [1, 2])
    frequency = numeric_response(auxiliary.ALQ120Q, 0, 366, [777, 999])
    quantity = numeric_response(auxiliary.ALQ130, 1, 95, [777, 999])
    below_threshold = lifetime.eq(2)
    no_past_year = frequency.eq(0)
    if ((below_threshold & frequency.notna()) | ((below_threshold | no_past_year)
                                                & quantity.notna())).any():
        raise ValueError("Inconsistent alcohol routing; review before fitting")

    status = pd.Series(np.nan, index=development.index, dtype=object)
    status.loc[frequency.gt(0) | quantity.notna()] = "drank_past_year"
    status.loc[below_threshold] = "under_12_lifetime_quantity_not_asked"
    status.loc[no_past_year] = "no_drinks_past_year"
    # Tagged zero is a numerical placeholder for an inapplicable question, NOT
    # evidence of zero lifetime alcohol use. alcohol_status must accompany it.
    quantity_code = quantity.copy()
    quantity_code.loc[below_threshold | no_past_year] = 0
    clean["alcohol_status"] = status
    clean["alcohol_quantity_code"] = quantity_code
    record("ALQ130", "raw_missing", development.ALQ130.isna())
    record("ALQ130", "structural_under_12_lifetime", below_threshold)
    record("ALQ130", "structural_no_drinks_past_year", no_past_year)
    record("ALQ130", "special_nonresponse_to_missing",
           development.ALQ130.notna() & quantity.isna())
    record("ALQ130", "quantity_unresolved_after_routing", quantity_code.isna())
    record("ALQ130", "alcohol_status_unresolved", status.isna())

    # Record additional age-related skips; age is NOT fed into C-only features.
    for column in ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
                   "MCQ220", "DMDEDUC2"]:
        record(column, "raw_missing_age_under20",
               development[column].isna() & development.RIDAGEYR.lt(20))
        record(column, "raw_missing_age20plus",
               development[column].isna() & development.RIDAGEYR.ge(20))

    replacements = {
        "SMQ020": [("smoking_status", "categorical", "SMQ020;SMQ040",
                    "Lifetime threshold and current cigarette smoking status")],
        "SMQ040": [],
        "ALQ130": [
            ("alcohol_status", "categorical", "ALQ101;ALQ110;ALQ120Q;ALQ130",
             "Alcohol questionnaire applicability / past-year status"),
            ("alcohol_quantity_code", "numerical", "ALQ110;ALQ120Q;ALQ130",
             "Drinks on drinking days; tagged 0 for inapplicable quantity question"),
        ],
    }
    specs = []
    for row in inventory.to_dict(orient="records"):
        feature = row["feature"]
        entries = replacements.get(feature, [(feature, row["feature_type"], feature,
                                                row["description"])])
        for name, feature_type, source_columns, description in entries:
            specs.append({"feature": name, "group": row["group"],
                          "feature_type": feature_type, "source_file": row["source_file"],
                          "source_columns": source_columns, "description": description})
    final_inventory = pd.DataFrame(specs)
    groups = {group: final_inventory.loc[final_inventory.group.eq(group), "feature"].tolist()
              for group in ["P", "C", "L", "D"]}
    clean = clean[final_inventory.feature.tolist()]
    for column in final_inventory.loc[final_inventory.feature_type.eq("categorical"), "feature"]:
        clean[column] = clean[column].astype(object).where(clean[column].notna(), np.nan)
    numerical = final_inventory.loc[final_inventory.feature_type.eq("numerical"), "feature"]
    if np.isinf(clean[numerical].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numerical predictor encountered")
    final_inventory["missing_count_development"] = [
        int(clean[column].isna().sum()) for column in final_inventory.feature
    ]
    final_inventory["missing_percent_development"] = (
        100 * final_inventory.missing_count_development / len(clean)
    )
    return clean, groups, final_inventory, pd.DataFrame(audits)
