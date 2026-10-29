"""Deterministic, non-fitted common-core rules shared by all three cycles."""

import json

import numpy as np
import pandas as pd

from ablation_features import CATEGORY_DOMAINS, clean_codes, numeric_response
from nhanes_official_utils import ARTIFACTS

PHQ = [f"DPQ{item:03d}" for item in range(10, 100, 10)]
REMOVED = {
    "SLD010H": "Missing / non-equivalent measurement in 2015-2016; no SLD012 substitution",
    "PAQ605": "Questionnaire definition changed",
    "PAD680": "Prompt/probe wording changed",
}
QUANTITY = "alcohol_quantity_code_common"
STATES = ["observed", "not_applicable", "unknown", "nonresponse"]


def common_schema():
    """Derive the 31-column ordered schema from the frozen original F4."""
    original = json.loads((ARTIFACTS / "final_feature_schema.json").read_text())
    dependencies = json.loads((ARTIFACTS / "nhanes_official_feature_dependencies.json").read_text())
    dependencies = {row["feature"]: row for row in dependencies["features"]}
    if original["n_features"] != 34:
        raise ValueError("Expected the frozen 34-feature F4 specification")
    schema = []
    for old in original["columns"]:
        if old["feature"] in REMOVED:
            continue
        row = dict(old)
        row["source_variables"] = dependencies[old["feature"]]["source_variables"]
        row["feature"] = QUANTITY if old["feature"] == "alcohol_quantity_code" else old["feature"]
        if row["feature"] == QUANTITY:
            row["description"] = "Drinks on drinking days, common top-code 15; tagged 0 only for routed inapplicability"
        row["source_columns"] = ";".join(row["source_variables"])
        schema.append(row)
    if len(schema) != 31 or len({row["feature"] for row in schema}) != 31:
        raise ValueError("Common core must have exactly 31 distinct predictors")
    return schema


def validate_ids(frame):
    if frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
        raise ValueError("Missing/duplicated SEQN: expected one participant per row")
    if not frame.SEQN.eq(frame.SEQN.round()).all():
        raise ValueError("Noninteger participant identifier")


def alcohol_features(raw, cycle):
    """Preserve previous status/structural-zero rules, with a common quantity cap.

    Metadata distinguish system-missing/unresolved (unknown) from explicit
    refusal/don't-know (nonresponse). Reasons retain the individual special codes.
    These metadata are not additional predictors. No routing is mode-imputed.
    """
    lifetime = clean_codes(raw.ALQ110, [1, 2])
    frequency = numeric_response(raw.ALQ120Q, 0, 366, [777, 999])
    quantity = numeric_response(raw.ALQ130, 1, 15 if cycle == "2015_2016" else 95, [777, 999])
    below, no_year = lifetime.eq(2), frequency.eq(0)
    if ((below & frequency.notna()) | ((below | no_year) & quantity.notna())).any():
        raise ValueError("Inconsistent alcohol routing; stop for source review")
    status = pd.Series(np.nan, index=raw.index, dtype=object)
    status.loc[frequency.gt(0) | quantity.notna()] = "drank_past_year"
    status.loc[below] = "under_12_lifetime_quantity_not_asked"
    status.loc[no_year] = "no_drinks_past_year"
    code = quantity.clip(upper=15)
    code.loc[below | no_year] = 0.0
    state = pd.Series("unknown", index=raw.index, dtype=object)
    reason = pd.Series("system_missing_or_unresolved_routing", index=raw.index, dtype=object)
    for column, specials in [("ALQ101", [7, 9]), ("ALQ110", [7, 9]),
                             ("ALQ120Q", [777, 999]), ("ALQ130", [777, 999])]:
        for special in specials:
            mask = raw[column].eq(special) & code.isna()
            state.loc[mask] = "nonresponse"
            # Keep all explicit codes if multiple routing items are unresolved.
            reason.loc[mask] = reason.loc[mask] + f";{column}={special}"
    state.loc[quantity.notna()] = "observed"
    reason.loc[quantity.notna()] = "valid_quantity"
    state.loc[below | no_year] = "not_applicable"
    reason.loc[below] = "under_12_lifetime_quantity_not_asked"
    reason.loc[no_year] = "no_drinks_past_year"
    return status, code, state, reason


def build_cohort(raw, cycle, schema):
    """Filter age >=20 and nine valid PHQ items; derive features without fitting."""
    validate_ids(raw)
    raw = raw.copy()
    for column in PHQ:
        raw[column] = clean_codes(raw[column], [0, 1, 2, 3])
    valid_phq = raw[PHQ].notna().all(axis=1)
    adult = raw.RIDAGEYR.ge(20)
    stages = {"total_n": len(raw), "valid_phq_n": int(valid_phq.sum()),
              "age20_n": int(adult.sum()), "cohort_n": int((valid_phq & adult).sum())}
    selected = raw.loc[valid_phq & adult].copy().sort_values("SEQN").reset_index(drop=True)
    clean = pd.DataFrame({"SEQN": selected.SEQN.astype("int64"), "cycle": cycle})
    for row in schema:
        name = row["feature"]
        if name in selected:
            clean[name] = selected[name].astype(float)
    for name in (set(CATEGORY_DOMAINS) & set(clean)):
        value = clean_codes(clean[name], CATEGORY_DOMAINS[name],
                            () if name in ["RIAGENDR", "RIDRETH1"] else (7, 9))
        clean[name] = value.map(lambda x: str(int(x)) if pd.notna(x) else np.nan).astype(object)
    for name, prefix in [("SBP_mean", "BPXSY"), ("DBP_mean", "BPXDI")]:
        # Preserve original mean of all available 1..4 readings, including valid 0 DBP.
        clean[name] = selected[[f"{prefix}{i}" for i in range(1, 5)]].mean(axis=1)
    lifetime = clean_codes(selected.SMQ020, [1, 2])
    current = clean_codes(selected.SMQ040, [1, 2, 3])
    if (lifetime.ne(1) & current.notna()).any():
        raise ValueError("Inconsistent smoking skip pattern")
    smoke = pd.Series(np.nan, index=selected.index, dtype=object)
    smoke.loc[lifetime.eq(2)] = "under_100_lifetime"
    for value, label in [(1, "current_daily"), (2, "current_some_days"), (3, "former")]:
        smoke.loc[lifetime.eq(1) & current.eq(value)] = label
    clean["smoking_status"] = smoke
    status, quantity, state, reason = alcohol_features(selected, cycle)
    clean["alcohol_status"], clean[QUANTITY] = status, quantity
    clean["alcohol_quantity_state"], clean["alcohol_quantity_reason"] = state, reason
    for column in PHQ:
        clean[column] = selected[column].astype("int64")
    clean["PHQ9_total"] = clean[PHQ].sum(axis=1).astype("int64")
    clean["severity"] = pd.cut(clean.PHQ9_total, [-1, 4, 9, 14, 19, 27], labels=False).astype("int64")
    clean["severity_class"] = "C" + clean.severity.astype(str)
    feature_names = [row["feature"] for row in schema]
    ordered = ["SEQN", "cycle", *feature_names, *PHQ, "PHQ9_total", "severity",
               "severity_class", "alcohol_quantity_state", "alcohol_quantity_reason"]
    clean = clean[ordered]
    validate_cohort(clean, cycle, schema)
    routing = {
        "cycle": cycle, "cohort_n": len(clean),
        "SMQ040_raw_missing_n": int(selected.SMQ040.isna().sum()),
        "SMQ040_structural_under100_n": int((selected.SMQ040.isna() & lifetime.eq(2)).sum()),
        "ALQ130_raw_missing_n": int(selected.ALQ130.isna().sum()),
        "ALQ130_valid_ge15_n": int(selected.ALQ130.between(15, 95).sum()),
        "ALQ130_valid_gt15_changed_n": int(selected.ALQ130.between(16, 95).sum()),
        **{f"quantity_{label}_n": int(state.eq(label).sum()) for label in STATES},
    }
    return clean, stages, routing


def validate_cohort(frame, cycle, schema):
    validate_ids(frame)
    if not frame.cycle.eq(cycle).all() or not frame.RIDAGEYR.ge(20).all():
        raise ValueError("Invalid cycle/age provenance")
    if not frame[PHQ].isin([0, 1, 2, 3]).all().all():
        raise ValueError("Incomplete/invalid PHQ items")
    if not frame.PHQ9_total.eq(frame[PHQ].sum(axis=1)).all():
        raise ValueError("PHQ total inconsistent with items")
    for row in schema:
        name = row["feature"]
        if row["feature_type"] == "numerical":
            if np.isinf(frame[name].to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite feature: {name}")
        elif name in CATEGORY_DOMAINS:
            if not frame[name].dropna().isin([str(x) for x in CATEGORY_DOMAINS[name]]).all():
                raise ValueError(f"Inconsistent category domain: {name}")
    if not frame[QUANTITY].dropna().between(0, 15).all():
        raise ValueError("Common quantity cap not respected")
    if not frame[QUANTITY].eq(0).equals(frame.alcohol_quantity_state.eq("not_applicable")):
        raise ValueError("Structural quantity zero lost its applicability tag")
