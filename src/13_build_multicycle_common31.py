"""Freeze, build and audit common31 cohorts; no model, split or fitted transform.

Run: python src/13_build_multicycle_common31.py
Existing research and official raw bytes are checksum-verified before and after.
Completed outputs are immutable to this command: it refuses an accidental rerun.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from common31_features import (PHQ, QUANTITY, REMOVED, STATES, build_cohort,
                               common_schema, validate_cohort, validate_ids)
from nhanes_official_utils import (ARTIFACTS, CYCLES, RAW, ROOT, TABLES, Downloader,
                                    check_preservation, save_csv, save_json, sha256)
from nhanes_xpt_loader import load_nhanes_xpt

DEST = ROOT / "data/processed_multicycle"
POLICY = ARTIFACTS / "common31_harmonization_policy.json"
SUMMARY = ARTIFACTS / "common31_build_audit.json"
PRESERVED = ARTIFACTS / "common31_preservation.json"
MISSING_GAP_PP = 5.0
CATEGORY_GAP_PP = 10.0
MEDIAN_SHIFT_IQR = 0.5


def preserve():
    if not PRESERVED.exists():
        paths = []
        for directory in ("data/raw", "data/raw_official", "data/processed", "artifacts", "results", "src", "tests"):
            paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.name != ".DS_Store")
        paths.extend(ROOT / name for name in ("README.md", "requirements.txt", ".gitignore"))
        own = {"nhanes_xpt_loader.py", "common31_features.py", "13_build_multicycle_common31.py",
               "test_common31.py", "test_nhanes_xpt_loader.py"}
        save_json(PRESERVED, {"sha256": {str(p.relative_to(ROOT)): sha256(p)
                                       for p in sorted(set(paths)) if p.exists() and p.name not in own}})
    saved = json.loads(PRESERVED.read_text())["sha256"]
    changed = [name for name, digest in saved.items()
               if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Pre-existing files changed: {changed}")
    return len(saved)


def freeze_policy(schema):
    compatibility = pd.read_csv(TABLES / "multicycle_feature_compatibility.csv").set_index("feature")
    for row in schema:
        old_name = "alcohol_quantity_code" if row["feature"] == QUANTITY else row["feature"]
        if compatibility.loc[old_name, "overall_status"] not in {
            "EXACT", "DERIVED_COMPATIBLE", "COMPATIBLE_WITH_RECODE"
        }:
            raise ValueError(f"STOP: unresolved comparability for {old_name}")
        row["comparability_note"] = compatibility.loc[old_name, "harmonization_note"]
    sources = ["artifacts/final_feature_schema.json", "artifacts/nhanes_official_feature_dependencies.json",
               "artifacts/nhanes_official_source_catalog.json", "artifacts/nhanes_official_file_ledger.json",
               "artifacts/nhanes_official_codebook_evidence.json", "artifacts/nhanes_official_pdf_evidence.json",
               "results/tables/nhanes_official_reader_audit.csv", "results/tables/multicycle_feature_compatibility.csv",
               "src/config.py", "src/ablation_features.py"]
    policy = {
        "feature_set": "F4_common31", "n_features": 31, "cycles": list(CYCLES),
        "ordered_features": [row["feature"] for row in schema], "columns": schema,
        "excluded_features": REMOVED, "cycle_is_predictor": False,
        "eligibility": "RIDAGEYR >=20 AND all nine PHQ items in {0,1,2,3}",
        "target_mapping": {"C0": [0, 4], "C1": [5, 9], "C2": [10, 14], "C3": [15, 19], "C4": [20, 27]},
        "categorical_nonresponse": "Variable-specific codes 7/9 -> missing; DIQ010=3 retained; no imputation",
        "BP_means": "Mean of all available readings 1..4; keep legitimate DBP=0; all missing -> missing",
        "smoking": "SMQ020=2 -> under_100_lifetime; SMQ020=1 plus SMQ040=1/2/3 -> current_daily/current_some_days/former; otherwise missing",
        "alcohol_status": "Same previous routing: ALQ110=2 -> under_12_lifetime_quantity_not_asked; ALQ120Q=0 -> no_drinks_past_year; positive frequency or valid quantity -> drank_past_year; otherwise missing",
        "alcohol_quantity": "ALQ130 valid 1..95 in G/H, 1..15 in I; 777/999 nonresponses; all valid >=15 ->15; routed inapplicability -> tagged0 with status; unresolved ->NaN",
        "alcohol_quantity_state": {"observed": "Valid quantity", "not_applicable": "ALQ110=2 or ALQ120Q=0",
                                   "unknown": "System missing / unresolved routing without explicit special code",
                                   "nonresponse": "Explicit refusal or don't-know in unresolved applicable routing/quantity; original code retained in reason"},
        "metadata_not_predictors": ["SEQN", "cycle", *PHQ, "PHQ9_total", "severity", "severity_class",
                                    "alcohol_quantity_state", "alcohol_quantity_reason"],
        "zero_normalization": "Only prior-audited, hash-verified numeric fields whose original bytes are all zero and pandas value equals 2**-260",
        "missingness_review_threshold_pp": MISSING_GAP_PP,
        "categorical_shift_review_threshold_pp": CATEGORY_GAP_PP,
        "numeric_median_shift_review_threshold_reference_IQR": MEDIAN_SHIFT_IQR,
        "shift_thresholds_are_descriptive_not_significance_tests": True,
        "reference_cycle_for_numeric_shift": "2013_2014",
        "statistics": "Unweighted cohort descriptions; std uses ddof=1; percent missing uses final cohort N; category proportions include a missing level",
        "no_fit_no_imputation_no_split_no_model": True,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
    }
    if POLICY.exists():
        if json.loads(POLICY.read_text()) != policy:
            raise ValueError("Frozen common31 policy differs; will not overwrite")
    else:
        save_json(POLICY, policy)
    return policy


def load_cycle(cycle, schema, catalog):
    needed = sorted(set(PHQ) | {v for row in schema for v in row["source_variables"]})
    mapping = catalog["cycles"][cycle]["variable_mapping"]
    by_module = defaultdict(list)
    for variable in needed:
        by_module[mapping[variable]["module"]].append(variable)
    frames, audits = {}, []
    for module, columns in sorted(by_module.items()):
        frame, audit = load_nhanes_xpt(RAW / cycle / "xpt" / f"{module}.xpt", ["SEQN", *columns])
        validate_ids(frame)
        frames[module] = frame
        audits.extend(audit)
    demo_name = mapping["RIDAGEYR"]["module"]
    merged = frames[demo_name].copy()
    for module, frame in frames.items():
        if module == demo_name:
            continue
        if not set(frame.SEQN).issubset(set(merged.SEQN)):
            raise ValueError(f"Module has participant outside DEMO: {module}")
        merged = merged.merge(frame, on="SEQN", how="left", validate="one_to_one")
    validate_ids(merged)
    return merged, frames, audits


def distribution_row(cycle, frame, stages):
    row = {"cycle": cycle, **stages, "valid_phq_percent_total": 100 * stages["valid_phq_n"] / stages["total_n"],
           "age20_percent_total": 100 * stages["age20_n"] / stages["total_n"],
           "duplicate_seqn_n": int(frame.SEQN.duplicated().sum()), "class_percent_denominator": len(frame)}
    for label in range(5):
        count = int(frame.severity.eq(label).sum())
        row[f"C{label}_n"], row[f"C{label}_percent"] = count, 100 * count / len(frame)
    return row


def feature_audits(cohorts, schema):
    missing_rows, shift_rows = [], []
    for spec in schema:
        feature = spec["feature"]
        missing = {cycle: 100 * frame[feature].isna().mean() for cycle, frame in cohorts.items()}
        gap = max(missing.values()) - min(missing.values())
        for cycle, frame in cohorts.items():
            missing_rows.append({"feature": feature, "feature_type": spec["feature_type"], "cycle": cycle,
                                 "cohort_n": len(frame), "n_missing": int(frame[feature].isna().sum()),
                                 "missing_percent": missing[cycle], "between_cycle_gap_pp": gap,
                                 "flag": "REVIEW_MISSINGNESS_GAP" if gap >= MISSING_GAP_PP else "",
                                 "structural_note": "Applicability represented by status/tagged0; unknown and explicit nonresponse retained separately" if feature in [QUANTITY, "alcohol_status", "smoking_status"] else "No newly identified age-eligibility skip at age>=20; missingness is not assumed random"})
        if spec["feature_type"] == "numerical":
            reference = cohorts["2013_2014"][feature].dropna()
            ref_median, ref_iqr = reference.median(), reference.quantile(.75) - reference.quantile(.25)
            for cycle, frame in cohorts.items():
                values = frame[feature].dropna()
                q1, q3 = values.quantile(.25), values.quantile(.75)
                shift = abs(values.median() - ref_median) / ref_iqr if ref_iqr > 0 else np.nan
                shift_rows.append({"feature": feature, "feature_type": "numerical", "cycle": cycle,
                                   "level": "", "n": len(values), "denominator": len(frame), "proportion": np.nan,
                                   "median": values.median(), "q1": q1, "q3": q3, "iqr": q3 - q1,
                                   "mean": values.mean(), "std": values.std(ddof=1),
                                   "median_shift_reference_iqr": shift, "category_gap_pp": np.nan,
                                   "flag": "REVIEW_MEDIAN_SHIFT" if pd.notna(shift) and shift >= MEDIAN_SHIFT_IQR else "",
                                   "measurement_note": spec["comparability_note"]})
        else:
            levels = sorted({str(value) for frame in cohorts.values() for value in frame[feature].dropna()}) + ["__MISSING__"]
            for level in levels:
                counts = {cycle: int((frame[feature].isna() if level == "__MISSING__" else frame[feature].eq(level)).sum())
                          for cycle, frame in cohorts.items()}
                proportions = {cycle: counts[cycle] / len(frame) for cycle, frame in cohorts.items()}
                category_gap = 100 * (max(proportions.values()) - min(proportions.values()))
                for cycle, frame in cohorts.items():
                    shift_rows.append({"feature": feature, "feature_type": "categorical", "cycle": cycle,
                                       "level": level, "n": counts[cycle], "denominator": len(frame),
                                       "proportion": proportions[cycle], "median": np.nan, "q1": np.nan, "q3": np.nan,
                                       "iqr": np.nan, "mean": np.nan, "std": np.nan,
                                       "median_shift_reference_iqr": np.nan, "category_gap_pp": category_gap,
                                       "flag": "REVIEW_CATEGORY_SHIFT" if category_gap >= CATEGORY_GAP_PP else "",
                                       "measurement_note": spec["comparability_note"]})
    return pd.DataFrame(missing_rows), pd.DataFrame(shift_rows)


def equal_values(left, right):
    return left.eq(right) | (left.isna() & right.isna())


def kaggle_parity(schema, official, official_frames):
    """Compare raw before/after reading AND harmonized adult cohorts, not models."""
    dependencies = json.loads((ARTIFACTS / "nhanes_official_feature_dependencies.json").read_text())["features"]
    variable_source = {v: row["source_file"] for row in dependencies for v in row["source_variables"]}
    variable_source.update({v: "questionnaire.csv" for v in PHQ})
    needed = sorted(set(PHQ) | {v for row in schema for v in row["source_variables"]})
    sources = defaultdict(list)
    for variable in needed:
        sources[variable_source[variable]].append(variable)
    kaggle = {}
    for filename, variables in sources.items():
        path = ROOT / "data/raw" / filename
        frame = pd.read_csv(path, usecols=["SEQN", *variables], float_precision="round_trip")
        validate_ids(frame)
        kaggle[filename] = frame
    raw = kaggle["demographic.csv"].copy()
    for filename, frame in kaggle.items():
        if filename != "demographic.csv":
            raw = raw.merge(frame, on="SEQN", how="left", validate="one_to_one")
    kaggle_cohort, _, _ = build_cohort(raw, "2013_2014", schema)
    if set(kaggle_cohort.SEQN) != set(official.SEQN):
        raise ValueError("Official-vs-Kaggle cohort membership mismatch after correction")
    raw_rows, details = [], []
    for module, normalized in official_frames.items():
        before = pd.read_sas(RAW / "2013_2014" / "xpt" / f"{module}.xpt", format="xport")
        before.columns = before.columns.str.upper()
        before = before.set_index("SEQN")
        after = normalized.set_index("SEQN")
        for variable in after:
            comparison = kaggle[variable_source[variable]].set_index("SEQN")
            ids = after.index.intersection(comparison.index)
            left, old, right = comparison.loc[ids, variable], before.loc[ids, variable], after.loc[ids, variable]
            bad_before, bad_after = ~equal_values(left, old), ~equal_values(left, right)
            raw_rows.append({"variable": variable, "common_ids_n": len(ids),
                             "mismatch_before_zero_normalization_n": int(bad_before.sum()),
                             "mismatch_after_zero_normalization_n": int(bad_after.sum()),
                             "mismatches_resolved_n": int(bad_before.sum() - bad_after.sum()),
                             "match_percent_after": 100 * (1 - bad_after.mean()),
                             "missing_pattern_mismatch_n": int(left.isna().ne(right.isna()).sum())})
            for seqn in ids[bad_after]:
                details.append({"scope": "raw_dependency", "SEQN": int(seqn), "variable": variable,
                                "kaggle": left.loc[seqn], "official": right.loc[seqn],
                                "note": "Documented source difference retained; official data not altered"})
    left, right = kaggle_cohort.set_index("SEQN"), official.set_index("SEQN")
    columns = [row["feature"] for row in schema] + PHQ + ["PHQ9_total", "severity"]
    feature_rows = []
    for column in columns:
        a, b = left.loc[right.index, column], right[column]
        bad = ~equal_values(a, b)
        feature_rows.append({"feature": column, "common_cohort_n": len(right), "n_mismatch": int(bad.sum()),
                             "match_percent": 100 * (1 - bad.mean()),
                             "missing_pattern_mismatch_n": int(a.isna().ne(b.isna()).sum())})
        for seqn in right.index[bad]:
            details.append({"scope": "common31_cohort", "SEQN": int(seqn), "variable": column,
                            "kaggle": a.loc[seqn], "official": b.loc[seqn],
                            "note": "Documented source difference retained; official data not altered"})
    for row in details:
        if (row["SEQN"], row["variable"], float(row["kaggle"]), float(row["official"])) != (73756, "PAQ665", 1., 2.):
            raise ValueError(f"Unexpected source mismatch; stop for review: {row}")
    save_csv(TABLES / "common31_2013_raw_parity.csv", raw_rows)
    save_csv(TABLES / "common31_2013_kaggle_parity.csv", feature_rows)
    if details:
        save_csv(TABLES / "common31_2013_source_differences.csv", details)
    return {"cohort_membership_identical": True, "cohort_n": len(right),
            "raw_dependencies_compared": len(raw_rows),
            "raw_mismatches_before": sum(row["mismatch_before_zero_normalization_n"] for row in raw_rows),
            "raw_mismatches_after": sum(row["mismatch_after_zero_normalization_n"] for row in raw_rows),
            "common31_plus_phq_mismatches": sum(row["n_mismatch"] for row in feature_rows),
            "documented_differences": details}


def main():
    if SUMMARY.exists():
        raise FileExistsError("Completed common31 artifacts already exist; refusing to overwrite")
    preserved_n = preserve()
    check_preservation()
    raw_files_n = Downloader().verify_all()
    schema = common_schema()
    policy = freeze_policy(schema)
    print("Frozen F4_common31:", policy["ordered_features"], flush=True)
    catalog = json.loads((ARTIFACTS / "nhanes_official_source_catalog.json").read_text())
    cohorts, stage_records, audit_rows, routing_rows, class_rows = {}, [], [], [], []
    official_frames = None
    DEST.mkdir(parents=True, exist_ok=True)
    for cycle in CYCLES:
        raw, frames, audits = load_cycle(cycle, schema, catalog)
        cohort, stages, routing = build_cohort(raw, cycle, schema)
        validate_cohort(cohort, cycle, schema)
        cohorts[cycle] = cohort
        stage_records.append(stages)
        audit_rows.extend(audits)
        routing_rows.append(routing)
        class_rows.append(distribution_row(cycle, cohort, stages))
        cohort.to_csv(DEST / f"cohort_{cycle}.csv", index=False)
        if cycle == "2013_2014":
            official_frames = frames
        print(f"Per-cycle audit PASS: {cycle}: {stages}; duplicates=0", flush=True)

    # Audit each cycle and cross-cycle schema/coding BEFORE concatenation.
    reference = cohorts["2013_2014"]
    for cycle, cohort in cohorts.items():
        validate_cohort(cohort, cycle, schema)
        if list(cohort.columns) != list(reference.columns) or not cohort.dtypes.equals(reference.dtypes):
            raise ValueError("STOP: cycle schemas/types differ")
    missingness, shifts = feature_audits(cohorts, schema)
    missingness.to_csv(TABLES / "multicycle_common31_missingness.csv", index=False)
    shifts.to_csv(TABLES / "multicycle_feature_shift_audit.csv", index=False)
    save_csv(TABLES / "multicycle_common31_routing_audit.csv", routing_rows)
    save_csv(TABLES / "xpt_zero_normalization_audit.csv", audit_rows)
    parity = kaggle_parity(schema, reference, official_frames)
    print("Parity PASS:", parity, flush=True)

    pooled = pd.concat([cohorts[cycle] for cycle in CYCLES], ignore_index=True)
    validate_ids(pooled)
    pooled_stages = {name: sum(record[name] for record in stage_records) for name in stage_records[0]}
    class_rows.append(distribution_row("pooled_2011_2016", pooled, pooled_stages))
    pooled.to_csv(DEST / "cohort_2011_2016_common31.csv", index=False)
    save_csv(TABLES / "multicycle_class_distribution.csv", class_rows)

    if preserve() != preserved_n or Downloader().verify_all() != raw_files_n:
        raise ValueError("Preservation check changed")
    check_preservation()
    missing_flags = sorted(missingness.loc[missingness.flag.ne(""), "feature"].unique().tolist())
    shift_flags = sorted(shifts.loc[shifts.flag.ne(""), "feature"].unique().tolist())
    base = class_rows[1]
    summary = {
        "stage": "HARMONIZATION_COHORT_CONSTRUCTION_AUDIT_ONLY", "training_performed": False,
        "splitting_performed": False, "report_updated": False, "policy_sha256": sha256(POLICY),
        "class_distribution": class_rows, "zero_corrections_n": sum(row["n_cells_corrected"] for row in audit_rows),
        "zero_corrections_by_cycle": {cycle: sum(row["n_cells_corrected"] for row in audit_rows if row["cycle"] == cycle) for cycle in CYCLES},
        "zero_audit_scope": "48 raw dependencies including 9 PHQ items, plus SEQN; 15 needed modules/cycle; excluded sleep module not processed",
        "pooled_C3_increase_vs_2013_2014": class_rows[-1]["C3_n"] - base["C3_n"],
        "pooled_C4_increase_vs_2013_2014": class_rows[-1]["C4_n"] - base["C4_n"],
        "missingness_review_flags": missing_flags, "distribution_shift_review_flags": shift_flags,
        "schema_coding_parity_passed": True, "unresolved_noncomparable_features": [],
        "automatic_training_authorized": False,
        "readiness": "REVIEW_DESCRIPTIVE_FLAGS_BEFORE_TRAINING" if missing_flags or shift_flags else "COMMON_CORE_AUDITS_PASS; future experiment requires its own approved protocol",
        "parity": parity, "preserved_old_files_n": preserved_n, "verified_official_raw_document_files_n": raw_files_n,
        "limitations": ["Unweighted, complete-PHQ adult cohort; not nationally representative estimates",
                        "Laboratory platform/site transitions retained with source documentation; no unvalidated calibration",
                        "PAQ context wording differences retained as an audit limitation; excluded PAQ605 and PAD680 are not predictors",
                        "Unknown and explicit alcohol nonresponse are separate metadata, not extra common31 predictors",
                        "Future fair single-vs-multi-cycle comparison must use common31 and the same quantity cap in both arms"],
    }
    outputs = [*DEST.glob("cohort_*.csv"), POLICY,
               *[TABLES / name for name in ["multicycle_class_distribution.csv", "multicycle_common31_missingness.csv",
                                           "multicycle_feature_shift_audit.csv", "multicycle_common31_routing_audit.csv",
                                           "xpt_zero_normalization_audit.csv", "common31_2013_raw_parity.csv",
                                           "common31_2013_kaggle_parity.csv", "common31_2013_source_differences.csv"]],
               *[ROOT / "src" / name for name in ["13_build_multicycle_common31.py", "common31_features.py", "nhanes_xpt_loader.py"]]]
    summary["output_and_code_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in outputs if path.exists()}
    save_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
