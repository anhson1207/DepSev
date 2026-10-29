"""Restore original membership and filter age >= 20 within each partition.

No train_test_split call and no held-out model predictions. Only SEQN, age,
and severity are read for the explicitly requested partition counts.
"""

from datetime import datetime, timezone

import pandas as pd

from config import (ARTIFACTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TABLES_DIR,
                    PRIMARY_MIN_AGE, PROTOCOL_VERSION, experiment_path)
from protocol import (FINAL_SPLIT, ORIGINAL_SPLIT, load_original_split,
                      membership_assertions, sha256_file)
from utils import load_json, save_json


def main():
    archive = ARTIFACTS_DIR / "archive/age20_v2"
    manifest = load_json(archive / "archive_manifest.json")
    archived_split = archive / "artifacts/split_indices.json"
    if sha256_file(archived_split) != manifest["files_sha256"]["artifacts/split_indices.json"]:
        raise RuntimeError("Missing or altered age20_v2 archive")
    reshuffled = load_json(archived_split)
    original = load_original_split()
    if reshuffled["previous_split_sha256"] != sha256_file(ORIGINAL_SPLIT):
        raise RuntimeError("Archive chain does not point to the first split")

    # Membership depends ONLY on original IDs and raw age, never on model scores.
    ages = pd.read_csv(RAW_DATA_DIR / "demographic.csv", usecols=["SEQN", "RIDAGEYR"])
    if ages.SEQN.duplicated().any() or ages.SEQN.isna().any():
        raise RuntimeError("Invalid demographic IDs")
    age_by_id = ages.set_index("SEQN").RIDAGEYR
    original_ids = set(original["development_seqn"]) | set(original["test_seqn"])
    if not original_ids <= set(age_by_id.index) or age_by_id.loc[list(original_ids)].isna().any():
        raise RuntimeError("Original participants have unknown age; review required")
    eligible = set(ages.loc[ages.RIDAGEYR.ge(PRIMARY_MIN_AGE), "SEQN"].astype(int))
    development = [int(i) for i in original["development_seqn"] if i in eligible]
    heldout = [int(i) for i in original["test_seqn"] if i in eligible]
    checks = membership_assertions(development, heldout, original)

    cohort_path = PROCESSED_DATA_DIR / "cohort.csv"
    if sha256_file(cohort_path) != reshuffled["cohort_sha256"]:
        raise RuntimeError("Expected the unchanged age >= 20 cohort from raw rebuild")
    # Target access for heldout is limited to the user-authorized class counts.
    cohort = pd.read_csv(cohort_path, usecols=["SEQN", "RIDAGEYR", "severity"])
    if (cohort.SEQN.duplicated().any() or not cohort.RIDAGEYR.ge(PRIMARY_MIN_AGE).all()
            or not cohort.severity.isin(range(5)).all()
            or set(cohort.SEQN) != set(development) | set(heldout)):
        raise RuntimeError("Cohort does not equal original membership filtered by age")
    counts = []
    for code in range(5):
        counts.append({"class": f"C{code}",
                       "development_count": int((cohort.SEQN.isin(development) & cohort.severity.eq(code)).sum()),
                       "heldout_count": int((cohort.SEQN.isin(heldout) & cohort.severity.eq(code)).sum())})
    previously_used = set(heldout) & set(reshuffled["development_seqn"])
    history = {
        "heldout_age20_previously_used_in_age20_v2_development_n": len(previously_used),
        "heldout_age20_previously_used_in_age20_v2_development_seqn": sorted(previously_used),
        "historically_untouched": False,
        "heldout_evaluation_performed": False,
        "note": "Restoring original membership cannot undo development exposure in age20_v2. No heldout model predictions/metrics are produced by this correction.",
    }
    payload = {
        "protocol": PROTOCOL_VERSION, "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "membership_rule": "original_development/test intersect RIDAGEYR >= 20 independently",
        "new_random_train_test_split": False,
        "eligibility": "RIDAGEYR >= 20; existing complete nine-item PHQ-9 cohort unchanged",
        "original_split_sha256": sha256_file(ORIGINAL_SPLIT),
        "retired_age20_v2_split_sha256": sha256_file(archived_split),
        "cohort_sha256": sha256_file(cohort_path),
        "age_source_sha256": sha256_file(RAW_DATA_DIR / "demographic.csv"),
        "original_split_random_state": original["random_state"],
        "original_split_test_size": original["test_size"],
        "development_n": len(development), "heldout_n": len(heldout),
        "excluded_age_under20_development": len(original["development_seqn"]) - len(development),
        "excluded_age_under20_heldout": len(original["test_seqn"]) - len(heldout),
        "development_seqn": development, "test_seqn": heldout,
        "membership_assertions": checks, "historical_exposure": history,
    }
    active_path = ARTIFACTS_DIR / "split_indices.json"
    if FINAL_SPLIT.exists():
        existing = load_json(FINAL_SPLIT)
        payload["created_utc"] = existing["created_utc"]
        if existing != payload or load_json(active_path) != payload:
            raise RuntimeError("Final protocol already exists but differs; review before any writes")
    else:
        if sha256_file(active_path) != sha256_file(archived_split):
            raise RuntimeError("Active split changed after its verified archive")
        save_json(payload, FINAL_SPLIT)
        save_json(payload, active_path)
        retired = archive / "retired_active_artifacts"
        retired.mkdir(exist_ok=True)
        for name in ["cv_folds.json", "ablation_config.json", "repeated_cv_folds.json",
                     "robustness_config.json", "cohort_summary.json"]:
            path = ARTIFACTS_DIR / name
            if sha256_file(path) != manifest["files_sha256"][f"artifacts/{name}"]:
                raise RuntimeError(f"Refusing to retire changed artifact: {name}")
            path.rename(retired / name)
    summary = {
        "protocol": PROTOCOL_VERSION, "cohort_n": len(cohort),
        "development_n": len(development), "heldout_n": len(heldout),
        "class_counts": counts, "membership_assertions": checks,
        "historical_exposure": history,
        "cohort_sha256": payload["cohort_sha256"], "split_sha256": sha256_file(FINAL_SPLIT),
    }
    save_json(summary, experiment_path(ARTIFACTS_DIR, "cohort_summary.json"))
    pd.DataFrame(counts).to_csv(experiment_path(TABLES_DIR, "class_distribution.csv"), index=False)
    print(f"Protocol: {PROTOCOL_VERSION}; no new random split")
    print(f"Development N={len(development)}; heldout N={len(heldout)}")
    print(pd.DataFrame(counts).to_string(index=False))
    for name, passed in checks.items():
        print(f"ASSERT {name}: {passed}")
    print(f"Historical warning: {len(previously_used)} restored heldout IDs were used in age20_v2 development")
    print("No heldout predictors, predictions or model metrics inspected.")


if __name__ == "__main__":
    main()
