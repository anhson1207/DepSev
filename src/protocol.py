"""Split provenance checks shared by all active final-protocol experiments."""

import hashlib

from config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROTOCOL_VERSION, experiment_path
from utils import load_json


ORIGINAL_ARCHIVE = ARTIFACTS_DIR / "archive/pre_age20_v1"
ORIGINAL_SPLIT = ORIGINAL_ARCHIVE / "artifacts/split_indices.json"
FINAL_SPLIT = experiment_path(ARTIFACTS_DIR, "split_indices.json")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original_split():
    """Verify the first preserved membership, not a later reshuffled version."""
    manifest = load_json(ORIGINAL_ARCHIVE / "archive_manifest.json")
    if sha256_file(ORIGINAL_SPLIT) != manifest["files_sha256"]["artifacts/split_indices.json"]:
        raise RuntimeError("Original split hash does not match its preserved archive")
    original = load_json(ORIGINAL_SPLIT)
    if set(original["development_seqn"]) & set(original["test_seqn"]):
        raise RuntimeError("Original split itself contains overlapping IDs")
    return original


def membership_assertions(development, heldout, original):
    """Fail closed if any participant crosses the ORIGINAL split boundary."""
    dev, test = set(development), set(heldout)
    old_dev, old_test = set(original["development_seqn"]), set(original["test_seqn"])
    checks = {
        "development_age20_intersection_original_test_is_empty": not (dev & old_test),
        "heldout_age20_intersection_original_development_is_empty": not (test & old_dev),
        "development_age20_intersection_heldout_age20_is_empty": not (dev & test),
        "all_heldout_age20_from_original_heldout": test <= old_test,
        "all_development_age20_from_original_development": dev <= old_dev,
        "unique_development_ids": len(development) == len(dev),
        "unique_heldout_ids": len(heldout) == len(test),
    }
    for name, passed in checks.items():
        if not passed:  # Keep the check active even if Python is started with -O.
            raise AssertionError(name)
        assert passed, name
    return checks


def load_final_split():
    split = load_json(FINAL_SPLIT)
    if split.get("protocol") != PROTOCOL_VERSION or split.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Only original_split_preserved_age20 is allowed for active experiments")
    if load_json(ARTIFACTS_DIR / "split_indices.json") != split:
        raise RuntimeError("Active split alias differs from the final-protocol split")
    if sha256_file(ORIGINAL_SPLIT) != split["original_split_sha256"]:
        raise RuntimeError("Original split provenance changed")
    if sha256_file(PROCESSED_DATA_DIR / "cohort.csv") != split["cohort_sha256"]:
        raise RuntimeError("Cohort changed since membership was restored")
    membership_assertions(split["development_seqn"], split["test_seqn"], load_original_split())
    return split
