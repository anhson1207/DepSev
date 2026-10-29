"""Snapshot the completed pre-age20 protocol before the authorized correction."""

import hashlib
import shutil

from config import ARTIFACTS_DIR, PROJECT_ROOT
from utils import save_json


def main():
    archive = ARTIFACTS_DIR / "archive" / "pre_age20_v1"
    if archive.exists():
        raise RuntimeError(f"Archive already exists; will not replace it: {archive}")
    paths = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "requirements.txt",
             PROJECT_ROOT / "data/processed/cohort.csv"]
    for pattern in ["src/*.py", "tests/*.py", "artifacts/*.json",
                    "results/*.md", "results/tables/*.csv", "results/figures/*.png"]:
        paths.extend(sorted(PROJECT_ROOT.glob(pattern)))
    manifest = {}
    for source in paths:
        relative = source.relative_to(PROJECT_ROOT)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        if original != hashlib.sha256(destination.read_bytes()).hexdigest():
            raise RuntimeError(f"Archive verification failed: {relative}")
        manifest[str(relative)] = original
    save_json({"reason": "User-authorized age >= 20 eligibility correction",
               "files_sha256": manifest}, archive / "archive_manifest.json")
    # Retire locks recoverably only AFTER every original has a verified copy.
    retired = archive / "retired_locks"
    retired.mkdir()
    for name in ["cv_folds.json", "ablation_config.json"]:
        (ARTIFACTS_DIR / name).rename(retired / name)
    print(f"Archived {len(manifest)} files: {archive}")
    print("Old CV/config locks retired; old split/cohort remain until the rebuild.")


if __name__ == "__main__":
    main()
