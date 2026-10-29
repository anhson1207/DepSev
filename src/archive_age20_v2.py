"""Preserve the completed reshuffled exploratory protocol before restoring membership."""

import hashlib
import shutil

from config import ARTIFACTS_DIR, PROJECT_ROOT
from utils import load_json, save_json


def main():
    archive = ARTIFACTS_DIR / "archive/age20_v2"
    if archive.exists():
        raise RuntimeError("age20_v2 archive already exists; do not overwrite it")
    split = load_json(ARTIFACTS_DIR / "split_indices.json")
    if split.get("protocol_version") != "age20_v2":
        raise RuntimeError("Only the active age20_v2 exploratory run may be archived here")
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
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != hashlib.sha256(destination.read_bytes()).hexdigest():
            raise RuntimeError(f"Archive verification failed: {relative}")
        manifest[str(relative)] = digest
    save_json({"protocol": "age20_v2", "status": "retired_exploratory",
               "reason": "Restore original split membership with age >= 20 filter only",
               "files_sha256": manifest}, archive / "archive_manifest.json")
    print(f"Verified snapshot: {len(manifest)} files in {archive}")
    print("No exploratory result removed or overwritten.")


if __name__ == "__main__":
    main()
