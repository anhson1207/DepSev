"""Run the project suite without rewriting legacy outputs or fitting XGBoost.

One legacy XGB fit test is explicitly skipped under the current RF-only request.
The legacy split idempotence test is exercised with write attempts intercepted:
their proposed JSON/CSV must equal the saved artifact, which is never rewritten.
"""

import argparse
import importlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import multicycle_protocol as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-name", required=True)
    args = parser.parse_args()
    before = p.preserve_legacy()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    legacy_xgb = importlib.import_module("test_tuning_estimator")
    method = legacy_xgb.TuningEstimatorTests.test_weights_and_preprocessing_on_real_inner_training_only
    method.__unittest_skip__ = True
    method.__unittest_skip_why__ = "Current request forbids XGB fitting; legacy artifact is unchanged"
    restore = importlib.import_module("02_restore_original_split")
    original_csv = pd.DataFrame.to_csv
    intercepted = []

    def check_json_only(payload, path):
        path = Path(path)
        if not path.exists() or json.loads(path.read_text()) != payload:
            raise AssertionError(f"Legacy test proposed a changed JSON: {path}")
        intercepted.append(str(path.relative_to(ROOT)))

    def protect_csv(frame, path_or_buf=None, *args, **kwargs):
        if isinstance(path_or_buf, (str, Path)):
            path = Path(path_or_buf).resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(p.OUT) and not path.is_relative_to(p.ART):
                proposed = original_csv(frame, None, *args, **kwargs)
                if not path.exists() or path.read_text() != proposed:
                    raise AssertionError(f"Legacy test proposed a changed CSV: {path}")
                intercepted.append(str(path.relative_to(ROOT)))
                return None
        return original_csv(frame, path_or_buf, *args, **kwargs)

    stream = io.StringIO()
    with patch.object(restore, "save_json", check_json_only), patch.object(pd.DataFrame, "to_csv", protect_csv):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert p.preserve_legacy() == before
    log = stream.getvalue() + "\nLegacy writes intercepted and compared only: " + repr(intercepted) + "\n"
    path = p.ART / args.log_name
    with path.open("x", encoding="utf-8") as output:
        output.write(log)
    print(log, flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
