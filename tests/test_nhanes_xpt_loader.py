"""Reader regression tests: verified zeros, missing values and real PHQ parity."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nhanes_official_utils import RAW, ROOT, sha256
from nhanes_xpt_loader import IBM_ZERO_ARTIFACT, load_nhanes_xpt, normalize_verified_zeros


class ZeroNormalizationTests(unittest.TestCase):
    def test_only_verified_bytes_corrected(self):
        original = pd.Series([IBM_ZERO_ARTIFACT, IBM_ZERO_ARTIFACT, np.nan, 0., 1e-80, 1.])
        result, count = normalize_verified_zeros(original, [True, False, False, True, False, False])
        self.assertEqual(count, 1)
        self.assertEqual(result.iloc[0], 0.)
        self.assertEqual(result.iloc[1], IBM_ZERO_ARTIFACT)
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertEqual(result.iloc[4], 1e-80)
        self.assertEqual(original.iloc[0], IBM_ZERO_ARTIFACT)

    def test_unexpected_zero_decoding_fails_closed(self):
        with self.assertRaises(ValueError):
            normalize_verified_zeros(pd.Series([1.]), [True])
        with self.assertRaises(ValueError):
            normalize_verified_zeros(pd.Series([0.]), [True, False])

    def test_real_phq_zero_counts_and_kaggle_parity(self):
        path = RAW / "2013_2014/xpt/DPQ_H.xpt"
        digest = sha256(path)
        columns = ["SEQN", *[f"DPQ{i:03d}" for i in range(10, 100, 10)]]
        frame, audit = load_nhanes_xpt(path, columns)
        self.assertEqual(int(frame.DPQ010.eq(0).sum()), 3976)
        self.assertEqual(int(frame.DPQ010.isna().sum()), 526)
        self.assertEqual(sha256(path), digest)
        for row in audit[1:]:
            self.assertEqual(row["zero_count_after"], row["codebook_zero_count_if_explicit"])
            self.assertTrue(frame[row["variable"]].dropna().isin([0, 1, 2, 3, 7, 9]).all())
        kaggle = pd.read_csv(ROOT / "data/raw/questionnaire.csv", usecols=columns).set_index("SEQN")
        pd.testing.assert_frame_equal(frame.set_index("SEQN"), kaggle.loc[frame.SEQN],
                                      check_dtype=False, check_index_type=False)

    def test_unaudited_variable_is_rejected(self):
        with self.assertRaises(ValueError):
            load_nhanes_xpt(RAW / "2013_2014/xpt/DPQ_H.xpt", ["DPQ100"])

    def test_changed_hash_is_rejected_without_touching_raw(self):
        with patch("nhanes_xpt_loader.sha256", return_value="not_the_audited_hash"):
            with self.assertRaisesRegex(ValueError, "Raw hash differs"):
                load_nhanes_xpt(RAW / "2013_2014/xpt/DPQ_H.xpt", ["SEQN", "DPQ010"])


if __name__ == "__main__":
    unittest.main()
