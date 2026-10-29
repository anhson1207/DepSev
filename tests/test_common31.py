"""Common-core deterministic rule tests; no training or generated research data."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common31_features import (PHQ, QUANTITY, REMOVED, alcohol_features, build_cohort,
                               common_schema, validate_ids)
from nhanes_official_utils import ROOT


class Common31Tests(unittest.TestCase):
    def test_schema_is_frozen_f4_minus_three_and_quantity_rename(self):
        schema = common_schema()
        features = [row["feature"] for row in schema]
        self.assertEqual(len(features), 31)
        self.assertFalse(set(features) & set(REMOVED))
        self.assertNotIn("SLD012", features)
        self.assertIn(QUANTITY, features)
        self.assertNotIn("cycle", features)
        self.assertEqual([sum(row["group"] == group for row in schema) for group in "PCLD"], [10, 9, 7, 5])

    def test_quantity_cap_and_missing_states(self):
        # Boundary cases for a pure function, not a generated experiment dataset.
        raw = pd.DataFrame({"ALQ101": [1, 1, 1, 2, 1, 1, 1, 1],
                            "ALQ110": [np.nan, np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan],
                            "ALQ120Q": [10, 10, 10, np.nan, 0, 10, 10, 10],
                            "ALQ130": [14, 15, 95, np.nan, np.nan, np.nan, 777, 999]})
        status, quantity, state, reason = alcohol_features(raw, "2013_2014")
        self.assertEqual(quantity.iloc[:5].tolist(), [14, 15, 15, 0, 0])
        self.assertTrue(quantity.iloc[5:].isna().all())
        self.assertEqual(state.tolist(), ["observed"] * 3 + ["not_applicable"] * 2 + ["unknown", "nonresponse", "nonresponse"])
        self.assertEqual(status.iloc[3], "under_12_lifetime_quantity_not_asked")
        self.assertIn("ALQ130=777", reason.iloc[6])
        self.assertIn("ALQ130=999", reason.iloc[7])
        with self.assertRaises(ValueError):
            alcohol_features(raw, "2015_2016")

    def test_inconsistent_routing_is_not_imputed(self):
        raw = pd.DataFrame({"ALQ101": [2], "ALQ110": [2], "ALQ120Q": [np.nan], "ALQ130": [2]})
        with self.assertRaises(ValueError):
            alcohol_features(raw, "2011_2012")

    def test_duplicate_participants_fail(self):
        with self.assertRaises(ValueError):
            validate_ids(pd.DataFrame({"SEQN": [1, 1]}))

    def test_real_kaggle_cohort_matches_existing_membership_and_target(self):
        schema = common_schema()
        import json
        deps = json.loads((ROOT / "artifacts/nhanes_official_feature_dependencies.json").read_text())["features"]
        sources = {v: row["source_file"] for row in deps for v in row["source_variables"]}
        sources.update({v: "questionnaire.csv" for v in PHQ})
        variables = set(PHQ) | {v for row in schema for v in row["source_variables"]}
        raw = None
        for name in ["demographic.csv", "examination.csv", "labs.csv", "questionnaire.csv"]:
            cols = [v for v in variables if sources[v] == name]
            part = pd.read_csv(ROOT / "data/raw" / name, usecols=["SEQN", *cols], float_precision="round_trip")
            raw = part if raw is None else raw.merge(part, on="SEQN", how="left", validate="one_to_one")
        cohort, stages, routing = build_cohort(raw, "2013_2014", schema)
        old = pd.read_csv(ROOT / "data/processed/cohort.csv").set_index("SEQN")
        current = cohort.set_index("SEQN")
        self.assertEqual(len(current), 5054)
        self.assertEqual(set(current.index), set(old.index))
        for column in ["PHQ9_total", "severity", "SBP_mean", "DBP_mean"]:
            pd.testing.assert_series_equal(current[column], old.loc[current.index, column],
                                            check_dtype=False, check_exact=False, rtol=1e-14, atol=0)
        for total, severity in [(0, 0), (4, 0), (5, 1), (9, 1), (10, 2), (14, 2), (15, 3), (19, 3), (20, 4), (27, 4)]:
            self.assertTrue(cohort.loc[cohort.PHQ9_total.eq(total), "severity"].eq(severity).all())
        self.assertEqual(stages["cohort_n"], 5054)
        self.assertGreater(routing["quantity_not_applicable_n"], 0)

    def test_exported_cohorts_reconcile_independently(self):
        directory = ROOT / "data/processed_multicycle"
        if not (directory / "cohort_2011_2016_common31.csv").exists():
            self.skipTest("Run the common31 builder before export integration checks")
        summary = pd.read_csv(ROOT / "results/tables/multicycle_class_distribution.csv").set_index("cycle")
        missing = pd.read_csv(ROOT / "results/tables/multicycle_common31_missingness.csv")
        parts = []
        for cycle in ["2011_2012", "2013_2014", "2015_2016"]:
            frame = pd.read_csv(directory / f"cohort_{cycle}.csv", float_precision="round_trip")
            self.assertFalse(frame.SEQN.duplicated().any())
            self.assertTrue(frame.RIDAGEYR.ge(20).all())
            self.assertTrue(frame[PHQ].isin([0, 1, 2, 3]).all().all())
            self.assertTrue(frame.PHQ9_total.eq(frame[PHQ].sum(axis=1)).all())
            # Independent ordinal mapping avoids reusing the builder's pd.cut.
            expected = np.searchsorted([5, 10, 15, 20], frame.PHQ9_total, side="right")
            np.testing.assert_array_equal(frame.severity, expected)
            self.assertEqual(len(frame), summary.loc[cycle, "cohort_n"])
            for label in range(5):
                self.assertEqual(frame.severity.eq(label).sum(), summary.loc[cycle, f"C{label}_n"])
            for row in missing.loc[missing.cycle.eq(cycle)].itertuples():
                self.assertEqual(frame[row.feature].isna().sum(), row.n_missing)
                self.assertAlmostEqual(100 * row.n_missing / len(frame), row.missing_percent)
            parts.append(frame)
        pooled = pd.read_csv(directory / "cohort_2011_2016_common31.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(pooled, pd.concat(parts, ignore_index=True))
        self.assertEqual(len(pooled), 14581)
        self.assertFalse(pooled.SEQN.duplicated().any())


if __name__ == "__main__":
    unittest.main()
