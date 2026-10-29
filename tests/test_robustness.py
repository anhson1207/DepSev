"""Audit the final-protocol membership and repeated CV without heldout prediction."""

import hashlib
import json
from importlib import import_module
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import ARTIFACTS_DIR, TABLES_DIR, experiment_path
from protocol import FINAL_SPLIT, load_final_split, load_original_split, membership_assertions

ABLATION = import_module("05_feature_ablation")


def read_table(name):
    return pd.read_csv(experiment_path(TABLES_DIR, name))


class RobustnessProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev, _ = ABLATION.load_inputs()
        cls.split = load_final_split()
        cls.original = load_original_split()
        cls.fixed = json.loads(experiment_path(ARTIFACTS_DIR, "cv_folds.json").read_text())
        cls.repeated = json.loads(experiment_path(ARTIFACTS_DIR, "repeated_cv_folds.json").read_text())
        cls.oof = read_table("repeated_cv_oof_predictions.csv")

    def test_archive_and_exploratory_results_preserved(self):
        # Hash only: do NOT inspect exploratory predictions containing restored test IDs.
        for name in ["pre_age20_v1", "age20_v2"]:
            archive = ARTIFACTS_DIR / "archive" / name
            manifest = json.loads((archive / "archive_manifest.json").read_text())
            for relative, expected in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256((archive / relative).read_bytes()).hexdigest(), expected)
                if name == "age20_v2" and relative.startswith("results/"):
                    self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        config = json.loads(experiment_path(ARTIFACTS_DIR, "robustness_config.json").read_text())
        for filename, expected in config["inputs_and_code_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), expected)

    def test_exact_membership_is_original_split_filtered_by_raw_age(self):
        ages = pd.read_csv(ROOT / "data/raw/demographic.csv", usecols=["SEQN", "RIDAGEYR"])
        eligible = set(ages.loc[ages.RIDAGEYR.ge(20), "SEQN"])
        for key in ["development_seqn", "test_seqn"]:
            self.assertEqual(self.split[key], [i for i in self.original[key] if i in eligible])
        checks = membership_assertions(self.split["development_seqn"], self.split["test_seqn"], self.original)
        self.assertTrue(all(checks.values()))
        self.assertEqual(self.split["protocol"], "original_split_preserved_age20")
        self.assertFalse(self.split["new_random_train_test_split"])
        self.assertEqual(len(self.dev), 4045)
        self.assertEqual(len(self.split["test_seqn"]), 1009)
        self.assertTrue(self.dev.RIDAGEYR.ge(20).all())
        pd.testing.assert_series_equal(
            self.dev.severity.value_counts().sort_index(),
            pd.Series([3016, 638, 243, 104, 44], index=pd.Index(range(5), name="severity"), name="count"))
        history = self.split["historical_exposure"]
        retired = json.loads((ARTIFACTS_DIR / "archive/age20_v2/artifacts/split_indices.json").read_text())
        exposed = set(self.split["test_seqn"]) & set(retired["development_seqn"])
        self.assertEqual(len(exposed), 813)
        self.assertEqual(set(history["heldout_age20_previously_used_in_age20_v2_development_seqn"]), exposed)
        self.assertFalse(history["historically_untouched"])
        self.assertFalse(history["heldout_evaluation_performed"])

    def test_membership_assertion_rejects_real_cross_boundary_ids(self):
        original_test_id = self.original["test_seqn"][0]
        with self.assertRaises(AssertionError):
            membership_assertions([*self.split["development_seqn"], original_test_id],
                                  self.split["test_seqn"], self.original)

    def test_saved_class_counts_match_requested_partition_sizes(self):
        counts = read_table("class_distribution.csv")
        self.assertEqual(counts["class"].tolist(), ["C0", "C1", "C2", "C3", "C4"])
        self.assertEqual(counts.development_count.tolist(), [3016, 638, 243, 104, 44])
        self.assertEqual(counts.heldout_count.tolist(), [748, 162, 60, 28, 11])
        self.assertEqual(counts.development_count.sum(), len(self.dev))
        self.assertEqual(counts.heldout_count.sum(), len(self.split["test_seqn"]))

    def test_repeated_assignments_and_repeat_one_predictions(self):
        self.assertEqual(self.repeated["development_seqn"], self.dev.SEQN.astype(int).tolist())
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
        for index, (train, validation) in enumerate(splitter.split(self.dev, self.dev.severity)):
            assignments = np.asarray(self.repeated["validation_fold_by_repeat"][index // 5])
            np.testing.assert_array_equal(np.flatnonzero(assignments == index % 5 + 1), validation)
            self.assertFalse(set(train) & set(validation))
            self.assertEqual(set(self.dev.severity.iloc[validation]), set(range(5)))
        np.testing.assert_array_equal(self.repeated["validation_fold_by_repeat"][0], self.fixed["validation_fold"])
        self.assertEqual(self.repeated["split_sha256"], hashlib.sha256(FINAL_SPLIT.read_bytes()).hexdigest())
        primary = read_table("ablation_oof_predictions.csv")
        primary = primary[primary.model.eq("XGBoost") & primary.feature_set.isin(["F3", "F4"])]
        first = self.oof[self.oof["repeat"].eq(1)]
        columns = ["feature_set", "SEQN", "fold", "y_true", "y_pred"]
        pd.testing.assert_frame_equal(
            first[columns].sort_values(columns[:2]).reset_index(drop=True),
            primary[columns].sort_values(columns[:2]).reset_index(drop=True))

    def test_all_oof_and_metrics_use_only_original_development(self):
        folds = read_table("repeated_cv_fold_results.csv")
        classes = read_table("repeated_cv_per_class.csv")
        summary = read_table("repeated_cv_summary.csv")
        paired = read_table("repeated_cv_paired_differences.csv")
        self.assertEqual(len(self.oof), 20 * len(self.dev))
        self.assertEqual(len(folds), 100)
        self.assertEqual(len(paired), 50)
        self.assertFalse(set(self.oof.SEQN) & set(self.original["test_seqn"]))
        expected_y = self.dev.set_index("SEQN").severity
        for (repeat, feature_set), rows in self.oof.groupby(["repeat", "feature_set"]):
            self.assertEqual(set(rows.SEQN), set(self.dev.SEQN))
            self.assertTrue(rows.SEQN.is_unique)
            mapping = dict(zip(self.dev.SEQN, self.repeated["validation_fold_by_repeat"][repeat - 1]))
            np.testing.assert_array_equal(rows.fold, rows.SEQN.map(mapping))
            np.testing.assert_array_equal(rows.y_true, rows.SEQN.map(expected_y))
            for fold, predictions in rows.groupby("fold"):
                metrics = ABLATION.summarize_metrics(predictions.y_true, predictions.y_pred)
                saved = folds[folds["repeat"].eq(repeat) & folds.feature_set.eq(feature_set) & folds.fold.eq(fold)].iloc[0]
                for metric, value in metrics.items():
                    self.assertAlmostEqual(value, saved[metric])
            _, recall, f1, support = precision_recall_fscore_support(
                rows.y_true, rows.y_pred, labels=np.arange(5), zero_division=0)
            saved_classes = classes[classes["repeat"].eq(repeat) & classes.feature_set.eq(feature_set)].sort_values("class")
            np.testing.assert_allclose(saved_classes.recall, recall)
            np.testing.assert_allclose(saved_classes.f1, f1)
            np.testing.assert_array_equal(saved_classes.support, support)
        for feature_set, rows in folds.groupby("feature_set"):
            saved = summary[summary.feature_set.eq(feature_set)].iloc[0]
            for statistic in ["mean", "median", "min", "max"]:
                self.assertAlmostEqual(saved[f"macro_f1_{statistic}"], getattr(rows.macro_f1, statistic)())
            self.assertAlmostEqual(saved.macro_f1_std, rows.macro_f1.std(ddof=1))
        pivot = folds.pivot(index=["repeat", "fold"], columns="feature_set", values="macro_f1")
        np.testing.assert_allclose(paired.delta_F4_minus_F3, pivot.F4 - pivot.F3)

    def test_balanced_weights_only_use_each_training_fold(self):
        weights = read_table("repeated_cv_weights.csv")
        self.assertEqual(len(weights), 500)
        for (repeat, feature_set, fold), rows in weights.groupby(["repeat", "feature_set", "fold"]):
            assignments = np.array(self.repeated["validation_fold_by_repeat"][repeat - 1])
            y_train = self.dev.loc[assignments != fold, "severity"]
            for row in rows.to_dict("records"):
                count = int(y_train.eq(int(row["class"][-1])).sum())
                self.assertEqual(row["class_count_train"], count)
                self.assertAlmostEqual(row["sample_weight"], len(y_train) / (5 * count))
                self.assertAlmostEqual(row["class_weight_sum"], len(y_train) / 5)

    def test_retired_runners_stop_before_reading_participant_data(self):
        for module_name in ["02_build_cohort", "03_eda", "04_train_baselines"]:
            runner = import_module(module_name)
            # Any accidental dataframe load would fail this test before the guard.
            with patch("pandas.read_csv", side_effect=AssertionError("Participant data read by legacy runner")):
                with self.assertRaises(RuntimeError):
                    runner.main()
        robustness = import_module("05_robustness")
        with self.assertRaisesRegex(RuntimeError, "Exploratory sensitivity is retired"):
            robustness.sensitivity_experiment(None, None, None, None, None, None, None)

    def test_restore_is_idempotent_and_never_invokes_random_split(self):
        before = hashlib.sha256(FINAL_SPLIT.read_bytes()).hexdigest()
        runner = import_module("02_restore_original_split")
        with patch("sklearn.model_selection.train_test_split", side_effect=AssertionError("New random split")):
            runner.main()
        self.assertEqual(hashlib.sha256(FINAL_SPLIT.read_bytes()).hexdigest(), before)
        self.assertEqual(load_final_split(), self.split)


if __name__ == "__main__":
    unittest.main()
