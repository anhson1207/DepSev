"""Independent protocol, leakage, sampling, and saved-result regression tests."""

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import multicycle_protocol as p


class MulticycleProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames, cls.policy, cls.old = p.load_inputs()

    def test_target_mapping_all_28_scores(self):
        scores = np.arange(28)
        for task, boundaries in [("T2", [5]), ("T3", [5, 10]), ("T4", [5, 10, 15]), ("T5", [5, 10, 15, 20])]:
            expected = [sum(score >= cutoff for cutoff in boundaries) for score in scores]
            np.testing.assert_array_equal(p.target_values(scores, task), expected)
        self.assertEqual(p.TASKS["T2"]["labels"], ["minimal symptoms", "symptomatic"])
        for invalid in [[-1], [28], [np.nan], [1.5]]:
            with self.assertRaises(ValueError):
                p.target_values(invalid, "T5")

    def test_common31_typing_and_no_leakage(self):
        X = p.predictor_frame(self.frames["2013_2014"], self.policy)
        self.assertEqual(X.shape, (5054, 31))
        self.assertEqual(list(X), self.policy["ordered_features"])
        self.assertFalse(set(X) & set(self.policy["metadata_not_predictors"]))
        self.assertFalse(any(name.startswith("DPQ") for name in X))
        pipeline = p.make_pipeline(self.policy, self.old)
        self.assertEqual(pipeline.named_steps["model"].get_params(), self.old["hyperparameters"])
        for key, value in p.RF.items():
            self.assertEqual(pipeline.named_steps["model"].get_params()[key], value)

    def test_matched_counts_no_replacement_no_control_cycle(self):
        pooled = self.frames["2011_2016_common31"]
        pool = pooled.loc[~pooled.cycle.eq("2013_2014")].reset_index(drop=True)
        counts = self.frames["2013_2014"].severity.value_counts().sort_index().to_numpy()
        self.assertEqual(counts.tolist(), [3764, 800, 303, 132, 55])
        memberships = []
        for seed in p.SEEDS:
            sample = p.matched_sample(pool, counts, seed)
            self.assertEqual(len(sample), 5054)
            self.assertTrue(p.participant_keys(sample).is_unique)
            self.assertFalse(sample.cycle.eq("2013_2014").any())
            np.testing.assert_array_equal(sample.severity.value_counts().sort_index(), counts)
            self.assertTrue(sample.groupby("severity").cycle.nunique().eq(2).all())
            pd.testing.assert_frame_equal(sample, p.matched_sample(pool, counts, seed))
            memberships.append(p.membership_hash(sample))
        self.assertEqual(len(set(memberships)), 10)

    def test_temporal_disjointness(self):
        module = importlib.import_module("14_temporal_generalization")
        train, test = module.temporal_partition(self.frames)
        self.assertEqual((len(train), len(test)), (9693, 4888))
        self.assertFalse(set(p.participant_keys(train)) & set(p.participant_keys(test)))

    def test_preprocessing_learns_only_train_with_unseen_validation_values(self):
        raw = self.frames["2013_2014"].iloc[:40]
        train, valid = p.predictor_frame(raw.iloc[:30], self.policy), p.predictor_frame(raw.iloc[30:], self.policy)
        numeric = [r["feature"] for r in self.policy["columns"] if r["feature_type"] == "numerical"]
        categorical = [r["feature"] for r in self.policy["columns"] if r["feature_type"] == "categorical"]
        train.loc[train.index[0], numeric[0]] = np.nan
        valid.loc[:, numeric[0]] = 99999
        valid.loc[:, categorical[0]] = "validation_only_category"
        pre = p.make_pipeline(self.policy, self.old).named_steps["preprocessor"]
        pre.fit(train)
        num = pre.named_transformers_["numerical"]
        cat = pre.named_transformers_["categorical"]
        np.testing.assert_allclose(num["imputer"].statistics_, train[numeric].median())
        means = num["scaler"].mean_.copy()
        self.assertEqual(num["scaler"].n_samples_seen_, len(train))
        self.assertNotIn("validation_only_category", cat["encoder"].categories_[0])
        pre.transform(valid)
        np.testing.assert_array_equal(means, num["scaler"].mean_)

    def test_fixed_folds_cover_every_participant_once(self):
        for key in ["2013_2014", "2011_2016_common31"]:
            frame = self.frames[key]
            assignments = p.fixed_folds(frame)
            np.testing.assert_array_equal(assignments, p.fixed_folds(frame))
            for label in range(5):
                counts = np.bincount(assignments[frame.severity.eq(label)])[1:]
                self.assertLessEqual(counts.max() - counts.min(), 1)


class MulticycleResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (p.ART / "summary_complete.json").exists():
            raise unittest.SkipTest("Saved-result tests run after all A/B/C/D outputs exist")
        cls.frames, cls.policy, cls.old = p.load_inputs()
        cls.records = importlib.import_module("multicycle_outputs").saved_records()

    def test_all_saved_predictions_and_metrics_recompute(self):
        for record in self.records:
            summary = record["summary"]
            name = f"{summary['experiment']}_{summary['sampling_seed']}_{summary['target']}"
            rows = pd.read_csv(p.ART / "checkpoints" / f"{name}_predictions.csv")
            self.assertFalse(rows.duplicated(["cycle", "SEQN"]).any())
            k = int(summary["target"][-1])
            probabilities = rows[[f"prob_{i}" for i in range(k)]].to_numpy()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_array_equal(probabilities.argmax(axis=1), rows.y_pred)
            truth = self.frames["2011_2016_common31"].set_index(["cycle", "SEQN"]).loc[list(zip(rows.cycle, rows.SEQN)), "PHQ9_total"]
            expected = np.digitize(truth, p.TASKS[summary["target"]]["cutoffs"])
            np.testing.assert_array_equal(rows.y_true, expected)
            for fold in record["folds"]:
                subset = rows.loc[rows.fold.eq(fold["fold"])]
                independent = {
                    "macro_f1": f1_score(subset.y_true, subset.y_pred, average="macro", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(subset.y_true, subset.y_pred),
                    "accuracy": accuracy_score(subset.y_true, subset.y_pred),
                    "macro_precision": precision_score(subset.y_true, subset.y_pred, average="macro", zero_division=0),
                    "macro_recall": recall_score(subset.y_true, subset.y_pred, average="macro", zero_division=0),
                    "weighted_f1": f1_score(subset.y_true, subset.y_pred, average="weighted", zero_division=0)}
                for metric, value in independent.items():
                    self.assertAlmostEqual(value, fold[metric])
            matrix = confusion_matrix(rows.y_true, rows.y_pred, labels=np.arange(k))
            for cls in record["per_class"]:
                code = cls["class_index"]
                self.assertEqual(cls["support"], matrix[code].sum())
                self.assertEqual(cls["predicted_count"], matrix[:, code].sum())
                self.assertEqual(cls["true_positives"], matrix[code, code])
                self.assertAlmostEqual(cls["recall"], matrix[code, code] / matrix[code].sum())

    def test_same_folds_across_targets_and_manifest(self):
        assignments = pd.read_csv(p.TABLES / "cv_fold_assignments.csv")
        for experiment in ["A", "B", "C"]:
            predictions = pd.read_csv(p.TABLES / f"{experiment}_oof_predictions.csv")
            for (seed, task), rows in predictions.groupby(["sampling_seed", "target"]):
                expected = assignments.loc[assignments.experiment.eq(experiment) & assignments.sampling_seed.eq(seed)]
                pd.testing.assert_frame_equal(rows[["cycle", "SEQN", "fold"]].reset_index(drop=True), expected[["cycle", "SEQN", "fold"]].reset_index(drop=True))

    def test_C_aggregation_uses_ten_sample_means(self):
        matched = pd.read_csv(p.TABLES / "matched_n_metrics.csv")
        comparison = pd.read_csv(p.TABLES / "comparison_summary.csv")
        self.assertEqual(len(matched), 40)
        for task in p.TASKS:
            part = matched.loc[matched.target.eq(task)]
            self.assertEqual(set(part.sampling_seed), set(p.SEEDS))
            for metric in ["macro_f1", "balanced_accuracy"]:
                row = comparison.loc[comparison.target.eq(task) & comparison.metric.eq(metric)].iloc[0]
                self.assertAlmostEqual(row.C_matched, part[f"{metric}_mean"].mean())
                self.assertAlmostEqual(row.C_matched_sd, part[f"{metric}_mean"].std(ddof=1))

    def test_actual_fit_statistics_and_temporal_scope(self):
        audits = p.read_json(p.ART / "preprocessing_fit_audit.json")
        self.assertEqual(len(audits), 244)
        pooled = self.frames["2011_2016_common31"].set_index(["cycle", "SEQN"])
        assignments = pd.read_csv(p.TABLES / "cv_fold_assignments.csv")
        for audit in audits:
            if audit["experiment"] == "D":
                train = pd.concat([self.frames[c] for c in p.CYCLES[:2]], ignore_index=True)
                self.assertEqual(audit["validation_n"], 4888)
            else:
                ids = assignments.loc[assignments.experiment.eq(audit["experiment"]) & assignments.sampling_seed.eq(audit["sampling_seed"]) & assignments.fold.ne(audit["fold"])]
                train = pooled.loc[list(zip(ids.cycle, ids.SEQN))].reset_index()
            self.assertEqual(p.membership_hash(train), audit["train_membership_hash"])
            np.testing.assert_allclose(train[audit["numerical_columns"]].median(), audit["imputer_medians"])
            self.assertEqual(len(train), audit["scaler_n_samples_seen"])
            self.assertEqual(audit["RF_config"], self.old["hyperparameters"])

    def test_preservation_and_output_hashes(self):
        self.assertGreater(p.preserve_legacy(), 800)
        config = p.read_json(p.ART / "experiment_config.json")
        p.verify_hashes(config["script_hashes"])
        p.verify_hashes(config["input_dataset_hashes"])
        for name in ["cv_complete.json", "temporal_complete.json", "summary_complete.json"]:
            p.verify_hashes(p.read_json(p.ART / name)["output_sha256"])


if __name__ == "__main__":
    unittest.main()
