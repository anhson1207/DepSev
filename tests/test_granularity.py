"""Independent checks of the development-only granularity experiment."""

from importlib import import_module
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
G = import_module("09_granularity_experiment")
O = import_module("10_class_overlap_analysis")


class GranularityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.inventory, cls.total, cls.folds, cls.assignment, cls.spec = G.load_development()
        cls.oof = pd.read_csv(G.TABLES_DIR / "granularity_oof_predictions.csv")
        cls.scores = pd.read_csv(G.TABLES_DIR / "granularity_fold_results.csv")
        cls.summary = pd.read_csv(G.TABLES_DIR / "granularity_results.csv")

    def test_only_development_and_original_folds(self):
        split = G.load_final_split()
        for task, rows in self.oof.groupby("task"):
            self.assertEqual(len(rows), 4045)
            self.assertTrue(rows.SEQN.is_unique)
            self.assertEqual(set(rows.SEQN), set(split["development_seqn"]))
            self.assertFalse(set(rows.SEQN) & set(split["test_seqn"]))
            rows = rows.set_index("SEQN").loc[self.X.index]
            np.testing.assert_array_equal(rows.fold, self.assignment)
            expected = np.minimum(self.y, int(task[-1]) - 1)
            np.testing.assert_array_equal(rows.y_true, expected)
            n = int(task[-1])
            probs = rows[[f"prob_{i}" for i in range(n)]].to_numpy()
            np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_array_equal(probs.argmax(axis=1), rows.y_pred)

    def test_distribution_from_score_cutoffs(self):
        distributions = pd.read_csv(G.TABLES_DIR / "granularity_class_distribution.csv")
        for task, cutoffs in [("T2", [5]), ("T3", [5, 10]), ("T4", [5, 10, 15]), ("T5", [5, 10, 15, 20])]:
            expected = np.bincount(np.digitize(self.total, cutoffs))
            rows = distributions[distributions.task.eq(task)].sort_values("class_index")
            np.testing.assert_array_equal(rows.n, expected)
            np.testing.assert_allclose(rows.percentage, expected / 4045 * 100)

    def test_fold_scores_and_sample_sd_recalculate(self):
        for row in self.scores.itertuples():
            sample = self.oof[self.oof.task.eq(row.task) & self.oof.fold.eq(row.fold)]
            self.assertAlmostEqual(row.accuracy, accuracy_score(sample.y_true, sample.y_pred))
            self.assertAlmostEqual(row.macro_f1, f1_score(sample.y_true, sample.y_pred, average="macro"))
            if row.task == "T2":
                self.assertTrue(np.isnan(row.within_one_class_accuracy))
            else:
                error = np.abs(sample.y_true - sample.y_pred)
                self.assertAlmostEqual(row.mean_absolute_class_error, error.mean())
                self.assertAlmostEqual(row.within_one_class_accuracy, (error <= 1).mean())
        for row in self.summary.itertuples():
            scores = self.scores[self.scores.task.eq(row.task)]
            for metric in G.METRICS:
                if row.task == "T2" and metric in G.METRICS[-2:]:
                    continue
                self.assertAlmostEqual(getattr(row, f"{metric}_mean"), scores[metric].mean())
                self.assertAlmostEqual(getattr(row, f"{metric}_std"), scores[metric].std(ddof=1))

    def test_actual_fit_partitions_and_imputer_statistics(self):
        fits = json.loads((G.ARTIFACTS_DIR / "granularity_fit_audit.json").read_text())
        self.assertEqual(len(fits), 20)
        numerical = self.spec["preprocessing"]["numerical"]["columns"]
        for fit in fits:
            train, valid = self.folds[fit["fold"] - 1]
            self.assertEqual(fit["training_ids_sha256"], G.id_hash(self.X.index[train]))
            self.assertEqual(fit["validation_ids_sha256"], G.id_hash(self.X.index[valid]))
            self.assertEqual(fit["scaler_n_samples_seen"], len(train))
            np.testing.assert_allclose(fit["numerical_imputer_medians"], self.X.iloc[train][numerical].median())
            self.assertEqual(fit["RF_config_sha256"], G.frozen.json_hash(self.spec["hyperparameters"]))

    def test_T5_reproduces_preexisting_RF(self):
        old = pd.read_csv(G.TABLES_DIR / "ensemble_oof_predictions.csv")
        old = old[old.model.eq("RF")].set_index("SEQN").sort_index()
        new = self.oof[self.oof.task.eq("T5")].set_index("SEQN").sort_index()
        np.testing.assert_array_equal(new[["fold", "y_true", "y_pred"]], old[["fold", "y_true", "y_pred"]])
        np.testing.assert_allclose(new[[f"prob_{i}" for i in range(5)]],
                                   old[[f"prob_C{i}" for i in range(5)]], rtol=0, atol=1e-14)

    def test_adjacent_counts_and_effect_size_from_observations(self):
        t5 = self.oof[self.oof.task.eq("T5")]
        matrix = confusion_matrix(t5.y_true, t5.y_pred, labels=range(5))
        table = pd.read_csv(G.TABLES_DIR / "adjacent_class_overlap.csv")
        for i, row in enumerate(table.itertuples()):
            self.assertEqual(row.higher_to_lower_count, matrix[i + 1, i])
            self.assertAlmostEqual(row.confusion_rate,
                                   (matrix[i, i + 1] + matrix[i + 1, i]) / matrix[[i, i + 1], :].sum())
        effects = pd.read_csv(G.TABLES_DIR / "overlap_numerical_adjacent_effects.csv")
        for row in effects[effects.class_pair.eq("C3-C4")].itertuples():
            a = self.X.loc[self.y.eq(3), row.feature].dropna().to_numpy()
            b = self.X.loc[self.y.eq(4), row.feature].dropna().to_numpy()
            direct = ((a[:, None] > b).sum() - (a[:, None] < b).sum()) / (len(a) * len(b))
            self.assertAlmostEqual(row.cliffs_delta_lower_minus_higher, direct)
            self.assertTrue(0 <= row.histogram_overlap <= 1 + 1e-12)

    def test_categorical_denominators_and_PCA_membership(self):
        counts = pd.read_csv(G.TABLES_DIR / "overlap_categorical_crosstabs.csv")
        for (_, code), rows in counts.groupby(["feature", "class"]):
            self.assertEqual(rows.n.sum(), self.y.eq(int(code[-1])).sum())
            self.assertAlmostEqual(rows.percentage_within_class.sum(), 100)
        coords = pd.read_csv(G.TABLES_DIR / "overlap_pca_coordinates.csv")
        self.assertTrue(coords.SEQN.is_unique)
        self.assertEqual(set(coords.SEQN), set(self.X.index))
        self.assertTrue(np.isfinite(coords[["PC1", "PC2"]]).all().all())

    def test_hashes_old_files_and_new_outputs(self):
        protocol = json.loads(G.PROTOCOL_PATH.read_text())
        G.verify_hashes(protocol["preserved_files_sha256"])
        G.verify_hashes(protocol["design"]["source_sha256"])
        for path in [G.RUN_PATH, O.RUN_PATH]:
            run = json.loads(path.read_text())
            self.assertFalse(run["heldout_used"])
            self.assertTrue(run["all_old_files_unchanged"])
            self.assertEqual(run["protocol_sha256"], G.sha256_file(G.PROTOCOL_PATH))
            G.verify_hashes(run["outputs_sha256"])


if __name__ == "__main__":
    unittest.main()
