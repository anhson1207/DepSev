"""Recalculate nested-CV evidence using development IDs and saved outputs only."""

from importlib import import_module
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, ParameterSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import ARTIFACTS_DIR, TABLES_DIR, experiment_path
from protocol import load_final_split, load_original_split, sha256_file
from tuning_estimators import id_hash

TUNING = import_module("05_nested_tuning")


class NestedTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev, _ = TUNING.ablation.load_inputs()
        cls.y = pd.Series(cls.dev.severity.to_numpy(), index=cls.dev.SEQN.astype(int))
        cls.split = load_final_split()
        cls.manifest = json.loads(TUNING.MANIFEST.read_text())
        cls.assignments = json.loads(TUNING.FOLDS_PATH.read_text())
        cls.scores = pd.read_csv(TABLES_DIR / "nested_cv_tuning_results.csv")
        cls.trials = pd.read_csv(TABLES_DIR / "nested_cv_inner_search_results.csv")
        cls.oof = pd.read_csv(TABLES_DIR / "nested_cv_oof_predictions.csv")
        cls.summary = pd.read_csv(TABLES_DIR / "nested_cv_summary.csv")
        cls.config = json.loads((ARTIFACTS_DIR / "tuned_model_configs.json").read_text())

    def test_protected_inputs_code_and_baselines_unchanged(self):
        for path, expected in self.manifest["protected_sha256"].items():
            self.assertEqual(sha256_file(Path(path)), expected)
        self.assertEqual(self.manifest["development_n"], len(self.dev))
        self.assertFalse(self.manifest["heldout_used"])

    def test_outer_and_inner_indices_match_requested_stratification(self):
        outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.dev, self.y))
        fixed = json.loads(experiment_path(ARTIFACTS_DIR, "cv_folds.json").read_text())
        for number, (train, valid) in enumerate(outer, 1):
            row = self.assignments["outer_partitions"][number - 1]
            self.assertEqual(row["outer_train_seqn"], self.y.iloc[train].index.tolist())
            self.assertEqual(row["outer_validation_seqn"], self.y.iloc[valid].index.tolist())
            np.testing.assert_array_equal(np.asarray(fixed["validation_fold"])[valid], number)
            inner = StratifiedKFold(3, shuffle=True, random_state=42).split(self.dev.iloc[train], self.y.iloc[train])
            for fold, (inner_train, inner_valid) in enumerate(inner, 1):
                np.testing.assert_array_equal(np.flatnonzero(np.array(row["inner_validation_fold"]) == fold), inner_valid)
                self.assertFalse(set(train[inner_train]) & set(valid))
                self.assertFalse(set(train[inner_valid]) & set(valid))
                self.assertFalse(set(inner_train) & set(inner_valid))
                self.assertEqual(set(self.y.iloc[train[inner_train]]), set(range(5)))

    def test_equal_search_budget_and_inner_only_parameter_selection(self):
        self.assertEqual(len(self.scores), 4 * 5)
        self.assertEqual(len(self.trials), 4 * 5 * 12)
        self.assertEqual(set(zip(self.scores.model, self.scores.feature_set)), set(TUNING.CANDIDATES))
        for row in self.scores.to_dict("records"):
            candidates = self.trials[self.trials.model.eq(row["model"])
                                     & self.trials.feature_set.eq(row["feature_set"])
                                     & self.trials.outer_fold.eq(row["outer_fold"])].sort_values("candidate_index")
            self.assertEqual(len(candidates), 12)
            actual = [json.loads(value) for value in candidates.params]
            expected = list(ParameterSampler(TUNING.SEARCH_SPACES[row["model"]], n_iter=12, random_state=42))
            self.assertEqual(actual, expected)
            means = candidates[[f"inner_fold_{i}_macro_f1" for i in [1, 2, 3]]].mean(axis=1)
            np.testing.assert_allclose(means, candidates.mean_inner_macro_f1)
            best = candidates.iloc[np.argmax(candidates.mean_inner_macro_f1.to_numpy())]
            self.assertEqual(json.loads(row["best_params"]), json.loads(best.params))
            self.assertAlmostEqual(row["best_inner_macro_f1"], best.mean_inner_macro_f1)

    def test_outer_oof_and_per_class_scores_recalculate(self):
        classes = pd.read_csv(TABLES_DIR / "nested_cv_per_class_results.csv")
        self.assertEqual(len(classes), 20)
        self.assertFalse(set(self.oof.SEQN) & set(load_original_split()["test_seqn"]))
        self.assertEqual(len(self.oof), 4 * len(self.y))
        for (model, feature_set), rows in self.oof.groupby(["model", "feature_set"]):
            self.assertTrue(rows.SEQN.is_unique)
            self.assertEqual(set(rows.SEQN), set(self.y.index))
            np.testing.assert_array_equal(rows.y_true, rows.SEQN.map(self.y))
            for number, predictions in rows.groupby("outer_fold"):
                self.assertEqual(set(predictions.SEQN), set(self.assignments["outer_partitions"][number - 1]["outer_validation_seqn"]))
                actual = TUNING.ablation.summarize_metrics(predictions.y_true, predictions.y_pred)
                saved = self.scores[self.scores.model.eq(model) & self.scores.feature_set.eq(feature_set)
                                    & self.scores.outer_fold.eq(number)].iloc[0]
                for key, value in actual.items():
                    self.assertAlmostEqual(value, saved[f"outer_{key}"])
                _, recall, f1, support = precision_recall_fscore_support(
                    predictions.y_true, predictions.y_pred, labels=np.arange(5), zero_division=0)
                for code in range(5):
                    self.assertAlmostEqual(saved[f"outer_F1_C{code}"], f1[code])
                    self.assertAlmostEqual(saved[f"outer_Recall_C{code}"], recall[code])
                    self.assertEqual(saved[f"outer_support_C{code}"], support[code])
            _, recall, f1, support = precision_recall_fscore_support(rows.y_true, rows.y_pred, labels=np.arange(5), zero_division=0)
            saved = classes[classes.model.eq(model) & classes.feature_set.eq(feature_set)].sort_values("class")
            np.testing.assert_allclose(saved.recall, recall)
            np.testing.assert_allclose(saved.f1, f1)
            np.testing.assert_array_equal(saved.support, support)
        for row in self.summary.to_dict("records"):
            scores = self.scores[self.scores.model.eq(row["model"]) & self.scores.feature_set.eq(row["feature_set"])]
            for metric in TUNING.ablation.METRICS:
                self.assertAlmostEqual(row[f"{metric}_mean"], scores[f"outer_{metric}"].mean())
                self.assertAlmostEqual(row[f"{metric}_std"], scores[f"outer_{metric}"].std(ddof=1))

    def test_every_xgboost_fit_uses_corresponding_training_partition(self):
        allowed = {}
        for feature_set in ["F3", "F4"]:
            for row in self.assignments["outer_partitions"]:
                ids = np.array(row["outer_train_seqn"])
                assignments = np.array(row["inner_validation_fold"])
                context = f"XGB_{feature_set}_outer{row['outer_fold']}"
                allowed[context + "_inner"] = {id_hash(ids[assignments != fold]): ids[assignments != fold] for fold in [1, 2, 3]}
                allowed[context + "_refit"] = {id_hash(ids): ids}
        expected_logs = 2 * 5 * (12 * 3 + 1)
        if self.config["selected_model"] == "XGBoost":
            final = json.loads((ARTIFACTS_DIR / "final_tuning_cv_assignments.json").read_text())
            ids = np.array(final["development_seqn"])
            assignments = np.array(final["validation_fold"])
            context = f"XGB_{self.config['selected_feature_set']}_full_development_inner"
            allowed[context] = {id_hash(ids[assignments != fold]): ids[assignments != fold] for fold in [1, 2, 3]}
            expected_logs += 12 * 3
        logs = list(TUNING.WEIGHT_DIR.glob("*.json"))
        self.assertEqual(len(logs), expected_logs)
        observed = {}
        for path in logs:
            row = json.loads(path.read_text())
            context, fingerprint = row["fit_context"], row["training_ids_sha256"]
            ids = allowed[context][fingerprint]
            self.assertFalse(set(ids) & set(self.split["test_seqn"]))
            y = self.y.loc[ids]
            counts = y.value_counts().sort_index().to_numpy()
            self.assertEqual(row["n_train"], len(y))
            np.testing.assert_array_equal(row["class_counts"], counts)
            np.testing.assert_allclose(row["class_weights"], len(y) / (5 * counts))
            np.testing.assert_allclose(row["class_weight_sums"], len(y) / 5)
            self.assertTrue(row["computed_inside_fit"])
            observed[(context, fingerprint)] = observed.get((context, fingerprint), 0) + 1
        for (context, _), count in observed.items():
            self.assertEqual(count, 1 if context.endswith("_refit") else 12)

    def test_final_search_only_selected_candidate_without_full_data_refit(self):
        selection_path = ARTIFACTS_DIR / "nested_cv_selection.json"
        selection = json.loads(selection_path.read_text())
        ranked = self.summary.sort_values(["macro_f1_mean", "macro_f1_std", "feature_set", "model"], ascending=[False, True, True, True])
        best = ranked.iloc[0]
        self.assertEqual((selection["model"], selection["feature_set"]), (best.model, best.feature_set))
        self.assertEqual(self.config["selection_sha256"], sha256_file(selection_path))
        self.assertEqual(selection["nested_results_sha256"], sha256_file(TABLES_DIR / "nested_cv_tuning_results.csv"))
        self.assertFalse(self.config["final_search_refit"])
        self.assertFalse(self.config["full_development_model_fitted"])
        self.assertFalse(self.config["heldout_used_for_selection"])
        trials = pd.read_csv(TABLES_DIR / "final_development_search_results.csv")
        self.assertEqual(len(trials), 12)
        self.assertEqual(set(trials.model), {best.model})
        self.assertEqual(set(trials.feature_set), {best.feature_set})
        self.assertEqual([json.loads(value) for value in trials.params],
                         list(ParameterSampler(TUNING.SEARCH_SPACES[best.model], n_iter=12, random_state=42)))
        winner = trials.iloc[np.argmax(trials.mean_inner_macro_f1.to_numpy())]
        self.assertEqual(self.config["best_params"], json.loads(winner.params))
        self.assertAlmostEqual(self.config["full_development_best_inner_macro_f1"], winner.mean_inner_macro_f1)
        assignments = json.loads((ARTIFACTS_DIR / "final_tuning_cv_assignments.json").read_text())
        self.assertEqual(assignments["development_seqn"], self.y.index.tolist())
        self.assertFalse(set(assignments["development_seqn"]) & set(self.split["test_seqn"]))

    def test_baseline_comparisons_use_same_outer_folds(self):
        paired = pd.read_csv(TABLES_DIR / "nested_cv_baseline_comparison.csv")
        self.assertEqual(len(paired), 20)
        self.assertTrue(paired.outer_fold.eq(paired.fold).all())
        np.testing.assert_allclose(paired.delta_tuned_minus_baseline, paired.outer_macro_f1 - paired.macro_f1)
        for row in self.summary.to_dict("records"):
            values = paired[paired.model.eq(row["model"]) & paired.feature_set.eq(row["feature_set"])].delta_tuned_minus_baseline
            self.assertAlmostEqual(row["delta_vs_baseline"], values.mean())


if __name__ == "__main__":
    unittest.main()
