"""Audit ensemble results, actual fit partitions and probability construction."""

from importlib import import_module
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ablation_features import prepare_features
from config import ARTIFACTS_DIR, TABLES_DIR, experiment_path
from protocol import load_final_split, load_original_split, sha256_file
from tuning_estimators import id_hash

ENSEMBLE = import_module("06_ensemble")


class EnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(ENSEMBLE.CONFIG_PATH.read_text())
        cls.manifest = json.loads(ENSEMBLE.MANIFEST_PATH.read_text())
        cls.partitions = json.loads(ENSEMBLE.FOLDS_PATH.read_text())["outer_partitions"]
        cls.audit = json.loads((ARTIFACTS_DIR / "ensemble_fit_audit.json").read_text())
        cls.run_metadata = json.loads((ARTIFACTS_DIR / "ensemble_run.json").read_text())
        cls.oof = pd.read_csv(TABLES_DIR / "ensemble_oof_predictions.csv")
        cls.inner = pd.read_csv(TABLES_DIR / "ensemble_inner_oof_predictions.csv")
        cls.scores = pd.read_csv(TABLES_DIR / "ensemble_fold_results.csv")
        cls.summary = pd.read_csv(ENSEMBLE.RESULT_PATH)
        cls.classes = pd.read_csv(TABLES_DIR / "ensemble_per_class_results.csv")
        cls.dev, raw_inventory = ENSEMBLE.ablation.load_inputs()
        cls.X, _, cls.inventory, _ = prepare_features(cls.dev, raw_inventory)
        cls.X.index = cls.dev.SEQN.astype(int)
        cls.y = pd.Series(cls.dev.severity.to_numpy(dtype=int), index=cls.X.index)

    def test_frozen_parameters_inputs_and_outputs(self):
        ENSEMBLE.verify_protected(self.manifest)
        for filename, expected in self.run_metadata["outputs_sha256"].items():
            self.assertEqual(sha256_file(Path(filename)), expected)
        self.assertEqual(self.run_metadata["manifest_sha256"], sha256_file(ENSEMBLE.MANIFEST_PATH))
        self.assertFalse(self.run_metadata["heldout_used"])
        self.assertFalse(self.run_metadata["full_development_model_fitted"])
        self.assertEqual(self.config["LR"]["C"], 0.3)
        self.assertEqual(self.config["LR"]["solver"], "newton-cg")
        self.assertEqual(self.config["LR"]["l1_ratio"], 0)
        self.assertFalse(self.config["LR_provenance"]["globally_optimal_claim"])
        rf = json.loads((ARTIFACTS_DIR / "tuned_model_configs.json").read_text())
        self.assertEqual(self.config["RF"], rf["estimator_params"])
        self.assertEqual(ENSEMBLE.verify_xgb_fixed(self.config),
                         {"objective": "multi:softprob", "num_class": 5, "eval_metric": "mlogloss"})
        self.assertEqual(self.config["voting_weights_LR_RF_XGB"],
                         {"SoftVoting": [1, 1, 1], "SoftVoting_RF2": [1, 2, 1]})

    def test_membership_outer_and_inner_oof_coverage(self):
        split = load_final_split()
        self.assertEqual(set(self.y.index), set(split["development_seqn"]))
        self.assertFalse(set(self.oof.SEQN) & set(load_original_split()["test_seqn"]))
        self.assertFalse(set(self.inner.SEQN) & set(split["test_seqn"]))
        self.assertEqual(len(self.oof), 6 * len(self.dev))
        self.assertEqual(len(self.inner), 4 * len(self.dev))
        expected_outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.X, self.y))
        for row, (train, valid) in zip(self.partitions, expected_outer):
            self.assertEqual(row["outer_train_seqn"], self.y.iloc[train].index.tolist())
            self.assertEqual(row["outer_validation_seqn"], self.y.iloc[valid].index.tolist())
            inner = self.inner[self.inner.outer_fold.eq(row["outer_fold"])]
            self.assertEqual(inner.SEQN.tolist(), row["outer_train_seqn"])
            self.assertTrue(inner.SEQN.is_unique)
            np.testing.assert_array_equal(inner.inner_fold, row["inner_validation_fold"])
            np.testing.assert_array_equal(inner.y_true, inner.SEQN.map(self.y))
            expected_inner = StratifiedKFold(3, shuffle=True, random_state=42).split(
                self.X.iloc[train], self.y.iloc[train])
            for number, (inner_train, inner_valid) in enumerate(expected_inner, 1):
                self.assertFalse(set(train[inner_train]) & set(valid))
                self.assertFalse(set(train[inner_valid]) & set(valid))
                np.testing.assert_array_equal(np.flatnonzero(inner.inner_fold.to_numpy() == number), inner_valid)
            for name in ENSEMBLE.BASES:
                probabilities = inner[[f"{name}_prob_C{c}" for c in range(5)]].to_numpy()
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            for name in ENSEMBLE.MODELS:
                outer = self.oof[self.oof.model.eq(name) & self.oof.fold.eq(row["outer_fold"])]
                self.assertEqual(outer.SEQN.tolist(), row["outer_validation_seqn"])

    def test_every_preprocessor_fit_uses_correct_training_partition(self):
        self.assertEqual(len(self.audit["base_fits"]), 60)
        expected = {}
        for row in self.partitions:
            ids = np.array(row["outer_train_seqn"])
            assignment = np.array(row["inner_validation_fold"])
            for name in ENSEMBLE.BASES:
                expected[f"outer{row['outer_fold']}_refit_{name}"] = (ids, row["outer_validation_seqn"])
                for fold in range(1, 4):
                    expected[f"outer{row['outer_fold']}_inner{fold}_{name}"] = (
                        ids[assignment != fold], ids[assignment == fold])
        self.assertEqual({r["context"] for r in self.audit["base_fits"]}, set(expected))
        numerical = self.inventory.loc[self.inventory.feature_type.eq("numerical"), "feature"].tolist()
        for row in self.audit["base_fits"]:
            train, valid = expected[row["context"]]
            self.assertEqual(row["training_ids_sha256"], id_hash(train))
            self.assertEqual(row["prediction_ids_sha256"], id_hash(valid))
            self.assertFalse(set(train) & set(valid))
            self.assertLess(row["n_train"], len(self.y))
            self.assertEqual(row["n_train"], len(train))
            self.assertEqual(row["scaler_n_samples_seen"], len(train))
            np.testing.assert_allclose(row["numerical_imputer_medians"], self.X.loc[train, numerical].median())
            np.testing.assert_array_equal(row["class_counts"], self.y.loc[train].value_counts().sort_index())
            if row["model"] == "XGB":
                self.assertEqual(row["XGB_fixed_verified"], self.config["XGB_fixed"])

    def test_xgb_weights_from_each_actual_training_partition(self):
        logs = list(ENSEMBLE.WEIGHT_DIR.glob("*.json"))
        self.assertEqual(len(logs), 20)
        actual = {r["context"]: r for r in self.audit["base_fits"] if r["model"] == "XGB"}
        for path in logs:
            row = json.loads(path.read_text())
            expected = actual[row["fit_context"]]
            self.assertEqual(row["training_ids_sha256"], expected["training_ids_sha256"])
            self.assertEqual(row["params"], self.config["XGB"])
            self.assertEqual(row["class_counts"], expected["class_counts"])
            self.assertTrue(row["computed_inside_fit"])
            counts = np.array(expected["class_counts"])
            np.testing.assert_allclose(row["class_weights"], expected["n_train"] / (5 * counts))
            np.testing.assert_allclose(row["class_weight_sums"], expected["n_train"] / 5)

    def test_probabilities_voting_and_stacking_reconstruction(self):
        self.assertEqual(len(self.audit["meta_fits"]), 5)
        for row in self.partitions:
            fold = row["outer_fold"]
            tables = {name: self.oof[self.oof.model.eq(name) & self.oof.fold.eq(fold)]
                      for name in ENSEMBLE.MODELS}
            probabilities = {name: table[ENSEMBLE.PROB_COLUMNS].to_numpy() for name, table in tables.items()}
            for name, table in tables.items():
                np.testing.assert_allclose(probabilities[name].sum(axis=1), 1, atol=1e-6)
                np.testing.assert_array_equal(table.y_pred, probabilities[name].argmax(axis=1))
                np.testing.assert_array_equal(table.y_true, table.SEQN.map(self.y))
            for name, weights in self.config["voting_weights_LR_RF_XGB"].items():
                expected = np.average(np.stack([probabilities[b] for b in ENSEMBLE.BASES]),
                                      axis=0, weights=weights)
                np.testing.assert_allclose(probabilities[name], expected, atol=1e-12)
            meta = self.audit["meta_fits"][fold - 1]
            self.assertEqual(meta["params"], self.config["meta_LR"])
            self.assertEqual(meta["training_ids_sha256"], id_hash(row["outer_train_seqn"]))
            self.assertEqual(meta["training_features"], "inner OOF only")
            self.assertEqual(meta["n_train"], len(row["outer_train_seqn"]))
            meta_valid = np.column_stack([probabilities[b] for b in ENSEMBLE.BASES])
            logits = meta_valid @ np.asarray(meta["coef"]).T + meta["intercept"]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            np.testing.assert_allclose(probabilities["Stacking"], exp / exp.sum(axis=1, keepdims=True), atol=1e-12)

    def test_metrics_recalculate_from_outer_oof(self):
        self.assertEqual(len(self.scores), 30)
        self.assertEqual(len(self.classes), 30)
        for name, rows in self.oof.groupby("model"):
            scores = self.scores[self.scores.model.eq(name)]
            self.assertTrue(rows.SEQN.is_unique)
            for fold, predictions in rows.groupby("fold"):
                saved = scores[scores.fold.eq(fold)].iloc[0]
                for key, value in ENSEMBLE.ablation.summarize_metrics(predictions.y_true, predictions.y_pred).items():
                    self.assertAlmostEqual(value, saved[key])
                _, recall, f1, support = precision_recall_fscore_support(
                    predictions.y_true, predictions.y_pred, labels=range(5), zero_division=0)
                for c in range(5):
                    self.assertAlmostEqual(saved[f"Recall_C{c}"], recall[c])
                    self.assertAlmostEqual(saved[f"F1_C{c}"], f1[c])
                    self.assertEqual(saved[f"support_C{c}"], support[c])
            saved = self.summary[self.summary.model.eq(name)].iloc[0]
            for metric in ENSEMBLE.ablation.METRICS:
                self.assertAlmostEqual(saved[f"{metric}_mean"], scores[metric].mean())
                self.assertAlmostEqual(saved[f"{metric}_std"], scores[metric].std(ddof=1))
            _, recall, f1, support = precision_recall_fscore_support(
                rows.y_true, rows.y_pred, labels=range(5), zero_division=0)
            per_class = self.classes[self.classes.model.eq(name)].sort_values("class")
            np.testing.assert_allclose(per_class.recall, recall)
            np.testing.assert_allclose(per_class.f1, f1)
            np.testing.assert_array_equal(per_class.support, support)

    def test_baseline_xgb_predictions_reproduced(self):
        baseline = pd.read_csv(experiment_path(TABLES_DIR, "ablation_oof_predictions.csv"))
        baseline = baseline[baseline.feature_set.eq("F4") & baseline.model.eq("XGBoost")].set_index("SEQN")
        actual = self.oof[self.oof.model.eq("XGB")].set_index("SEQN").loc[baseline.index]
        np.testing.assert_array_equal(actual.y_pred, baseline.y_pred)
        np.testing.assert_array_equal(actual.fold, baseline.fold)

    def test_rf_reproduces_nested_predictions_where_parameters_match(self):
        parameters = pd.read_csv(TABLES_DIR / "best_params_by_fold.csv")
        parameters = parameters[parameters.model.eq("Random Forest")]
        nested = pd.read_csv(TABLES_DIR / "nested_cv_oof_predictions.csv")
        nested = nested[nested.model.eq("Random Forest") & nested.feature_set.eq("F4")]
        for row in parameters.itertuples(index=False):
            best = {k.removeprefix("model__"): v for k, v in json.loads(row.best_params).items()}
            if all(self.config["RF"][k] == v for k, v in best.items()):
                before = nested[nested.outer_fold.eq(row.outer_fold)].set_index("SEQN")
                after = self.oof[self.oof.model.eq("RF") & self.oof.fold.eq(row.outer_fold)].set_index("SEQN")
                np.testing.assert_array_equal(after.loc[before.index].y_pred, before.y_pred)

    def test_candidate_follows_prespecified_rule_and_stops_before_refit(self):
        candidate = json.loads((ARTIFACTS_DIR / "ensemble_candidate.json").read_text())
        paired = pd.read_csv(TABLES_DIR / "ensemble_fold_comparison.csv")
        scores = self.summary.set_index("model")
        eligible = []
        for row in candidate["comparisons"]:
            name = row["model"]
            delta = paired[name] - paired.RF
            self.assertAlmostEqual(row["mean_delta_vs_RF"], delta.mean())
            self.assertEqual(row["folds_improved"], (delta > 0).sum())
            qualifies = (delta.mean() >= 0.01 and (delta > 0).sum() >= 4 and
                         scores.loc[name, "macro_f1_std"] <= scores.loc["RF", "macro_f1_std"])
            self.assertEqual(row["qualifies"], qualifies)
            if qualifies:
                eligible.append(name)
        winner = max(eligible, key=lambda n: scores.loc[n, "macro_f1_mean"]) if eligible else "RF"
        self.assertEqual(candidate["candidate"], winner)
        self.assertFalse(candidate["full_development_model_fitted"])
        self.assertFalse(candidate["heldout_used"])
        self.assertTrue(candidate["requires_user_confirmation_before_final_refit"])


if __name__ == "__main__":
    unittest.main()
