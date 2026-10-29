"""Recalculate saved supplementary evaluation without predicting held-out again."""

from datetime import datetime
from importlib import import_module
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_protocol as final
from config import ARTIFACTS_DIR, TABLES_DIR
from protocol import load_final_split, sha256_file
from tuning_estimators import id_hash

FINAL = import_module("07_final_test")
EXPLAIN = import_module("08_explain")


class FinalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = final.verify_specification()
        cls.pipeline, cls.lock = final.load_frozen_model()
        cls.evaluation = final.verify_evaluation()
        cls.split = load_final_split()
        cls.predictions = pd.read_csv(final.PREDICTIONS)
        cls.X, cls.y, cls.inventory = final.load_features("development")

    def test_frozen_schema_config_target_and_membership_hashes(self):
        schema = json.loads(final.SCHEMA.read_text())
        mapping = json.loads(final.MAPPING.read_text())
        self.assertEqual(self.spec["features"], self.X.columns.tolist())
        self.assertEqual(schema["ordered_features"], self.X.columns.tolist())
        self.assertEqual(self.X.shape, (4045, 34))
        self.assertEqual(self.spec["hyperparameters"], self.pipeline.named_steps["model"].get_params())
        self.assertEqual(self.spec["hashes"]["RF_config_sha256"], final.json_hash(self.spec["hyperparameters"]))
        self.assertEqual(self.spec["hashes"]["feature_schema_sha256"], final.json_hash(schema))
        for name, ids in [("development", self.split["development_seqn"]), ("heldout", self.split["test_seqn"])]:
            self.assertEqual(self.spec["hashes"][f"{name}_ids_sha256"], final.membership_hash(ids))
        self.assertFalse(set(self.X.index) & set(self.split["test_seqn"]))
        self.assertEqual(self.spec["target_mapping"], mapping)
        self.assertEqual(len(mapping["source_items"]), 9)
        scores = [score for row in mapping["classes"] for score in range(row["phq9_min"], row["phq9_max"] + 1)]
        self.assertEqual(scores, list(range(28)))
        self.assertFalse(self.spec["historically_untouched"])
        self.assertFalse(self.spec["heldout_used_for_selection"])
        self.assertTrue(self.spec["selected_before_heldout"])

    def test_fit_preprocessing_only_full_development(self):
        numerical = self.pipeline.named_steps["preprocessor"].named_transformers_["numerical"]
        columns = self.spec["preprocessing"]["numerical"]["columns"]
        np.testing.assert_allclose(numerical["imputer"].statistics_, self.X[columns].median())
        expected = self.X[columns].fillna(self.X[columns].median()).to_numpy()
        np.testing.assert_allclose(numerical["scaler"].mean_, expected.mean(axis=0))
        np.testing.assert_allclose(numerical["scaler"].var_, expected.var(axis=0))
        self.assertEqual(numerical["scaler"].n_samples_seen_, 4045)
        self.assertEqual(self.lock["training_ids_sha256"], id_hash(self.X.index))
        self.assertEqual(self.lock["class_counts"], self.y.value_counts().sort_index().tolist())
        self.assertFalse(self.lock["heldout_used_for_fit"])
        self.assertEqual(self.lock["metadata_sha256"], sha256_file(ARTIFACTS_DIR / "final_preprocessor_metadata.json"))
        self.assertEqual(self.lock["schema_sha256"], sha256_file(final.SCHEMA))
        self.assertEqual(self.lock["mapping_sha256"], sha256_file(final.MAPPING))

    def test_freeze_fit_evaluation_explanation_order(self):
        fit_start = json.loads(FINAL.FIT_CLAIM.read_text())
        evaluation_start = json.loads(FINAL.EVALUATION_CLAIM.read_text())
        times = [self.spec["created_utc"], fit_start["started_utc"], self.lock["fit_completed_utc"],
                 self.lock["locked_utc"], evaluation_start["started_utc"], self.evaluation["completed_utc"]]
        parsed = [datetime.fromisoformat(value) for value in times]
        self.assertEqual(parsed, sorted(parsed))
        explanation = json.loads(EXPLAIN.PROTOCOL.read_text())
        self.assertEqual(explanation["saved_evaluation_sha256"], sha256_file(final.EVALUATION_LOCK))
        self.assertEqual(explanation["model_sha256"], self.lock["model_sha256"])
        self.assertEqual(self.evaluation["prediction_calls"], 1)
        for path in [EXPLAIN.PERM_LOCK, EXPLAIN.SHAP_LOCK]:
            record = EXPLAIN.verify_saved(path)
            self.assertGreater(datetime.fromisoformat(record["completed_utc"]), parsed[-1])
        self.assertEqual(self.evaluation["evaluation"], "supplementary held-out evaluation")

    def test_saved_predictions_membership_argmax_and_metrics(self):
        p = self.predictions
        self.assertEqual(len(p), 1009)
        self.assertTrue(p.SEQN.is_unique)
        self.assertEqual(set(p.SEQN), set(self.split["test_seqn"]))
        self.assertFalse(set(p.SEQN) & set(self.X.index))
        probabilities = p[[f"prob_C{i}" for i in range(5)]].to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_array_equal(p.y_pred, probabilities.argmax(axis=1))
        saved = pd.read_csv(TABLES_DIR / "final_heldout_metrics.csv").iloc[0]
        metrics = {
            "macro_f1": f1_score(p.y_true, p.y_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(p.y_true, p.y_pred),
            "macro_precision": precision_score(p.y_true, p.y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(p.y_true, p.y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(p.y_true, p.y_pred, average="weighted", zero_division=0),
            "accuracy": accuracy_score(p.y_true, p.y_pred)}
        for name, value in metrics.items():
            self.assertAlmostEqual(saved[name], value)
        per_class = pd.read_csv(TABLES_DIR / "final_heldout_per_class_results.csv").sort_values("class")
        precision, recall, f1, support = precision_recall_fscore_support(p.y_true, p.y_pred, labels=range(5), zero_division=0)
        for name, values in [("precision", precision), ("recall", recall), ("f1", f1), ("support", support)]:
            np.testing.assert_allclose(per_class[name], values)
        matrix = confusion_matrix(p.y_true, p.y_pred, labels=range(5))
        np.testing.assert_array_equal(pd.read_csv(TABLES_DIR / "final_confusion_matrix.csv").iloc[:, 1:], matrix)
        np.testing.assert_array_equal(per_class.true_positive, np.diag(matrix))
        np.testing.assert_array_equal(per_class.predicted_count, matrix.sum(axis=0))

    def test_completed_stages_do_not_refit_or_predict_again(self):
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Unexpected refit")), \
                patch.object(Pipeline, "predict_proba", side_effect=AssertionError("Repeated held-out prediction")), \
                patch.object(FINAL, "load_features", side_effect=AssertionError("Unexpected data load")):
            FINAL.refit()
            FINAL.supplementary_evaluation()

    def test_permutation_repeat_statistics_and_development_scope(self):
        record = EXPLAIN.verify_saved(EXPLAIN.PERM_LOCK)
        self.assertEqual(record["n"], 4045)
        self.assertFalse(record["heldout_used"])
        table = pd.read_csv(TABLES_DIR / "permutation_importance.csv")
        self.assertEqual(len(table), 34 * 12)
        self.assertEqual(set(table["class"]), {"global", "C0", "C1", "C2", "C3", "C4"})
        with np.load(ARTIFACTS_DIR / "permutation_importance_raw.npz", allow_pickle=False) as raw:
            self.assertEqual(raw["features"].tolist(), self.X.columns.tolist())
            for metric, rows in table.groupby("metric"):
                rows = rows.set_index("feature").loc[self.X.columns]
                values = raw[metric]
                self.assertEqual(values.shape, (34, 10))
                np.testing.assert_allclose(rows.importance_mean, values.mean(axis=1), atol=1e-14)
                np.testing.assert_allclose(rows.importance_std, values.std(axis=1, ddof=1), atol=1e-14)
        self.assertEqual(set(table.partition), {"development_in_sample"})

    def test_shap_shapes_grouping_additivity_and_class_importance(self):
        record = EXPLAIN.verify_saved(EXPLAIN.SHAP_LOCK)
        self.assertFalse(record["heldout_used"])
        self.assertEqual(record["explained_n"], 4045)
        mapping = json.loads((ARTIFACTS_DIR / "shap_encoded_feature_mapping.json").read_text())
        sources = np.array(mapping["original_features"])
        importance = pd.read_csv(TABLES_DIR / "shap_global_importance.csv").set_index("feature").loc[self.X.columns]
        with np.load(ARTIFACTS_DIR / "shap_values_final_rf.npz", allow_pickle=False) as raw:
            values, encoded = raw["grouped_values"], raw["encoded_values"]
            self.assertEqual(values.shape, (4045, 34, 5))
            self.assertEqual(encoded.shape, (4045, len(sources), 5))
            np.testing.assert_array_equal(raw["SEQN"], self.X.index)
            np.testing.assert_array_equal(raw["classes"], self.pipeline.classes_)
            np.testing.assert_allclose(values.sum(axis=1) + raw["base_values"], raw["development_probabilities"], atol=1e-7)
            for i, feature in enumerate(self.X.columns):
                np.testing.assert_allclose(values[:, i, :], encoded[:, sources == feature, :].sum(axis=1))
            np.testing.assert_allclose(importance.mean_abs_shap_global, np.abs(values).mean(axis=(0, 2)))
            for c in range(5):
                np.testing.assert_allclose(importance[f"mean_abs_shap_C{c}"], np.abs(values[:, :, c]).mean(axis=0))
                per_class = pd.read_csv(TABLES_DIR / f"shap_importance_C{c}.csv").set_index("feature").loc[self.X.columns]
                np.testing.assert_allclose(per_class.mean_abs_shap, importance[f"mean_abs_shap_C{c}"])
            # Validate both supported API layouts using actual SHAP values, not synthetic data.
            legacy = [encoded[:11, :, c] for c in range(5)]
            normalized, _ = EXPLAIN.normalize_shap_values(legacy, 11, encoded.shape[1], self.pipeline.classes_)
            np.testing.assert_array_equal(normalized, encoded[:11])
            reordered, _ = EXPLAIN.normalize_shap_values(encoded[:11].transpose(2, 0, 1), 11,
                                                         encoded.shape[1], self.pipeline.classes_)
            np.testing.assert_array_equal(reordered, encoded[:11])


if __name__ == "__main__":
    unittest.main()
