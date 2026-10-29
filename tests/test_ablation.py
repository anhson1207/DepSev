"""Research protocol checks using only the real development data and saved OOF.

Run after ablation: .venv/bin/python -m unittest discover -s tests -v
No synthetic participant data, test-set evaluation, or additional training.
"""

import hashlib
import json
from pathlib import Path
import runpy
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ablation_features import prepare_features
from config import experiment_path

ABLATION = runpy.run_path(str(ROOT / "src" / "05_feature_ablation.py"))


class AblationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev, cls.inventory = ABLATION["load_inputs"]()
        cls.X, cls.groups, cls.final_inventory, cls.audit = prepare_features(cls.dev, cls.inventory)
        cls.lock = json.loads((experiment_path(ROOT / "artifacts", "cv_folds.json")).read_text())
        cls.metadata = json.loads((experiment_path(ROOT / "artifacts", "ablation_config.json")).read_text())
        cls.oof = pd.read_csv(experiment_path(ROOT / "results/tables", "ablation_oof_predictions.csv"))
        cls.fold_scores = pd.read_csv(experiment_path(ROOT / "results/tables", "ablation_fold_results.csv"))

    def test_protected_inputs_are_unchanged(self):
        for path, expected in self.metadata["protected_inputs_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)

    def test_cv_matches_locked_row_order_and_has_no_test_ids(self):
        split = json.loads((ROOT / "artifacts/split_indices.json").read_text())
        ids = self.dev.SEQN.astype(int)
        self.assertEqual(ids.tolist(), self.lock["development_seqn"])
        self.assertFalse(set(ids) & set(split["test_seqn"]))
        self.assertEqual(set(self.oof.SEQN), set(ids))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for number, (train, validation) in enumerate(cv.split(self.X, self.dev.severity), 1):
            self.assertFalse(set(train) & set(validation))
            np.testing.assert_array_equal(np.array(self.lock["validation_fold"])[validation], number)
            self.assertEqual(set(self.dev.severity.iloc[validation]), set(range(5)))

    def test_structural_missingness_and_variable_specific_codes(self):
        raw, clean = self.dev, self.X
        self.assertTrue(clean.loc[raw.SMQ020.eq(2), "smoking_status"].eq("under_100_lifetime").all())
        self.assertTrue(clean.loc[raw.SMQ020.eq(9), "smoking_status"].isna().all())
        for feature, invalid_codes in [("PAD680", [7777, 9999]), ("SLD010H", [77, 99])]:
            self.assertTrue(clean.loc[raw[feature].isin(invalid_codes), feature].isna().all())
        self.assertTrue(clean.loc[raw.SLD010H.eq(9), "SLD010H"].eq(9).all())
        self.assertTrue(clean.loc[raw.SLD010H.eq(12), "SLD010H"].eq(12).all())
        self.assertTrue(clean.loc[raw.DIQ010.eq(3), "DIQ010"].eq("3").all())
        structural = clean.alcohol_status.isin([
            "under_12_lifetime_quantity_not_asked", "no_drinks_past_year"])
        self.assertTrue(clean.loc[structural, "alcohol_quantity_code"].eq(0).all())
        self.assertFalse(clean.loc[~structural, "alcohol_quantity_code"].eq(0).any())
        self.assertTrue(clean.loc[raw.ALQ130.eq(999), "alcohol_quantity_code"].isna().all())
        # F1 physiological inputs are exactly the existing baseline values.
        pd.testing.assert_frame_equal(raw[self.groups["P"]], clean[self.groups["P"]])
        self.assertFalse(set(clean) & {"SEQN", "severity", "PHQ9_total", "DPQ100"})
        self.assertFalse(any(column.startswith("DPQ") for column in clean))

    def test_fold_only_imputation_and_scaling(self):
        # Fit only the preprocessor on a REAL training fold and inspect its state.
        fold_ids = np.asarray(self.lock["validation_fold"])
        X_train = self.X.loc[fold_ids != 1]
        preprocessor = ABLATION["build_preprocessor"](self.X.columns.tolist(), self.final_inventory)
        preprocessor.fit(X_train)
        numerical = self.final_inventory.loc[
            self.final_inventory.feature_type.eq("numerical"), "feature"].tolist()
        fitted = preprocessor.named_transformers_["numerical"]
        np.testing.assert_allclose(fitted["imputer"].statistics_, X_train[numerical].median().to_numpy())
        imputed = fitted["imputer"].transform(X_train[numerical])
        np.testing.assert_allclose(fitted["scaler"].mean_, imputed.mean(axis=0))
        self.assertEqual(fitted["scaler"].n_samples_seen_, len(X_train))
        self.assertTrue(np.isfinite(preprocessor.transform(self.X.loc[fold_ids == 1])).all())

    def test_weight_audit_uses_training_counts_for_every_configuration(self):
        weights = pd.read_csv(experiment_path(ROOT / "results/tables", "xgboost_fold_weights.csv"))
        self.assertEqual(len(weights), 4 * 5 * 5)
        for (feature_set, fold), rows in weights.groupby(["feature_set", "fold"]):
            y_train = self.dev.loc[np.array(self.lock["validation_fold"]) != fold, "severity"]
            for row in rows.to_dict(orient="records"):
                count = int(y_train.eq(int(row["class"][-1])).sum())
                self.assertEqual(row["class_count_train"], count)
                self.assertAlmostEqual(row["sample_weight"], len(y_train) / (5 * count))
                self.assertAlmostEqual(row["class_weight_sum"], len(y_train) / 5)

    def test_oof_coverage_and_independent_metric_recalculation(self):
        results = pd.read_csv(experiment_path(ROOT / "results/tables", "feature_ablation_results.csv"))
        dummy = pd.read_csv(experiment_path(ROOT / "results/tables", "dummy_cv_results.csv"))
        all_results = pd.concat([results, dummy], ignore_index=True)
        self.assertEqual(len(results), 16)
        self.assertEqual(len(dummy), 2)
        self.assertEqual(len(self.fold_scores), 18 * 5)
        expected_folds = dict(zip(self.lock["development_seqn"], self.lock["validation_fold"]))
        classes = pd.read_csv(experiment_path(ROOT / "results/tables", "per_class_cv_results.csv"))
        matrices = pd.read_csv(experiment_path(ROOT / "results/tables", "ablation_confusion_matrices.csv"))
        self.assertEqual(len(classes), 18 * 5)
        for (feature_set, model), oof in self.oof.groupby(["feature_set", "model"]):
            self.assertEqual(len(oof), len(self.dev))
            self.assertTrue(oof.SEQN.is_unique)
            np.testing.assert_array_equal(oof.fold, oof.SEQN.map(expected_folds))
            expected_y = self.dev.set_index("SEQN").severity
            np.testing.assert_array_equal(oof.y_true, oof.SEQN.map(expected_y))
            recalculated = []
            for fold, subset in oof.groupby("fold"):
                values = ABLATION["summarize_metrics"](subset.y_true, subset.y_pred)
                saved = self.fold_scores.loc[
                    self.fold_scores.feature_set.eq(feature_set)
                    & self.fold_scores.model.eq(model) & self.fold_scores.fold.eq(fold)].iloc[0]
                for metric, value in values.items():
                    self.assertAlmostEqual(value, saved[metric])
                recalculated.append(values)
            saved_summary = all_results.loc[
                all_results.feature_set.eq(feature_set) & all_results.model.eq(model)].iloc[0]
            for metric in ABLATION["METRICS"]:
                values = [row[metric] for row in recalculated]
                self.assertAlmostEqual(np.mean(values), saved_summary[f"{metric}_mean"])
                self.assertAlmostEqual(np.std(values, ddof=1), saved_summary[f"{metric}_std"])
            _, recall, f1, support = precision_recall_fscore_support(
                oof.y_true, oof.y_pred, labels=np.arange(5), zero_division=0)
            saved_classes = classes[classes.feature_set.eq(feature_set) & classes.model.eq(model)].sort_values("class")
            np.testing.assert_allclose(saved_classes.recall, recall)
            np.testing.assert_allclose(saved_classes.f1, f1)
            np.testing.assert_array_equal(saved_classes.support, support)
            saved_matrix = matrices[matrices.feature_set.eq(feature_set) & matrices.model.eq(model)]
            saved_matrix = saved_matrix.pivot(index="true_class", columns="predicted_class", values="count")
            np.testing.assert_array_equal(saved_matrix, confusion_matrix(oof.y_true, oof.y_pred, labels=np.arange(5)))


if __name__ == "__main__":
    unittest.main()
