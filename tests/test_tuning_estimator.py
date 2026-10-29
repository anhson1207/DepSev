"""Preflight checks on real development partitions, never synthetic or test data."""

from importlib import import_module
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ablation_features import prepare_features
from tuning_estimators import FoldBalancedXGBClassifier, id_hash

TUNING = import_module("05_nested_tuning")


class TuningEstimatorTests(unittest.TestCase):
    def test_clone_and_external_weight_rejection(self):
        estimator = FoldBalancedXGBClassifier(n_estimators=2)
        self.assertTrue(is_classifier(estimator))
        self.assertEqual(clone(estimator).get_params(), estimator.get_params())
        with self.assertRaisesRegex(ValueError, "External sample_weight"):
            estimator.fit(None, None, sample_weight=[])

    def test_weights_and_preprocessing_on_real_inner_training_only(self):
        development, raw_inventory = TUNING.ablation.load_inputs()
        X, groups, inventory, _ = prepare_features(development, raw_inventory)
        ids = development.SEQN.astype(int)
        X.index = ids
        y = pd.Series(development.severity.to_numpy(), index=ids)
        outer, _ = TUNING.ablation.lock_folds(development)
        outer_train, outer_validation = outer[0]
        inner_train, inner_validation = next(StratifiedKFold(3, shuffle=True, random_state=42).split(
            X.iloc[outer_train], y.iloc[outer_train]))
        train = outer_train[inner_train]
        features = [f for g in "PCLD" for f in groups[g]]
        with tempfile.TemporaryDirectory(prefix="depsev_tuning_test_") as audit_dir:
            pipeline = TUNING.make_pipeline("XGBoost", features, inventory, "preflight")
            pipeline.set_params(model__n_estimators=2, model__audit_dir=audit_dir)
            pipeline.fit(X.iloc[train][features], y.iloc[train])
            fitted = pipeline.named_steps["model"]
            counts = y.iloc[train].value_counts().sort_index().to_numpy()
            np.testing.assert_array_equal(fitted.class_counts_, counts)
            np.testing.assert_allclose(fitted.class_weights_, len(train) / (5 * counts))
            self.assertEqual(fitted.training_ids_hash_, id_hash(y.iloc[train].index))
            self.assertFalse(set(y.iloc[train].index) & set(y.iloc[outer_validation].index))
            numerical = inventory.loc[inventory.feature_type.eq("numerical"), "feature"].tolist()
            preprocessing = pipeline.named_steps["preprocessor"].named_transformers_["numerical"]
            np.testing.assert_allclose(preprocessing["imputer"].statistics_, X.iloc[train][numerical].median())
            self.assertEqual(preprocessing["scaler"].n_samples_seen_, len(train))
            predictions = pipeline.predict(X.iloc[outer_train[inner_validation]][features])
            self.assertEqual(len(predictions), len(inner_validation))
            logs = list(Path(audit_dir).glob("*.json"))
            self.assertEqual(len(logs), 1)
            payload = json.loads(logs[0].read_text())
            self.assertTrue(payload["computed_inside_fit"])
            self.assertEqual(payload["n_train"], len(train))


if __name__ == "__main__":
    unittest.main()
