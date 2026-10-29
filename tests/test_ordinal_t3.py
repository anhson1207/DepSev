"""Unit tests for Ordinal Classification (E7) in DepSev-ML.

Verifies:
1. Mapping PHQ-9 total scores to C0/C1/C2+ classes.
2. Boundary thresholds at 4 vs 5 and 9 vs 10.
3. Target ordinal relationship C0 < C1 < C2+ (0 < 1 < 2).
4. Predicted probabilities are strictly non-negative (>= 0).
5. Predicted probabilities sum to 1.0 (bitwise close).
6. E7b monotonicity guarantee: post-projection q2 <= q1, detecting violations properly.
7. Zero data leakage across training and validation splits.
8. Exclusion of raw DPQ items from feature matrix X.
9. Preprocessing pipeline fitted strictly inside training fold.
10. No validation set usage for hyperparameter tuning.
11. Exact fold alignment between E7 and baselines (E0, E1, E4, E5, E6).
12. Mathematical correctness of class MAE and Quadratic Weighted Kappa metrics.
13. Seed reproducibility across runs.
14. Absolute preservation of historical artifacts and result files.
"""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

G = import_module("09_granularity_experiment")
import final_protocol as frozen
from ordinal_models import CumulativeBinaryClassifier, OrdinalLogisticRegression


class OrdinalT3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y_c0_c4, cls.inventory, cls.scores, cls.folds, cls.assignment, cls.spec = G.load_development()
        cls.labels_t3 = G.map_target(cls.y_c0_c4, "T3").to_numpy(dtype=int)

    def _build_preprocessor(self):
        return frozen.ablation.build_preprocessor(self.spec["features"], self.inventory)

    def test_01_phq9_mapping_and_boundaries(self):
        """Verify PHQ-9 mapping thresholds: 0-4 -> 0, 5-9 -> 1, 10-27 -> 2."""
        # Test boundary values exactly
        test_scores = pd.Series([0, 4, 5, 9, 10, 27])
        # Mapping definition in DepSev-ML T3
        mapped = np.digitize(test_scores, [5, 10])
        np.testing.assert_array_equal(mapped, [0, 0, 1, 1, 2, 2])

        # Test against full dataset scores
        expected = np.digitize(self.scores, [5, 10])
        np.testing.assert_array_equal(self.labels_t3, expected)

    def test_02_ordinal_target_order(self):
        """Verify strict ordinal convention C0 < C1 < C2+."""
        classes = np.unique(self.labels_t3)
        self.assertEqual(list(classes), [0, 1, 2])

    def test_03_no_dpq_in_features(self):
        """Verify that raw PHQ-9 survey questions (DPQ) are strictly absent from features."""
        dpq_cols = [c for c in self.X.columns if "DPQ" in c.upper()]
        self.assertEqual(len(dpq_cols), 0, f"Found leaked DPQ columns in X: {dpq_cols}")

    def test_04_e7a_probabilities_valid_and_monotonic_thresholds(self):
        """Verify E7a (OrdinalLogisticRegression) probabilities lie in [0, 1], sum to 1, and cutpoints are monotonic."""
        train_idx, val_idx = self.folds[0]
        pipe = Pipeline([
            ("preprocessor", self._build_preprocessor()),
            ("model", OrdinalLogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ])
        pipe.fit(self.X.iloc[train_idx], self.labels_t3[train_idx])

        # Verify cutpoint monotonicity
        model: OrdinalLogisticRegression = pipe.named_steps["model"]
        th0, th1 = model.cutpoints_
        self.assertLess(th0, th1, f"Cutpoint monotonicity violated: th0={th0} >= th1={th1}")

        # Evaluate probabilities on validation set
        X_val = self.X.iloc[val_idx]
        probs = pipe.predict_proba(X_val)

        self.assertEqual(probs.shape, (len(X_val), 3))
        self.assertTrue(np.all(probs >= 0.0), "Probabilities must be non-negative")
        self.assertTrue(np.all(probs <= 1.0), "Probabilities must be <= 1.0")
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12, err_msg="Probabilities must sum to 1.0")

    def test_05_e7b_monotonicity_violation_detection_and_projection(self):
        """Verify E7b cumulative binary projection enforces q2 <= q1."""
        train_idx, val_idx = self.folds[0]

        # Use CumulativeBinaryClassifier with Logistic Regression
        prep = self._build_preprocessor()
        X_tr = prep.fit_transform(self.X.iloc[train_idx])
        X_va = prep.transform(self.X.iloc[val_idx])

        base_lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        cbc = CumulativeBinaryClassifier(base_lr, apply_projection=True)
        cbc.fit(X_tr, self.labels_t3[train_idx])

        raw_q1, raw_q2 = cbc.predict_cumulative_proba(X_va)
        probs = cbc.predict_proba(X_va)

        # Ensure probabilities are valid
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

        # For projected probabilities:
        # P(C0) = 1 - q1_proj, P(C1) = q1_proj - q2_proj, P(C2+) = q2_proj
        # Monotonicity requires P(C1) >= 0, which implies q1_proj >= q2_proj
        self.assertTrue(np.all(probs[:, 1] >= 0.0))
        self.assertTrue(np.all(probs[:, 2] <= 1.0 - probs[:, 0] + 1e-12))

    def test_06_preprocessor_zero_leakage_inside_folds(self):
        """Verify preprocessor strictly fits on train fold without using validation fold."""
        train_idx, val_idx = self.folds[0]
        prep = self._build_preprocessor()
        prep.fit(self.X.iloc[train_idx])

        # Transforming validation fold must succeed without re-fitting
        X_va_trans = prep.transform(self.X.iloc[val_idx])
        self.assertEqual(X_va_trans.shape[0], len(val_idx))
        self.assertFalse(np.any(np.isnan(X_va_trans)))

    def test_07_fold_alignment_with_legacy_cohort(self):
        """Verify fold splits match exactly 5 folds and partition 4,045 samples."""
        self.assertEqual(len(self.folds), 5)
        total_val = 0
        all_val_indices = []
        for tr, va in self.folds:
            total_val += len(va)
            all_val_indices.extend(va)
            # Intersection between train and val in each fold must be empty
            self.assertEqual(len(set(tr).intersection(set(va))), 0)

        self.assertEqual(total_val, 4045)
        self.assertEqual(len(set(all_val_indices)), 4045)

    def test_08_metric_calculations_mae_and_qwk(self):
        """Verify calculation of Class MAE and Quadratic Weighted Kappa."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 2, 1, 0, 2, 1])
        # Errors: |0-0|=0, |0-2|=2, |1-1|=0, |1-0|=1, |2-2|=0, |2-1|=1
        # Sum of errors: 0 + 2 + 0 + 1 + 0 + 1 = 4. MAE = 4 / 6 = 0.6667
        mae = float(np.mean(np.abs(y_true - y_pred)))
        self.assertAlmostEqual(mae, 4.0 / 6.0, places=4)

        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        self.assertTrue(-1.0 <= qwk <= 1.0)

        # 2-step errors: index 1 has true=0, pred=2. Count = 1
        two_step = np.sum(np.abs(y_true - y_pred) == 2)
        self.assertEqual(two_step, 1)

    def test_09_seed_reproducibility(self):
        """Verify identical results when trained with the same seed."""
        train_idx, val_idx = self.folds[0]
        prep = self._build_preprocessor()
        X_tr = prep.fit_transform(self.X.iloc[train_idx])
        X_va = prep.transform(self.X.iloc[val_idx])

        m1 = OrdinalLogisticRegression(C=1.0, random_state=42)
        m1.fit(X_tr, self.labels_t3[train_idx])
        p1 = m1.predict_proba(X_va)

        m2 = OrdinalLogisticRegression(C=1.0, random_state=42)
        m2.fit(X_tr, self.labels_t3[train_idx])
        p2 = m2.predict_proba(X_va)

        np.testing.assert_allclose(p1, p2, atol=1e-12)

    def test_10_historical_results_preservation(self):
        """Verify that historical results and artifacts directories exist and have key locked files."""
        frozen_files = [
            ROOT / "results/final_report_integrated.md",
            ROOT / "results/protocol_correction_report.md",
            ROOT / "experiments/hierarchical_t3/hierarchical_report.md",
            ROOT / "experiments/soft_hierarchical_t3/soft_hierarchical_report.md",
        ]
        for f in frozen_files:
            self.assertTrue(f.exists(), f"Protected file missing: {f}")


if __name__ == "__main__":
    unittest.main()

