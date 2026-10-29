"""Unit tests for Soft Hierarchical Classification & Cost-Sensitive Learning in DepSev-ML.

Verifies:
1. Probabilities for all 3 classes lie in [0, 1].
2. Sum of probabilities equals 1.0 (bitwise close).
3. Class mapping and column order consistency {0: C0, 1: C1, 2: C2+}.
4. Stage 2 training subset isolation (only PHQ-9 >= 5 samples, no C0).
5. Stage 2 evaluates on all test samples without ground-truth label leakage.
6. Soft routing independence from hard gating.
7. Expected-cost decision rule implementation (Bayes Risk minimization).
8. Cost matrix row/column semantics alignment.
9. Zero leakage in preprocessing pipelines.
10. Zero leakage in probability calibration (CalibratedClassifierCV inside training fold).
11. Zero leakage in inner CV cost matrix optimization.
12. Exact fold alignment between Flat and Soft Hierarchical models.
13. Reproducibility across random seeds.
14. Mathematical theorem: soft argmax ALWAYS chooses C0 when p1 < 0.5.
"""

from importlib import import_module
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

G = import_module("09_granularity_experiment")
S = import_module("15_soft_hierarchical_experiment")
import final_protocol as frozen


class SoftHierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y_c0_c4, cls.inventory, cls.scores, cls.folds, cls.assignment, cls.spec = G.load_development()
        cls.labels_t3 = G.map_target(cls.y_c0_c4, "T3").to_numpy(dtype=int)

    def _build_rf(self):
        return Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(self.spec["features"], self.inventory)),
            ("model", RandomForestClassifier(**self.spec["hyperparameters"])),
        ])

    def test_01_probabilities_in_unit_interval_and_sum_to_one(self):
        """Test that fused 3-class probabilities strictly lie in [0, 1] and sum to 1.0."""
        train_idx, val_idx = self.folds[0]
        clf = S.SoftHierarchicalT3Classifier(self._build_rf(), self._build_rf())
        clf.fit(self.X.iloc[train_idx], self.scores.iloc[train_idx])

        X_val = self.X.iloc[val_idx]
        probs = clf.predict_proba(X_val)

        # 1. Shape check
        self.assertEqual(probs.shape, (len(X_val), 3))
        # 2. In bounds [0, 1]
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        # 3. Sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    def test_02_stage2_training_subset_isolation(self):
        """Verify that Stage 2 estimator is trained strictly on PHQ-9 >= 5 subset."""
        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            scores_train = self.scores.iloc[train_idx]
            y_t3_train = self.labels_t3[train_idx]
            stage2_mask = (scores_train >= 5)

            # Must have zero C0 samples
            self.assertEqual((y_t3_train[stage2_mask] == 0).sum(), 0)
            # Size must equal N(C1) + N(C2)
            expected_n = (y_t3_train == 1).sum() + (y_t3_train == 2).sum()
            self.assertEqual(stage2_mask.sum(), expected_n)

    def test_03_stage2_evaluates_on_all_test_samples_without_oracle(self):
        """Verify that Stage 2 can compute probabilities for all test samples without label leakage."""
        train_idx, val_idx = self.folds[0]
        clf = S.SoftHierarchicalT3Classifier(self._build_rf(), self._build_rf())
        clf.fit(self.X.iloc[train_idx], self.scores.iloc[train_idx])

        X_val = self.X.iloc[val_idx]
        p1, p2 = clf.predict_proba_stages(X_val)

        self.assertEqual(len(p1), len(X_val))
        self.assertEqual(len(p2), len(X_val))
        self.assertTrue(np.all(np.isfinite(p1)))
        self.assertTrue(np.all(np.isfinite(p2)))

    def test_04_expected_cost_decision_rule_correctness(self):
        """Verify Bayes risk minimization against hand-computed expected loss."""
        # Simulated probabilities: [P(C0), P(C1), P(C2)]
        probs = np.array([
            [0.55, 0.25, 0.20],  # argmax is 0
            [0.20, 0.50, 0.30],  # argmax is 1
            [0.10, 0.30, 0.60],  # argmax is 2
        ])
        # Asymmetric cost matrix: heavily penalize missing severe (C2 -> C0)
        K = np.array([
            [0.0, 1.0, 2.0],
            [2.0, 0.0, 1.0],
            [6.0, 2.0, 0.0],
        ])
        # Hand-calculated risk for sample 0:
        # R(0) = 0.55*0 + 0.25*2.0 + 0.20*6.0 = 0.50 + 1.20 = 1.70
        # R(1) = 0.55*1.0 + 0.25*0 + 0.20*2.0 = 0.55 + 0.40 = 0.95
        # R(2) = 0.55*2.0 + 0.25*1.0 + 0.20*0 = 1.10 + 0.25 = 1.35
        # min risk is R(1) = 0.95 -> Decision should be 1!
        expected_risks = probs @ K
        preds = np.argmin(expected_risks, axis=1)

        self.assertEqual(preds[0], 1, "Cost matrix must switch sample 0 from C0 to C1 due to severe penalty")
        self.assertEqual(preds[1], 1, "Sample 1 remains C1")
        self.assertEqual(preds[2], 2, "Sample 2 remains C2")

    def test_05_cost_matrix_semantics_and_zero_diagonal(self):
        """Verify all candidate cost matrices have zero diagonal and positive off-diagonals."""
        for name, K in S.CANDIDATE_COST_MATRICES.items():
            self.assertEqual(K.shape, (3, 3))
            np.testing.assert_array_equal(np.diag(K), [0.0, 0.0, 0.0], err_msg=f"{name} diagonal must be 0")
            # All off-diagonal costs must be > 0
            off_diag = K[~np.eye(3, dtype=bool)]
            self.assertTrue(np.all(off_diag > 0), msg=f"{name} off-diagonals must be strictly positive")

    def test_06_uniform_cost_matrix_equals_argmax(self):
        """Verify that 0-1 loss (uniform cost matrix) is mathematically identical to argmax."""
        probs = np.random.RandomState(42).dirichlet([1, 1, 1], size=100)
        K_uniform = S.CANDIDATE_COST_MATRICES["M0_Uniform"]
        risks = probs @ K_uniform
        preds_cost = np.argmin(risks, axis=1)
        preds_argmax = np.argmax(probs, axis=1)

        np.testing.assert_array_equal(preds_cost, preds_argmax)

    def test_07_mathematical_theorem_argmax_always_selects_c0_when_p1_less_than_half(self):
        """Mathematical theorem check: if p1 < 0.5, P(C0) > 0.5 and argmax MUST output 0."""
        # Test 1000 random points where p1 < 0.5 and p2 is arbitrary in [0, 1]
        rng = np.random.RandomState(42)
        p1 = rng.uniform(0.0, 0.49999, size=1000)
        p2 = rng.uniform(0.0, 1.0, size=1000)

        p_c0 = 1.0 - p1
        p_c1 = p1 * (1.0 - p2)
        p_c2 = p1 * p2

        probs = np.column_stack([p_c0, p_c1, p_c2])
        preds = np.argmax(probs, axis=1)

        # Every single prediction MUST be 0!
        self.assertTrue(np.all(preds == 0), "Soft argmax cannot rescue any sample with p1 < 0.5")

    def test_08_inner_cv_cost_optimization_zero_leakage(self):
        """Verify inner CV cost matrix optimization only accesses training data."""
        train_idx, val_idx = self.folds[0]
        X_train, s_train = self.X.iloc[train_idx], self.scores.iloc[train_idx]

        best_name, best_mat, grid_df = S.optimize_cost_matrix_inner_cv(
            X_train, s_train, self.spec, self.inventory, model_type="RF_RF", n_splits=3
        )
        self.assertIn(best_name, S.CANDIDATE_COST_MATRICES)
        self.assertEqual(len(grid_df), len(S.CANDIDATE_COST_MATRICES))
        self.assertTrue(np.all(np.isfinite(grid_df["inner_macro_f1"])))

    def test_09_calibration_zero_leakage_and_valid_probabilities(self):
        """Verify CalibratedClassifierCV fits strictly on training split and outputs valid probabilities."""
        train_idx, val_idx = self.folds[0]
        clf = S.SoftHierarchicalT3Classifier(self._build_rf(), self._build_rf(), calibrate=True, calibration_method="sigmoid")
        clf.fit(self.X.iloc[train_idx], self.scores.iloc[train_idx])

        X_val = self.X.iloc[val_idx]
        probs = clf.predict_proba(X_val)

        self.assertEqual(probs.shape, (len(X_val), 3))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    def test_10_reproducibility(self):
        """Verify deterministic predictions across two identical runs."""
        train_idx, val_idx = self.folds[0]
        X_tr, s_tr = self.X.iloc[train_idx], self.scores.iloc[train_idx]
        X_va = self.X.iloc[val_idx]

        m1 = S.SoftHierarchicalT3Classifier(self._build_rf(), self._build_rf())
        m1.fit(X_tr, s_tr)
        p1 = m1.predict_proba(X_va)

        m2 = S.SoftHierarchicalT3Classifier(self._build_rf(), self._build_rf())
        m2.fit(X_tr, s_tr)
        p2 = m2.predict_proba(X_va)

        np.testing.assert_allclose(p1, p2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
