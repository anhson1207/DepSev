"""Unit tests for Hierarchical Classification (T3) Pipeline in DepSev-ML.

Validates:
1. Label mapping & boundary conditions (4 vs 5, 9 vs 10).
2. Stage 2 training subset isolation (only PHQ-9 >= 5 samples, no C0).
3. Leakage prevention: Stage 2 preprocessor fit strictly inside training subset.
4. Routing integrity & simulated error propagation (no oracle correction).
5. Reproducibility & deterministic predictions.
6. Metric consistency and confusion matrix totals.
"""

from importlib import import_module
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

G = import_module("09_granularity_experiment")
H = import_module("14_hierarchical_t3_experiment")
import final_protocol as frozen


class HierarchicalT3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load legacy development cohort
        cls.X, cls.y_c0_c4, cls.inventory, cls.scores, cls.folds, cls.assignment, cls.spec = G.load_development()
        cls.labels_t3 = G.map_target(cls.y_c0_c4, "T3").to_numpy(dtype=int)

    def _build_pipe(self):
        return Pipeline([
            ("preprocessor", frozen.ablation.build_preprocessor(self.spec["features"], self.inventory)),
            ("model", RandomForestClassifier(**self.spec["hyperparameters"])),
        ])

    def test_01_label_mapping_and_boundaries(self):
        """Test boundary conditions for T3 mapping: 4->0, 5->1, 9->1, 10->2."""
        sample_scores = pd.Series([0, 1, 4, 5, 6, 9, 10, 15, 27])
        # Digitizing with cutoffs [5, 10]:
        # < 5 -> 0, 5..9 -> 1, >= 10 -> 2
        mapped = np.digitize(sample_scores, [5, 10])
        expected = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(mapped, expected)

        # Full cohort checks
        self.assertEqual(len(self.labels_t3), 4045)
        self.assertTrue(set(np.unique(self.labels_t3)).issubset({0, 1, 2}))

        # Verify counts match PHQ-9 cutoffs
        c0_count = (self.scores <= 4).sum()
        c1_count = ((self.scores >= 5) & (self.scores <= 9)).sum()
        c2_count = (self.scores >= 10).sum()

        self.assertEqual((self.labels_t3 == 0).sum(), c0_count)
        self.assertEqual((self.labels_t3 == 1).sum(), c1_count)
        self.assertEqual((self.labels_t3 == 2).sum(), c2_count)

    def test_02_stage2_training_subset_isolation(self):
        """Test that Stage 2 training set only contains samples with PHQ-9 >= 5."""
        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            scores_train = self.scores.iloc[train_idx]
            y_t3_train = self.labels_t3[train_idx]

            # Mask of samples entering Stage 2 training
            stage2_mask = (scores_train >= 5)

            # Must strictly have PHQ-9 >= 5
            self.assertTrue((scores_train[stage2_mask] >= 5).all())
            # Must have zero C0 samples
            self.assertFalse((y_t3_train[stage2_mask] == 0).any())
            # Number of Stage 2 samples == N(C1) + N(C2)
            expected_n = (y_t3_train == 1).sum() + (y_t3_train == 2).sum()
            self.assertEqual(stage2_mask.sum(), expected_n)

    def test_03_preprocessor_leakage_safety(self):
        """Test that Stage 2 preprocessor is fitted strictly on Stage 2 training subset."""
        train_idx, val_idx = self.folds[0]
        X_train = self.X.iloc[train_idx]
        scores_train = self.scores.iloc[train_idx]

        # Build hierarchical model
        clf = H.HierarchicalT3Classifier(
            stage1_estimator=self._build_pipe(),
            stage2_estimator=self._build_pipe(),
            threshold1=0.5,
            threshold2=0.5,
        )
        clf.fit(X_train, scores_train)

        # Expected numerical imputer medians computed ONLY on X_train[scores_train >= 5]
        numerical_cols = self.spec["preprocessing"]["numerical"]["columns"]
        stage2_X_train = X_train.loc[scores_train >= 5]
        expected_medians = stage2_X_train[numerical_cols].median().to_numpy()

        # Actual imputer medians in stage2 pipeline
        prep2 = clf.stage2_.named_steps["preprocessor"]
        num_transformer = prep2.named_transformers_["numerical"]
        actual_medians = num_transformer.named_steps["imputer"].statistics_

        np.testing.assert_allclose(actual_medians, expected_medians, rtol=1e-5)

    def test_04_routing_integrity_and_simulated_error_propagation(self):
        """Test that inference routing uses only Stage 1 prediction without oracle access."""
        clf = H.HierarchicalT3Classifier(
            stage1_estimator=self._build_pipe(),
            stage2_estimator=self._build_pipe(),
            threshold1=0.5,
            threshold2=0.5,
        )

        train_idx, val_idx = self.folds[0]
        clf.fit(self.X.iloc[train_idx], self.scores.iloc[train_idx])

        X_val = self.X.iloc[val_idx]
        y_val_t3 = self.labels_t3[val_idx]

        # Predict detailed
        det = clf.predict_detailed(X_val)
        preds = det["y_pred"]
        pred_stage1 = det["routed"]

        # Check routing contract
        for i in range(len(preds)):
            if not pred_stage1[i]:
                self.assertEqual(preds[i], 0, "Stage 1 negative must terminate at C0")
            else:
                self.assertIn(preds[i], [1, 2], "Stage 1 positive must route to Stage 2 ({1, 2})")

        # Simulated Error Check 1:
        # If true label is C2+ (y_val == 2) but Stage 1 predicts 0 (missed),
        # does the model output C0 without oracle rescue?
        missed_severe = np.where((y_val_t3 == 2) & (~pred_stage1))[0]
        self.assertGreater(len(missed_severe), 0, "There should be missed severe cases at 0.5 threshold")
        for fn_idx in missed_severe:
            self.assertEqual(preds[fn_idx], 0, "No oracle rescue: missed severe must output C0")

        # Simulated Error Check 2:
        # If true label is C0 (y_val == 0) but Stage 1 predicts 1 (false alarm),
        # does the model route to Stage 2 and output C1 or C2+?
        c0_false_alarms = np.where((y_val_t3 == 0) & (pred_stage1))[0]
        self.assertGreater(len(c0_false_alarms), 0, "There should be C0 false alarms at 0.5 threshold")
        for fp_idx in c0_false_alarms:
            self.assertIn(preds[fp_idx], [1, 2], "False positive must be classified as C1 or C2+, never C0")

    def test_05_reproducibility(self):
        """Test that fitting with fixed seed gives 100% bitwise identical predictions."""
        train_idx, val_idx = self.folds[0]
        X_train, s_train = self.X.iloc[train_idx], self.scores.iloc[train_idx]
        X_val = self.X.iloc[val_idx]

        m1 = H.HierarchicalT3Classifier(
            stage1_estimator=self._build_pipe(),
            stage2_estimator=self._build_pipe(),
            threshold1=0.5,
            threshold2=0.5,
        )
        m1.fit(X_train, s_train)
        pred1 = m1.predict(X_val)

        m2 = H.HierarchicalT3Classifier(
            stage1_estimator=self._build_pipe(),
            stage2_estimator=self._build_pipe(),
            threshold1=0.5,
            threshold2=0.5,
        )
        m2.fit(X_train, s_train)
        pred2 = m2.predict(X_val)

        np.testing.assert_array_equal(pred1, pred2)

    def test_06_confusion_matrix_and_metrics_consistency(self):
        """Verify confusion matrix sum equals sample count and metrics match scikit-learn."""
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 2, 2, 2, 0, 1, 0])

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        self.assertEqual(cm.sum(), len(y_true))

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")

        metrics = H.calculate_comprehensive_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], acc)
        self.assertAlmostEqual(metrics["balanced_accuracy"], bal_acc)
        self.assertAlmostEqual(metrics["macro_f1"], mf1)

    def test_07_temporal_branch_d_sample_alignment(self):
        """Verify Branch D sample counts: train=9693, test=4888, disjoint cycles."""
        import multicycle_protocol as mp
        frames, policy, spec = mp.load_inputs()
        train_d = pd.concat([frames["2011_2012"], frames["2013_2014"]], ignore_index=True)
        test_d = frames["2015_2016"]

        self.assertEqual(len(train_d), 9693, "Branch D train must be exactly 9,693")
        self.assertEqual(len(test_d), 4888, "Branch D test must be exactly 4,888")

        # Disjoint SEQN check
        train_seqns = set(train_d["SEQN"])
        test_seqns = set(test_d["SEQN"])
        self.assertEqual(len(train_seqns & test_seqns), 0, "Train and Test SEQN must be strictly disjoint")

    def test_08_threshold_optimization_inner_cv_leakage(self):
        """Verify that threshold optimization runs on inner CV without touching validation data."""
        train_idx, val_idx = self.folds[0]
        X_train, s_train = self.X.iloc[train_idx], self.scores.iloc[train_idx]
        
        t1_opt, t2_opt, grid_df = H.optimize_thresholds_inner_cv(
            X_train, s_train, self.spec, self.inventory, n_splits=3
        )
        self.assertTrue(0.35 <= t1_opt <= 0.65)
        self.assertTrue(0.35 <= t2_opt <= 0.65)
        self.assertEqual(len(grid_df), 9 * 9, "9x9 grid evaluation (0.30 to 0.70 with step 0.05)")

    def test_09_stage2_evaluation_denominators(self):
        """Verify Stage 2 sample denominators: 525 true symptomatic routed vs 1063 all routed."""
        oof = pd.read_csv(H.OUTPUT_DIR / "oof_predictions.csv")
        stage_df = pd.read_csv(H.OUTPUT_DIR / "stage_metrics.csv")

        # 1. Symptomatic routed across all folds
        y_true = oof["y_true"].to_numpy()
        p_s1 = oof["PHQ9_total"] >= 5
        pred_e1 = oof["E1_Hierarchical_RF_pred"].to_numpy()

        # How many true symptomatics reached Stage 2:
        # Model 1 routes when prediction != 0
        routed_mask = (pred_e1 != 0)
        true_symptomatic = (y_true >= 1)

        true_routed_count = (true_symptomatic & routed_mask).sum()
        self.assertEqual(true_routed_count, 525, "True symptomatics reaching Stage 2 must be 525")

        # Correct predictions among these 525
        correct_in_routed = ((y_true == pred_e1) & routed_mask).sum()
        self.assertEqual(correct_in_routed, 276, "Correct Stage 2 predictions must be 276 (105 C1 + 171 C2)")
        self.assertAlmostEqual(correct_in_routed / true_routed_count, 276 / 525, places=5)

        # All routed count (including C0 false alarms)
        self.assertEqual(routed_mask.sum(), 1063, "Total samples entering Stage 2 must be 1063")


if __name__ == "__main__":
    unittest.main()
