"""Ordinal Classification models for DepSev-ML.

Implements:
1. OrdinalLogisticRegression:
   Proportional Odds (Cumulative Logit) model with L2 regularization and
   guaranteed strictly monotonic thresholds via exponential reparameterization.
   scikit-learn compatible estimator.

2. CumulativeBinaryClassifier (Frank & Hall 2001):
   Fits two independent binary classifiers for P(Y >= 1) and P(Y >= 2).
   Detects monotonicity violations (q2 > q1) and resolves them via
   least-squares isotonic projection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight


class OrdinalLogisticRegression(BaseEstimator, ClassifierMixin):
    """Proportional Odds Cumulative Logit model for 3-class ordinal targets (0 < 1 < 2).

    Parameterization:
      P(Y <= 0 | x) = sigma(theta_0 - w^T x)
      P(Y <= 1 | x) = sigma(theta_1 - w^T x)
      P(Y <= 2 | x) = 1.0

    To strictly enforce theta_0 < theta_1, we reparameterize:
      theta_0 = alpha
      theta_1 = alpha + exp(delta), where delta in R.
      Thus theta_1 - theta_0 = exp(delta) > 0 strictly.

    Class probabilities:
      P(Y = 0 | x) = P(Y <= 0 | x)
      P(Y = 1 | x) = P(Y <= 1 | x) - P(Y <= 0 | x)
      P(Y = 2 | x) = 1.0 - P(Y <= 1 | x)

    Objective:
      Negative Log-Likelihood + 1/(2C) * ||w||^2.
      Thresholds alpha and delta are unregularized.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = 42,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> OrdinalLogisticRegression:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X_arr.shape}")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"X and y length mismatch: {len(X_arr)} vs {len(y_arr)}")

        self.classes_ = np.unique(y_arr)
        if not np.array_equal(self.classes_, np.array([0, 1, 2])):
            raise ValueError(f"OrdinalLogisticRegression currently expects classes [0, 1, 2], got {self.classes_}")

        N, d = X_arr.shape
        self.n_features_in_ = d

        # Sample weights
        if self.class_weight is not None:
            sample_weight = compute_sample_weight(self.class_weight, y_arr)
        else:
            sample_weight = np.ones(N, dtype=float)

        # Smart initialization based on empirical class frequencies
        n0 = np.sum(sample_weight[y_arr == 0])
        n1 = np.sum(sample_weight[y_arr == 1])
        n2 = np.sum(sample_weight[y_arr == 2])
        total_w = n0 + n1 + n2

        p0_marginal = np.clip(n0 / total_w, 1e-4, 1.0 - 1e-4)
        p01_marginal = np.clip((n0 + n1) / total_w, p0_marginal + 1e-4, 1.0 - 1e-4)

        alpha_init = float(logit(p0_marginal))
        theta1_init = float(logit(p01_marginal))
        diff_init = max(theta1_init - alpha_init, 1e-2)
        delta_init = float(np.log(diff_init))

        init_params = np.concatenate([np.zeros(d, dtype=float), [alpha_init, delta_init]])

        # Objective function with exact analytical gradients
        C_val = float(self.C)

        def objective(params: np.ndarray) -> Tuple[float, np.ndarray]:
            w = params[:d]
            alpha = params[d]
            delta = params[d + 1]
            diff = np.exp(np.clip(delta, -15.0, 15.0))
            th0 = alpha
            th1 = alpha + diff

            scores = X_arr @ w
            z0 = th0 - scores
            z1 = th1 - scores

            s0 = expit(z0)
            s1 = expit(z1)

            # Numerical stability guard
            eps = 1e-15
            p0 = np.clip(s0, eps, 1.0 - eps)
            p1 = np.clip(s1 - s0, eps, 1.0 - eps)
            p2 = np.clip(1.0 - s1, eps, 1.0 - eps)

            m0 = (y_arr == 0)
            m1 = (y_arr == 1)
            m2 = (y_arr == 2)

            w0 = sample_weight[m0]
            w1 = sample_weight[m1]
            w2 = sample_weight[m2]

            loss = -(
                np.sum(w0 * np.log(p0[m0])) +
                np.sum(w1 * np.log(p1[m1])) +
                np.sum(w2 * np.log(p2[m2]))
            )
            # L2 penalty on w (not on thresholds)
            loss += (0.5 / C_val) * np.sum(w ** 2)

            # Gradients with respect to z0 and z1
            g0 = np.zeros(N, dtype=float)
            g1 = np.zeros(N, dtype=float)

            # Class 0: d(-log s0)/dz0 = -(1 - s0)
            g0[m0] = -sample_weight[m0] * (1.0 - s0[m0])

            # Class 1: d(-log(s1 - s0))/dz
            ds0_1 = s0[m1] * (1.0 - s0[m1])
            ds1_1 = s1[m1] * (1.0 - s1[m1])
            g0[m1] = sample_weight[m1] * (ds0_1 / p1[m1])
            g1[m1] = -sample_weight[m1] * (ds1_1 / p1[m1])

            # Class 2: d(-log(1 - s1))/dz1 = s1
            g1[m2] = sample_weight[m2] * s1[m2]

            # Derivatives w.r.t parameters:
            # dz0/dw = -x, dz1/dw = -x
            # dz0/dalpha = 1, dz1/dalpha = 1
            # dz0/ddelta = 0, dz1/ddelta = diff
            grad_w = -X_arr.T @ (g0 + g1) + (1.0 / C_val) * w
            grad_alpha = float(np.sum(g0 + g1))
            grad_delta = float(diff * np.sum(g1))

            grad = np.concatenate([grad_w, [grad_alpha, grad_delta]])
            return loss, grad

        res = minimize(
            objective,
            init_params,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        self.coef_ = res.x[:d]
        alpha_opt = res.x[d]
        diff_opt = float(np.exp(np.clip(res.x[d + 1], -15.0, 15.0)))
        theta0_opt = alpha_opt
        theta1_opt = alpha_opt + diff_opt
        self.cutpoints_ = np.array([theta0_opt, theta1_opt])
        self.n_iter_ = res.nit
        self.converged_ = bool(res.success)
        self.opt_message_ = str(res.message)

        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        scores = X_arr @ self.coef_
        th0, th1 = self.cutpoints_

        s0 = expit(th0 - scores)
        s1 = expit(th1 - scores)

        p0 = s0
        p1 = np.maximum(0.0, s1 - s0)
        p2 = 1.0 - s1

        probs = np.column_stack([p0, p1, p2])
        # Clip and re-normalize strictly
        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class CumulativeBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Cumulative Binary Classification (Frank & Hall 2001) for 3-class ordinal targets.

    Classifier 1 models: P(Y >= 1 | X) -> binary target: 0 if Y=0, 1 if Y in {1, 2}.
    Classifier 2 models: P(Y >= 2 | X) -> binary target: 0 if Y in {0, 1}, 1 if Y=2.

    Monotonicity condition:
      Since {Y >= 2} subset {Y >= 1}, theoretically q2 <= q1.
      If independent binary estimators predict q2 > q1, an isotonic least-squares
      projection is applied:
        q1_proj = q2_proj = (q1 + q2) / 2.
    """

    def __init__(
        self,
        base_estimator: Any,
        apply_projection: bool = True,
    ):
        self.base_estimator = base_estimator
        self.apply_projection = apply_projection
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> CumulativeBinaryClassifier:
        y_arr = np.asarray(y, dtype=int)

        # Target 1: Y >= 1 (0 vs {1, 2})
        z1 = (y_arr >= 1).astype(int)
        self.clf1_ = clone(self.base_estimator)
        self.clf1_.fit(X, z1)

        # Target 2: Y >= 2 ({0, 1} vs 2)
        z2 = (y_arr >= 2).astype(int)
        self.clf2_ = clone(self.base_estimator)
        self.clf2_.fit(X, z2)

        return self

    def predict_cumulative_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw cumulative probabilities q1 = P(Y >= 1) and q2 = P(Y >= 2)."""
        # Scikit-learn binary predict_proba returns [P(0), P(1)]
        q1 = self.clf1_.predict_proba(X)[:, 1]
        q2 = self.clf2_.predict_proba(X)[:, 1]
        return q1, q2

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        q1, q2 = self.predict_cumulative_proba(X)
        q1 = np.clip(q1, 0.0, 1.0)
        q2 = np.clip(q2, 0.0, 1.0)

        # Track violation
        violation_mask = (q2 > q1)
        self.last_violation_rate_ = float(np.mean(violation_mask))
        self.last_violation_count_ = int(np.sum(violation_mask))

        q1_proj = q1.copy()
        q2_proj = q2.copy()

        if self.apply_projection and np.any(violation_mask):
            # Least-squares isotonic projection onto q2 <= q1
            avg = 0.5 * (q1[violation_mask] + q2[violation_mask])
            q1_proj[violation_mask] = avg
            q2_proj[violation_mask] = avg

        p_c0 = 1.0 - q1_proj
        p_c1 = np.maximum(0.0, q1_proj - q2_proj)
        p_c2 = q2_proj

        probs = np.column_stack([p_c0, p_c1, p_c2])
        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

