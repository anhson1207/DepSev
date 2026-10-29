"""A cloneable XGBoost estimator that balances ONLY the labels passed to fit."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from utils import save_json


def id_hash(ids):
    """Order-sensitive fingerprint of actual training participant IDs."""
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


class FoldBalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Weights are recomputed inside every inner fit and outer-training refit.

    Do not pass precomputed sample_weight to RandomizedSearchCV. Such weights
    would be sliced but would retain class frequencies from the wrong partition.
    y must retain its SEQN index when audit_dir is set.
    """

    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05,
                 min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                 gamma=0, reg_alpha=0, reg_lambda=1, random_state=42, n_jobs=1,
                 audit_dir=None, fit_context=""):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.audit_dir = audit_dir
        self.fit_context = fit_context

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("External sample_weight is forbidden; compute only from this fit's y")
        self.classes_, counts = np.unique(y, return_counts=True)
        if not np.array_equal(self.classes_, np.arange(5)):
            raise ValueError("Every tuning training partition must contain C0-C4")
        weights = compute_sample_weight("balanced", y)
        self.class_counts_ = counts
        self.class_weights_ = len(y) / (5 * counts)
        np.testing.assert_allclose(weights, self.class_weights_[np.asarray(y, dtype=int)])
        if self.audit_dir is not None:
            if not hasattr(y, "index") or not y.index.is_unique:
                raise ValueError("Audited fits require unique SEQN-indexed labels")
            self.training_ids_hash_ = id_hash(y.index)
        parameters = {key: value for key, value in self.get_params().items()
                      if key not in {"audit_dir", "fit_context"}}
        self.estimator_ = XGBClassifier(
            **parameters, objective="multi:softprob", num_class=5, eval_metric="mlogloss")
        self.estimator_.fit(X, y, sample_weight=weights)
        self.n_features_in_ = self.estimator_.n_features_in_
        if self.audit_dir is not None:
            params_hash = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
            path = Path(self.audit_dir) / f"{self.fit_context}_{params_hash}_{self.training_ids_hash_}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_json({"fit_context": self.fit_context, "params": parameters,
                       "training_ids_sha256": self.training_ids_hash_, "n_train": len(y),
                       "classes": self.classes_.tolist(), "class_counts": counts.tolist(),
                       "class_weights": self.class_weights_.tolist(),
                       "class_weight_sums": [float(weights[np.asarray(y) == c].sum()) for c in self.classes_],
                       "computed_inside_fit": True}, path)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)
