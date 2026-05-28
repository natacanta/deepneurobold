"""
classifiers.logistic_regression
================================
L2-regularised logistic regression classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseClassifier, ClassifierResult


# ---------------------------------------------------------------------------
# Legacy dataclass (kept for runner compatibility)
# ---------------------------------------------------------------------------

@dataclass
class LogRegModel:
    """Thin wrapper around a fitted LogisticRegression."""
    clf: LogisticRegression
    scaler: Optional[StandardScaler]


# ---------------------------------------------------------------------------
# Standalone functions (mirrors original deepneurobold API)
# ---------------------------------------------------------------------------

def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_scaler: bool,
    seed: int,
) -> LogRegModel:
    """Fit L2-regularised logistic regression with balanced class weights."""
    scaler = None
    X = X_train
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=int(seed),
    )
    clf.fit(X, y_train)
    return LogRegModel(clf=clf, scaler=scaler)


def predict_proba(model: LogRegModel, X: np.ndarray) -> np.ndarray:
    """Return positive-class probability for each sample."""
    X2 = model.scaler.transform(X) if model.scaler is not None else X
    return model.clf.predict_proba(X2)


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class LogisticRegressionClassifier(BaseClassifier):
    """
    L2-regularised logistic regression with balanced class weights.

    Parameters
    ----------
    C : float
        Inverse regularisation strength.
    use_scaler : bool
        Apply StandardScaler before fitting (recommended).
    seed : int
    config : dict, optional
    """

    def __init__(
        self,
        C: float = 1.0,
        use_scaler: bool = True,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(use_scaler=use_scaler, seed=seed, config=config)
        self.C = C

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> ClassifierResult:
        """Fit L2 logistic regression."""
        model = train_logreg(X_train, y_train, self.use_scaler, self.seed)
        return ClassifierResult(
            model=model.clf,
            scaler=model.scaler,
            classes_=model.clf.classes_,
        )

    def predict_proba(self, result: ClassifierResult, X: np.ndarray) -> np.ndarray:
        """Return positive-class probability."""
        X2 = result.scaler.transform(X) if result.scaler is not None else X
        return result.model.predict_proba(X2)[:, 1].astype(np.float32)
