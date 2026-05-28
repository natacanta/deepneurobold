"""
classifiers.svm
===============
Linear SVM with Platt scaling for probability calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .base import BaseClassifier, ClassifierResult


# ---------------------------------------------------------------------------
# Legacy dataclass (kept for runner compatibility)
# ---------------------------------------------------------------------------

@dataclass
class SVMModel:
    """Thin wrapper around a calibrated LinearSVC."""
    clf: CalibratedClassifierCV
    scaler: Optional[StandardScaler]


# ---------------------------------------------------------------------------
# Standalone functions (mirrors original deepneurobold API)
# ---------------------------------------------------------------------------

def train_svm_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_scaler: bool,
    seed: int,
    calibration_cv: int = 2,
) -> SVMModel:
    """
    Fit a calibrated linear SVM.

    Default ``calibration_cv=2`` matches trial_143 / original deepneurobold behavior
    (chosen to avoid empty folds when one class is very small under aggressive
    sampling). ``trial_144_classifier_comparison`` calls this with
    ``calibration_cv=3`` per the publication spec.
    """
    scaler = None
    X = X_train
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_train)

    base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=int(seed),
        max_iter=5000,
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=int(calibration_cv))
    clf.fit(X, y_train)
    return SVMModel(clf=clf, scaler=scaler)


def predict_proba(model: SVMModel, X: np.ndarray) -> np.ndarray:
    """Return positive-class probability for each sample."""
    X2 = model.scaler.transform(X) if model.scaler is not None else X
    return model.clf.predict_proba(X2)


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class SVMClassifier(BaseClassifier):
    """
    Linear SVM with Platt probability calibration.

    Parameters
    ----------
    C : float
        Regularisation parameter.
    use_scaler : bool
        Apply StandardScaler before fitting (recommended for SVM).
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
        """Fit a calibrated linear SVM."""
        model = train_svm_linear(X_train, y_train, self.use_scaler, self.seed)
        return ClassifierResult(
            model=model.clf,
            scaler=model.scaler,
        )

    def predict_proba(self, result: ClassifierResult, X: np.ndarray) -> np.ndarray:
        """Return positive-class probability."""
        X2 = result.scaler.transform(X) if result.scaler is not None else X
        return result.model.predict_proba(X2)[:, 1].astype(np.float32)
