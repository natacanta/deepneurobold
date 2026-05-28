"""
classifiers.random_forest
=========================
Balanced Random Forest classifier with proximity matrix and feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier as _SKLearnRF
from sklearn.preprocessing import StandardScaler

from .base import BaseClassifier, ClassifierResult


# ---------------------------------------------------------------------------
# Legacy dataclass (kept for runner compatibility)
# ---------------------------------------------------------------------------

@dataclass
class RFModel:
    """Thin wrapper around a fitted RandomForestClassifier."""
    clf: _SKLearnRF
    scaler: Optional[StandardScaler]


# ---------------------------------------------------------------------------
# Standalone functions (mirrors original deepneurobold API)
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_scaler: bool,
    seed: int,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
) -> RFModel:
    """Fit a balanced RF. Scaler is optional (not needed for trees)."""
    scaler = None
    X = X_train
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_train)

    clf = _SKLearnRF(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(seed),
    )
    clf.fit(X, y_train)
    return RFModel(clf=clf, scaler=scaler)


def predict_proba(model: RFModel, X: np.ndarray) -> np.ndarray:
    """Return positive-class probability for each sample."""
    X2 = model.scaler.transform(X) if model.scaler is not None else X
    return model.clf.predict_proba(X2)


def get_feature_importance(model: RFModel) -> np.ndarray:
    """Return mean decrease in impurity per feature (BOLD timepoint)."""
    return model.clf.feature_importances_


def get_sample_proximity(model: RFModel, X: np.ndarray) -> np.ndarray:
    """
    Compute the RF proximity matrix for *X*.

    Two samples are proximate when they end up in the same leaf node across
    all trees.  The value is the fraction of trees where this co-occurrence
    happens (in [0, 1]).

    Note: run on a balanced subsample — do not use on the whole brain.
    """
    X2 = model.scaler.transform(X) if model.scaler is not None else X
    terminals = model.clf.apply(X2)  # (n_samples, n_estimators)
    n_samples = X2.shape[0]
    prox = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(model.clf.n_estimators):
        tree_nodes = terminals[:, i]
        prox += (tree_nodes[:, None] == tree_nodes[None, :]).astype(np.float32)
    return prox / model.clf.n_estimators


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class RandomForestClassifier(BaseClassifier):
    """
    Balanced Random Forest for voxel-wise tumor probability estimation.

    Parameters
    ----------
    n_estimators : int
    use_scaler : bool
    seed : int
    config : dict, optional
    """

    def __init__(
        self,
        n_estimators: int = 500,
        use_scaler: bool = False,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(use_scaler=use_scaler, seed=seed, config=config)
        self.n_estimators = n_estimators

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> ClassifierResult:
        """Fit a balanced Random Forest."""
        model = train_random_forest(X_train, y_train, self.use_scaler, self.seed)
        return ClassifierResult(
            model=model.clf,
            scaler=model.scaler,
            classes_=model.clf.classes_,
        )

    def predict_proba(self, result: ClassifierResult, X: np.ndarray) -> np.ndarray:
        """Return positive-class probability."""
        X2 = result.scaler.transform(X) if result.scaler is not None else X
        return result.model.predict_proba(X2)[:, 1].astype(np.float32)

    def feature_importance(self, result: ClassifierResult) -> np.ndarray:
        """Return mean decrease in impurity per feature."""
        return result.model.feature_importances_

    def proximity_matrix(self, result: ClassifierResult, X: np.ndarray) -> np.ndarray:
        """
        Compute the RF proximity matrix for *X*.

        Returns
        -------
        np.ndarray, shape (n_samples, n_samples), float32 in [0, 1]
        """
        X2 = result.scaler.transform(X) if result.scaler is not None else X
        terminals = result.model.apply(X2)
        n_samples = X2.shape[0]
        prox = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(result.model.n_estimators):
            nodes = terminals[:, i]
            prox += (nodes[:, None] == nodes[None, :]).astype(np.float32)
        return prox / result.model.n_estimators
