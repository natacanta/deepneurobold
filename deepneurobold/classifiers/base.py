"""
classifiers.base
================
Abstract base class for all voxel-wise classifiers.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from deepneurobold.core.base import BaseComponent


@dataclass
class ClassifierResult:
    """Container for classifier outputs."""
    model: Any
    scaler: Optional[Any] = None
    classes_: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseClassifier(BaseComponent):
    """
    Abstract interface for voxel-wise supervised classifiers.

    All classifiers expose a unified ``train`` / ``predict_proba`` interface
    so that the experiment runner can swap them without code changes.

    Parameters
    ----------
    use_scaler : bool
        If True, apply ``StandardScaler`` before fitting.
    seed : int
        Random seed for reproducibility.
    config : dict, optional
        Runtime configuration.
    """

    def __init__(
        self,
        use_scaler: bool = True,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=self.__class__.__name__, config=config)
        self.use_scaler = use_scaler
        self.seed = seed

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> ClassifierResult:
        """
        Fit the classifier on training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,), int labels (0/1)

        Returns
        -------
        ClassifierResult
        """

    @abstractmethod
    def predict_proba(self, result: ClassifierResult, X: np.ndarray) -> np.ndarray:
        """
        Predict positive-class probability for each sample.

        Parameters
        ----------
        result : ClassifierResult
            Output of :meth:`train`.
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,), float32 in [0, 1]
        """

    def validate(self) -> bool:
        return True
