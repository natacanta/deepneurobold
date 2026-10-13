"""Multi-layer perceptron (shallow, sklearn) for classifier comparison.

Added on top of the existing RF / linear SVM / L2 logistic regression
suite as a sanity check that a small neural network does not outperform
the Random Forest in this setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class MLPModel:
    clf: MLPClassifier
    scaler: Optional[StandardScaler]


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_scaler: bool,
    seed: int,
) -> MLPModel:
    """Train a shallow multi-layer perceptron.

    Architecture: two hidden layers (32, 16), ReLU activation, Adam optimizer.
    Feature scaling is mandatory for MLP; the `use_scaler` argument is kept
    for API consistency but is always applied here.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X_train)

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(256, len(X)),
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=int(seed),
    )
    clf.fit(X, y_train)
    return MLPModel(clf=clf, scaler=scaler)


def predict_proba(model: MLPModel, X: np.ndarray) -> np.ndarray:
    if model.scaler is not None:
        X = model.scaler.transform(X)
    return model.clf.predict_proba(X).astype(np.float32)
