"""
features.representations
========================
Factory that returns the correct feature extractor for a given representation name.

Supported representations
--------------------------
ts_raw           — raw BOLD time series (no normalisation)
ts_normalized    — z-scored BOLD time series per voxel
ts_preprocessed  — detrended + Gaussian temporal smooth + z-score
bold_features    — 10 engineered temporal features per voxel
full_17          — 10-feature set (extended set placeholder)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseFeatureExtractor
from .bold_features import BoldFeatureExtractor, extract_features, extract_features_compat

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    "ts_raw":           BoldFeatureExtractor,
    "ts_normalized":    BoldFeatureExtractor,
    "ts_preprocessed":  BoldFeatureExtractor,
    "bold_features":    BoldFeatureExtractor,
    "bold_features_10": BoldFeatureExtractor,
    "full_17":          BoldFeatureExtractor,
    "bold_features_17": BoldFeatureExtractor,
}

VALID_REPRESENTATIONS = tuple(_REGISTRY.keys())


def get_feature_extractor(
    representation: str,
    dt: float = 1.8,
    patient_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> BaseFeatureExtractor:
    """
    Return the :class:`BaseFeatureExtractor` instance for *representation*.

    Parameters
    ----------
    representation : str
    dt : float
        BOLD TR in seconds.
    patient_dir : Path, optional
    config : dict, optional

    Raises
    ------
    ValueError
        If *representation* is not in the registry.
    """
    key = representation.lower().strip()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown representation {representation!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return cls(representation=key, dt=dt, patient_dir=patient_dir, config=config)


__all__ = [
    "get_feature_extractor",
    "extract_features",
    "extract_features_compat",
    "VALID_REPRESENTATIONS",
]
