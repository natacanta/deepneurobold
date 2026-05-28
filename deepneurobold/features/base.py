"""
features.base
=============
Abstract base class for BOLD feature extractors.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from deepneurobold.core.base import BaseComponent


class BaseFeatureExtractor(BaseComponent):
    """
    Extract a feature matrix from a 4-D BOLD volume.

    Every feature representation (raw time series, FFT features, lag maps,
    etc.) subclasses :class:`BaseFeatureExtractor` and implements
    :meth:`extract`.

    Parameters
    ----------
    representation : str
        Name of the feature representation (e.g. ``"ts_preprocessed"``,
        ``"bold_features_10"``).
    dt : float
        BOLD repetition time (TR) in seconds.
    patient_dir : Path, optional
        Patient directory (needed for representations that load auxiliary files).
    config : dict, optional
        Runtime configuration.
    """

    def __init__(
        self,
        representation: str,
        dt: float = 1.8,
        patient_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=f"FeatureExtractor[{representation}]", config=config)
        self.representation = representation
        self.dt = dt
        self.patient_dir = Path(patient_dir) if patient_dir else None

    @abstractmethod
    def extract(self, bold_4d: Any, mask_flat: np.ndarray) -> np.ndarray:
        """
        Extract features for all voxels within the mask.

        Parameters
        ----------
        bold_4d : nibabel.Nifti1Image
            4-D BOLD volume.
        mask_flat : np.ndarray, bool, shape (n_voxels,)
            Flattened brain mask; only masked voxels are processed.

        Returns
        -------
        np.ndarray, shape (n_masked_voxels, n_features)
            Feature matrix, rows ordered by mask flatnonzero index.
        """

    def validate(self) -> bool:
        return bool(self.representation)
