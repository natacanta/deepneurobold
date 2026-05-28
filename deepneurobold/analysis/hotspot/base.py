"""
analysis.hotspot.base
=====================
Abstract base class for hotspot detection strategies.

The hotspot detection pipeline:

1. Joint three-channel probability map (mean of necrosis / enhancing /
   enh+necrosis channels) per voxel.
2. Adaptive percentile threshold computed on the **test region** (edema +
   30 mm ring around tumor core, excluding core).
3. Connected components on thresholded joint map.
4. Filter by minimum size (≥150 vox), minimum distance from core (≥5 mm),
   and FDR-corrected z-test vs. test-region background.
5. Three-channel qualifier — dominant tumour subtype per surviving cluster.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from deepneurobold.core.base import BaseComponent


class BaseHotspotDetector(BaseComponent):
    """
    Abstract interface for detecting pre-lesional tumor infiltration hotspots.

    A hotspot is a spatially coherent cluster of voxels in the perilesional
    zone (edema + 30 mm ring around tumor core) whose joint probability of
    resembling tumor tissue is significantly elevated relative to the
    surrounding test-region background.

    Parameters
    ----------
    run_dir : Path
        Output directory for this experiment run.
    brain_mask_flat : np.ndarray, bool
        Flattened brain mask, shape (n_voxels,).
    test_region_flat : np.ndarray, bool
        Flattened test region mask (edema + 30 mm band, excluding core).
    edema_flat : np.ndarray, bool
        Flattened edema mask.
    vol_shape_3d : tuple
        3-D volume shape (x, y, z).
    ref_img : nib.Nifti1Image
        Reference image for affine/header.
    config : dict, optional
        Runtime configuration.
    """

    def __init__(
        self,
        run_dir: Path,
        brain_mask_flat: np.ndarray,
        test_region_flat: np.ndarray,
        edema_flat: np.ndarray,
        vol_shape_3d: Tuple[int, int, int],
        ref_img: nib.Nifti1Image,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=self.__class__.__name__, config=config)
        self.run_dir          = Path(run_dir)
        self.brain_mask_flat  = np.asarray(brain_mask_flat, dtype=bool).ravel()
        self.test_region_flat = np.asarray(test_region_flat, dtype=bool).ravel()
        self.edema_flat       = np.asarray(edema_flat, dtype=bool).ravel()
        self.vol_shape_3d     = tuple(int(x) for x in vol_shape_3d)
        self.ref_img          = ref_img

    @abstractmethod
    def compute_joint_probability_map(
        self, prob_maps: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Average individual channel probability maps into a joint map.

        Each voxel receives three independent probability scores (necrosis,
        enhancing, enhancing+necrosis). The joint map is used for cluster
        definition and ranking; individual channels are used only to label
        the dominant tissue subtype within each surviving cluster.

        Parameters
        ----------
        prob_maps : dict
            Mapping ``run_name → prob_flat`` for available channels.

        Returns
        -------
        np.ndarray, shape (n_voxels,), float32
        """

    @abstractmethod
    def detect_clusters(
        self,
        joint_prob_flat: np.ndarray,
        test_region_flat: np.ndarray,
        tumor_core_flat: Optional[np.ndarray],
        percentile: int,
    ) -> pd.DataFrame:
        """
        Threshold → connected components → size/distance/FDR filter → rank.

        Parameters
        ----------
        joint_prob_flat : np.ndarray
        test_region_flat : np.ndarray, bool
        tumor_core_flat : np.ndarray, bool or None
        percentile : int

        Returns
        -------
        pd.DataFrame
            Columns: rank, size_voxels, volume_ml, mean_prob, min_dist_mm,
            pct_rank_in_test, p_raw, p_bh_adj, sig_fdr,
            prob_necrosis, prob_enhancing, prob_enh_nec, dominant_class.
        """

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Run hotspot detection across all percentiles and save NIfTI outputs."""

    def validate(self) -> bool:
        return self.run_dir.exists()
