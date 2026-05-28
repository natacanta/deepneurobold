"""
data.loader
===========
BOLD fMRI and brain mask loader with branch-aware path resolution.

Branch selection priority:
  1. ``DNB_BOLD_BRANCH_DIR`` env var (absolute path to a preprocessing branch dir)
  2. ``DNB_BOLD_BRANCH`` env var in ``{"nomcst", "mcst"}``
  3. Legacy default: ``PREPROCESSING/bold``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import nibabel as nib
import numpy as np

from .base import BaseDataLoader

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
BoldBranch = Literal["nomcst", "mcst"]
BoldSource = Literal[
    "in_t1",
    "brain_in_t1",
    "brain_in_t1_spatial",
    "brain_in_t1_spatiotemporal",
    "preprocessed",
    "spatiotemporal",
    "st_preprocessed",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_branch_dir(patient_dir: Path) -> Path:
    patient_dir = Path(patient_dir)

    env_dir = os.environ.get("DNB_BOLD_BRANCH_DIR", "").strip()
    if env_dir:
        return Path(env_dir)

    env_branch = os.environ.get("DNB_BOLD_BRANCH", "").strip().lower()
    if env_branch in ("nomcst", "mcst"):
        return patient_dir / "PREPROCESSING" / "bold" / f"branch_{env_branch}"

    return patient_dir / "PREPROCESSING" / "bold"


def _resolve_bold_path(patient_dir: Path, source: BoldSource) -> Path:
    bdir = _get_branch_dir(patient_dir)

    if source == "in_t1":
        return bdir / "BOLD_inT1.nii.gz"
    if source == "brain_in_t1":
        return bdir / "BOLD_brain_inT1.nii.gz"
    if source == "brain_in_t1_spatial":
        return bdir / "BOLD_brain_inT1_spatial_smooth.nii.gz"
    if source == "brain_in_t1_spatiotemporal":
        return bdir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz"
    if source in ("preprocessed", "spatiotemporal", "st_preprocessed"):
        return bdir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz"

    raise ValueError(f"Unknown BoldSource: {source}")


# ---------------------------------------------------------------------------
# Standalone function (mirrors original deepneurobold API)
# ---------------------------------------------------------------------------

def load_bold_and_brain_mask(
    patient_dir: Path,
    *,
    bold_source: BoldSource = "in_t1",
) -> Tuple[Any, np.ndarray, nib.Nifti1Image]:
    """
    Load a 4-D BOLD volume and the corresponding brain mask.

    Parameters
    ----------
    patient_dir : Path
    bold_source : BoldSource
        Which BOLD file to load (default ``"in_t1"``).

    Returns
    -------
    bold_4d : nibabel proxy (lazy)
    brain_flat : np.ndarray, bool, shape (n_voxels,)
    ref_img : nib.Nifti1Image
    """
    patient_dir = Path(patient_dir)
    bdir = _get_branch_dir(patient_dir)
    bold_p = _resolve_bold_path(patient_dir, bold_source)
    brain_p = patient_dir / "PREPROCESSING" / "t1" / "t1_brain_mask.nii.gz"

    env_dir = os.environ.get("DNB_BOLD_BRANCH_DIR", "").strip()
    env_branch = os.environ.get("DNB_BOLD_BRANCH", "").strip().lower()

    if env_dir:
        branch_mode = "DNB_BOLD_BRANCH_DIR"
    elif env_branch in ("nomcst", "mcst"):
        branch_mode = "DNB_BOLD_BRANCH"
    else:
        branch_mode = "legacy_default"

    print("[io] BOLD resolve:")
    print(f"  patient_dir: {patient_dir}")
    print(f"  bold_source: {bold_source}")
    print(f"  branch_mode: {branch_mode}")
    print(f"  DNB_BOLD_BRANCH_DIR: {env_dir if env_dir else '<unset>'}")
    print(f"  DNB_BOLD_BRANCH    : {env_branch if env_branch else '<unset>'}")
    print(f"  resolved_bold_dir  : {bdir}")
    print(f"  resolved_bold_path : {bold_p}")
    print(f"  brain_mask_path    : {brain_p}")

    if not bold_p.exists():
        raise FileNotFoundError(f"Missing BOLD volume: {bold_p} (resolved bold dir: {bdir})")
    if not brain_p.exists():
        raise FileNotFoundError(f"Missing brain mask: {brain_p}")

    bold_img = nib.load(str(bold_p))
    bold_4d = bold_img.dataobj  # lazy proxy

    brain_img = nib.load(str(brain_p))
    brain_3d = (np.asarray(brain_img.dataobj) > 0)

    if bold_img.shape[:3] != brain_3d.shape:
        raise RuntimeError(
            f"Shape mismatch: BOLD {bold_img.shape[:3]} vs brain {brain_3d.shape}"
        )

    return bold_4d, brain_3d.ravel().astype(bool), bold_img


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class BoldLoader(BaseDataLoader):
    """
    Load a 4-D BOLD NIfTI volume and the corresponding brain mask.

    Parameters
    ----------
    patient_dir : Path
        Patient root directory.
    bold_source : str
        Which BOLD variant to load. One of ``BoldSource``.
    config : dict, optional
        Runtime configuration.
    """

    VALID_SOURCES: Tuple[str, ...] = (
        "in_t1",
        "brain_in_t1",
        "brain_in_t1_spatial",
        "brain_in_t1_spatiotemporal",
        "preprocessed",
        "spatiotemporal",
        "st_preprocessed",
    )

    def __init__(
        self,
        patient_dir: Path,
        bold_source: str = "in_t1",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(patient_dir=patient_dir, config=config)
        if bold_source not in self.VALID_SOURCES:
            raise ValueError(f"bold_source must be one of {self.VALID_SOURCES}")
        self.bold_source: BoldSource = bold_source  # type: ignore[assignment]

    def load(self) -> Dict[str, Any]:
        """
        Return the BOLD 4-D image and flattened brain mask.

        Returns
        -------
        dict with keys:
            ``bold_4d``     : nibabel proxy (lazy)
            ``brain_flat``  : np.ndarray, bool, shape (n_voxels,)
            ``ref_img``     : nib.Nifti1Image
        """
        bold_4d, brain_flat, ref_img = load_bold_and_brain_mask(
            self.patient_dir, bold_source=self.bold_source
        )
        return {
            "bold_4d": bold_4d,
            "brain_flat": brain_flat,
            "ref_img": ref_img,
        }
