#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepneurobold.preprocessing
=============================
Per-patient preprocessing pipeline.

This package exposes the canonical :class:`BasePreprocessing` orchestrator
from :mod:`.base`, plus the building blocks used during preprocessing
(DICOM->NIfTI conversion, segmentation->T1 alignment, ANTs sequences->T1
registration, FAST tissue PVEs, large-vessel masking, BOLD smoothing
and FFT-derived maps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib

# Canonical entry point — full pipeline orchestrator.
from deepneurobold.preprocessing.base import BasePreprocessing

# Building blocks (re-exported for convenience and stable downstream imports).
from deepneurobold.preprocessing.dicom2nifti import convert_patient_dicom
from deepneurobold.preprocessing.mask_to_T1 import MaskToT1
from deepneurobold.preprocessing.sequences_to_t1_old import SequencesToT1
from deepneurobold.preprocessing.gm_wm_csf import derive_healthy_gm_wm_csf_pves
from deepneurobold.preprocessing.large_vessels import run_large_vessel_mask
from deepneurobold.preprocessing.smooth_bold import (
    run_spatial_and_temporal_smoothing,
    run_fft_stage_to_niftis_blockwise,
)
from deepneurobold.preprocessing.fslenv import run_in_fsl

try:
    from deepneurobold.preprocessing.ensure_gz import ensure_gz  # type: ignore
except Exception:  # noqa: BLE001
    ensure_gz = None  # optional helper, not always present


__all__ = [
    "BasePreprocessing",
    "convert_patient_dicom",
    "MaskToT1",
    "SequencesToT1",
    "derive_healthy_gm_wm_csf_pves",
    "run_large_vessel_mask",
    "run_spatial_and_temporal_smoothing",
    "run_fft_stage_to_niftis_blockwise",
    "run_in_fsl",
    "ensure_gz",
    "ensure_t1_reoriented",
]


def ensure_t1_reoriented(patient_dir: Path) -> Optional[str]:
    """Ensure ``PREPROCESSING/t1/t1_reoriented.nii.gz`` exists for *patient_dir*.

    If the canonical RAS-reoriented T1 is already present it is returned
    unchanged. Otherwise the function reads ``PREPROCESSING/t1/t1.nii.gz``,
    applies :func:`nibabel.as_closest_canonical`, and writes the result.

    Returns the absolute path as a string, or ``None`` if the source file
    is missing.
    """
    t1_dir = Path(patient_dir) / "PREPROCESSING" / "t1"
    src = t1_dir / "t1.nii.gz"
    dst = t1_dir / "t1_reoriented.nii.gz"
    if dst.exists():
        return str(dst)
    if src.exists():
        print(f"[{Path(patient_dir).name}] Reorienting T1 -> t1_reoriented.nii.gz", flush=True)
        img = nib.load(str(src))
        nib.save(nib.as_closest_canonical(img), str(dst))
        return str(dst)
    return None
