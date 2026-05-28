"""
analysis.bold_quality
=====================
Per-patient quality features for resting-state BOLD acquisitions.

Features computed (all over voxels inside the **healthy** brain mask, so the
tumor itself does not bias the global metrics):

  * ``tsnr_mean``                — temporal SNR averaged over the healthy mask
  * ``tsnr_median``              — robust counterpart
  * ``framewise_displacement``   — mean FD across volumes (Power 2012)
  * ``fd_max``                   — worst FD value in the series
  * ``fd_pct_above_0_5mm``       — % of volumes with FD > 0.5 mm (clinical
                                    threshold often used for fMRI exclusion)
  * ``dvars_mean``, ``dvars_max``
  * ``n_spikes_z3``              — # frames with DVARS z-score > 3
  * ``drift_pct_per_min``        — linear drift in the global signal,
                                    expressed as % of baseline per minute

The BOLD series can be either the raw or the smoothed branch — the function
just expects a 4D NIfTI in T1 space. A healthy-tissue mask is mandatory to
keep the metrics tumor-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np


@dataclass
class BoldQualityResult:
    tsnr_mean: float
    tsnr_median: float
    framewise_displacement_mean_mm: float
    framewise_displacement_max_mm: float
    fd_pct_above_0_5mm: float
    dvars_mean: float
    dvars_max: float
    n_spikes_z3: int
    drift_pct_per_min: float
    n_volumes: int
    n_voxels_healthy: int
    tr_sec: float

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bold_quality_features(
    bold_4d: Optional[np.ndarray] = None,
    healthy_mask_3d: Optional[np.ndarray] = None,
    *,
    healthy_ts: Optional[np.ndarray] = None,
    centroid_ts_mm: Optional[np.ndarray] = None,
    tr_sec: float = 1.8,
    voxel_size_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    motion_params: Optional[np.ndarray] = None,
    head_radius_mm: float = 50.0,
) -> BoldQualityResult:
    """Compute BOLD QA features.

    Two supported input modes:
      (A) ``bold_4d`` + ``healthy_mask_3d`` — convenient for small volumes
      (B) ``healthy_ts`` (shape ``(Nh, T)``) + optionally ``centroid_ts_mm``
          (shape ``(T, 3)`` in mm) — memory-safe path for big clinical BOLDs.
          When ``motion_params`` is None and ``centroid_ts_mm`` is None,
          framewise displacement comes back as NaN.

    Parameters
    ----------
    bold_4d : np.ndarray, optional
        4D BOLD time series (X, Y, Z, T). Ignored when ``healthy_ts`` provided.
    healthy_mask_3d : np.ndarray, optional
        Boolean / integer 3D mask of healthy brain voxels. Required with
        ``bold_4d``.
    healthy_ts : np.ndarray, optional
        Pre-extracted (Nh, T) z-able array of BOLD time-series at the healthy
        voxels (memory-safe path).
    centroid_ts_mm : np.ndarray, optional
        (T, 3) intensity centroid in mm (for FD surrogate when no motion
        params are available).
    tr_sec : float
        Repetition time in seconds, used to convert drift to %/min.
    voxel_size_mm : tuple(float, float, float)
        Voxel size in mm.
    motion_params : np.ndarray, optional
        ``(T, 6)`` motion parameter time series (3 translations in mm followed
        by 3 rotations in radians), as produced by MCFLIRT.
    head_radius_mm : float
        Radius used to convert angular rotations to surface mm displacement.

    Returns
    -------
    BoldQualityResult
    """
    if healthy_ts is not None:
        bold_h = np.asarray(healthy_ts, dtype=np.float64)
        if bold_h.ndim != 2:
            raise ValueError(f"healthy_ts must be (Nh, T), got {bold_h.shape}")
    elif bold_4d is not None:
        bold_4d = np.asarray(bold_4d)
        if bold_4d.ndim != 4:
            raise ValueError(f"bold_4d must be 4D, got shape {bold_4d.shape}")
        if healthy_mask_3d is None:
            raise ValueError("healthy_mask_3d required with bold_4d")
        healthy = np.asarray(healthy_mask_3d).astype(bool)
        if healthy.shape != bold_4d.shape[:3]:
            raise ValueError(
                f"Mask shape {healthy.shape} != BOLD spatial shape {bold_4d.shape[:3]}"
            )
        bold_t = bold_4d.reshape(-1, bold_4d.shape[-1])
        bold_h = bold_t[healthy.ravel()].astype(np.float64)
    else:
        raise ValueError("provide either bold_4d+healthy_mask_3d or healthy_ts")
    n_vox_h, n_t = bold_h.shape
    if n_vox_h == 0:
        raise ValueError("Healthy mask is empty; cannot compute BOLD QA features.")

    # ----- tSNR per voxel -----
    mean_voxel = bold_h.mean(axis=1)
    std_voxel  = bold_h.std(axis=1, ddof=1)
    nz = std_voxel > 1e-8
    tsnr_voxel = np.zeros(n_vox_h, dtype=np.float64)
    tsnr_voxel[nz] = mean_voxel[nz] / std_voxel[nz]
    tsnr_mean   = float(np.mean(tsnr_voxel[nz])) if nz.any() else float("nan")
    tsnr_median = float(np.median(tsnr_voxel[nz])) if nz.any() else float("nan")

    # ----- Framewise displacement -----
    if motion_params is not None and motion_params.shape == (n_t, 6):
        fd = _fd_from_motion_params(motion_params, head_radius_mm=head_radius_mm)
    elif centroid_ts_mm is not None:
        diff = np.linalg.norm(np.diff(np.asarray(centroid_ts_mm), axis=0), axis=1)
        fd = diff
    elif bold_4d is not None:
        fd = _fd_from_centroid(bold_4d, voxel_size_mm)
    else:
        fd = np.array([], dtype=np.float64)
    fd_mean = float(np.mean(fd)) if fd.size else float("nan")
    fd_max  = float(np.max(fd))  if fd.size else float("nan")
    fd_pct  = float(100.0 * np.mean(fd > 0.5)) if fd.size else float("nan")

    # ----- DVARS -----
    diff = np.diff(bold_h, axis=1)
    dvars = np.sqrt((diff ** 2).mean(axis=0))           # (T-1,)
    dvars_mean = float(np.mean(dvars)) if dvars.size else float("nan")
    dvars_max  = float(np.max(dvars))  if dvars.size else float("nan")
    if dvars.size > 0 and np.std(dvars) > 0:
        z = (dvars - np.mean(dvars)) / np.std(dvars)
        n_spikes = int(np.sum(z > 3.0))
    else:
        n_spikes = 0

    # ----- Linear drift in global signal -----
    global_signal = bold_h.mean(axis=0)                  # (T,)
    t = np.arange(n_t) * float(tr_sec)
    if n_t >= 3 and global_signal.std() > 0:
        slope, intercept = np.polyfit(t, global_signal, 1)
        baseline = float(intercept) if intercept != 0 else float(global_signal.mean())
        drift_pct_per_sec = 100.0 * slope / baseline
        drift_pct_per_min = float(drift_pct_per_sec * 60.0)
    else:
        drift_pct_per_min = float("nan")

    return BoldQualityResult(
        tsnr_mean=round(tsnr_mean, 3),
        tsnr_median=round(tsnr_median, 3),
        framewise_displacement_mean_mm=round(fd_mean, 4),
        framewise_displacement_max_mm=round(fd_max, 4),
        fd_pct_above_0_5mm=round(fd_pct, 2),
        dvars_mean=round(dvars_mean, 4),
        dvars_max=round(dvars_max, 4),
        n_spikes_z3=int(n_spikes),
        drift_pct_per_min=round(drift_pct_per_min, 4),
        n_volumes=int(n_t),
        n_voxels_healthy=int(n_vox_h),
        tr_sec=float(tr_sec),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fd_from_motion_params(
    motion_params: np.ndarray,
    head_radius_mm: float,
) -> np.ndarray:
    """Power 2012 framewise displacement from MCFLIRT-style motion params.

    motion_params : (T, 6)  with cols [tx, ty, tz, rx, ry, rz] (rot in rad).
    Returns FD in mm, shape (T-1,).
    """
    mp = np.asarray(motion_params, dtype=np.float64)
    rot_mm = mp[:, 3:] * float(head_radius_mm)           # arc length
    full = np.concatenate([mp[:, :3], rot_mm], axis=1)
    diff = np.abs(np.diff(full, axis=0))
    return diff.sum(axis=1)


def _fd_from_centroid(
    bold_4d: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
) -> np.ndarray:
    """Surrogate FD when no motion params are available: track the centroid
    of intensity across volumes."""
    bold = bold_4d
    n_t = bold.shape[-1]
    coords = np.indices(bold.shape[:3], dtype=np.float64)
    com = np.zeros((n_t, 3), dtype=np.float64)
    for t in range(n_t):
        w = bold[..., t].astype(np.float64)
        total = w.sum()
        if total <= 0:
            com[t] = com[t - 1] if t > 0 else 0.0
            continue
        for ax in range(3):
            com[t, ax] = (coords[ax] * w).sum() / total
    com_mm = com * np.asarray(voxel_size_mm)
    diff = np.linalg.norm(np.diff(com_mm, axis=0), axis=1)
    return diff


# ---------------------------------------------------------------------------
# Convenience loader for the canonical project layout
# ---------------------------------------------------------------------------

def bold_quality_for_patient(
    patient_dir: Path,
    *,
    branch: str = "nomcst",
    bold_filename: str = "BOLD_brain_inT1.nii.gz",
    healthy_mask_name: str = "healthy_mask.nii.gz",
    tr_sec: float = 1.8,
) -> Optional[BoldQualityResult]:
    """Compute BOLD QA for one patient using the canonical data layout.

    Memory-safe: reads BOLD one frame at a time via ``nib.dataobj``, never
    materialising the full 4D array (clinical BOLDs run ~12 GB at float32).

    Returns ``None`` if any required file is missing.
    """
    patient_dir = Path(patient_dir)
    bold_path = patient_dir / "PREPROCESSING" / "bold" / f"branch_{branch}" / bold_filename
    mask_path = patient_dir / "PREPROCESSING" / "mask" / healthy_mask_name
    if not bold_path.exists() or not mask_path.exists():
        return None

    bold_img = nib.load(str(bold_path))
    mask_3d = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
    if bold_img.shape[:3] != mask_3d.shape:
        raise ValueError(
            f"BOLD spatial shape {bold_img.shape[:3]} != mask shape {mask_3d.shape}"
        )

    aff = bold_img.affine
    vox_mm = (
        float(np.linalg.norm(aff[:3, 0])),
        float(np.linalg.norm(aff[:3, 1])),
        float(np.linalg.norm(aff[:3, 2])),
    )
    n_t = int(bold_img.shape[3])
    healthy_coords = np.argwhere(mask_3d)
    n_h = healthy_coords.shape[0]
    if n_h == 0:
        return None

    # Memory-safe extraction: read one frame at a time, never materialise the
    # full 4D. We deliberately SKIP the intensity-centroid (FD surrogate)
    # because the BOLD here lives in the ``nomcst`` branch (no motion
    # correction) and a proper FD requires MCFLIRT motion params instead.
    healthy_ts = np.empty((n_h, n_t), dtype=np.float32)
    xi, yi, zi = healthy_coords[:, 0], healthy_coords[:, 1], healthy_coords[:, 2]
    dataobj = bold_img.dataobj
    for t in range(n_t):
        frame = np.asarray(dataobj[:, :, :, t], dtype=np.float32)
        healthy_ts[:, t] = frame[xi, yi, zi]

    return bold_quality_features(
        healthy_ts=healthy_ts,
        tr_sec=tr_sec, voxel_size_mm=vox_mm,
    )
