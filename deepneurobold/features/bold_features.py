"""
features.bold_features
======================
BOLD-derived feature extractors: raw time-series and engineered features.

Representations
---------------
ts_raw            — raw BOLD time series (no normalisation)
ts_normalized     — z-scored time series per voxel
ts_preprocessed   — detrended + temporal Gaussian smoothing + z-score
bold_features     — 10 engineered temporal features per voxel
full_17           — 10 engineered features (extended set placeholder)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy import ndimage, signal, stats

from .base import BaseFeatureExtractor


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------
Representation = Literal[
    "ts_raw",
    "ts_normalized",
    "ts_preprocessed",
    "bold_features",
    "full_17",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _materialize_bold_float32(bold_4d: Any) -> np.ndarray:
    return np.asanyarray(bold_4d, dtype=np.float32)


def _zscore_rows(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)


def _extract_ts_raw(bold_np: np.ndarray, mask_flat_bool: np.ndarray) -> Dict[str, np.ndarray]:
    X, Y, Z, T = bold_np.shape
    V = X * Y * Z
    idx = np.flatnonzero(mask_flat_bool)
    flat = bold_np.reshape(V, T)
    ts = flat[idx, :].astype(np.float32, copy=False)
    return {
        "features": np.ascontiguousarray(ts, dtype=np.float32),
        "names": np.array([f"t{i}" for i in range(T)], dtype=object),
    }


# ---------------------------------------------------------------------------
# 10-feature computation (public — used by runner for feature importance)
# ---------------------------------------------------------------------------

def calculate_10_bold_features(
    ts_matrix: np.ndarray,
    tr: float,
    reference_ts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 10 temporal features per voxel from the BOLD time series.

    Parameters
    ----------
    ts_matrix : np.ndarray, shape (n_voxels, n_time)
    tr : float
        Repetition time in seconds.
    reference_ts : np.ndarray, optional
        Reference time series for correlation (defaults to global mean).

    Returns
    -------
    feats : np.ndarray, shape (n_voxels, 10), float32
    names : np.ndarray of str, shape (10,)
    """
    n_voxels, n_time = ts_matrix.shape
    baseline = np.mean(ts_matrix, axis=1, keepdims=True)
    ts_norm = ts_matrix - baseline

    f_std       = np.std(ts_matrix, axis=1)
    f_dip_min   = np.min(ts_norm, axis=1)
    f_t_min     = np.argmin(ts_norm, axis=1).astype(np.float32)
    f_t_min_sec = f_t_min * float(tr)
    f_auc_neg   = np.sum(np.where(ts_norm < 0, ts_norm, 0), axis=1)
    f_auc_pos   = np.sum(np.where(ts_norm > 0, ts_norm, 0), axis=1)

    ref = reference_ts if reference_ts is not None else np.mean(ts_matrix, axis=0)

    f_corr_tpl  = np.zeros(n_voxels, dtype=np.float32)
    f_slope_down = np.zeros(n_voxels, dtype=np.float32)
    f_slope_up   = np.zeros(n_voxels, dtype=np.float32)
    f_t_rise     = np.zeros(n_voxels, dtype=np.float32)

    for i in range(n_voxels):
        idx_min = int(f_t_min[i])
        val_min = float(f_dip_min[i])
        if idx_min > 0:
            f_slope_down[i] = val_min / (idx_min * float(tr))
        if idx_min < n_time - 1:
            f_slope_up[i] = -val_min / ((n_time - 1 - idx_min) * float(tr))
            post_min = ts_norm[i, idx_min:]
            recovery = np.where(post_min > val_min * 0.5)[0]
            f_t_rise[i] = (
                (recovery[0] * float(tr)) if recovery.size > 0
                else ((n_time - idx_min) * float(tr))
            )
        if f_std[i] > 1e-9:
            r, _ = stats.pearsonr(ts_matrix[i], ref)
            f_corr_tpl[i] = float(r) if np.isfinite(r) else 0.0

    feats = np.column_stack([
        f_std, f_dip_min, f_t_min, f_t_min_sec,
        f_slope_down, f_slope_up, f_auc_neg, f_auc_pos,
        f_t_rise, f_corr_tpl,
    ]).astype(np.float32)
    names = np.array([
        "std", "dip_min", "t_min", "t_min_sec",
        "slope_down", "slope_up", "auc_neg", "auc_pos",
        "t_rise", "corr_tpl",
    ], dtype=object)
    return feats, names


# ---------------------------------------------------------------------------
# Main extraction function (mirrors original deepneurobold API)
# ---------------------------------------------------------------------------

def extract_features(
    bold_4d: Optional[Any] = None,
    mask_flat: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    bold_4d_path: Optional[Union[str, Path]] = None,
    mask_3d_path: Optional[Union[str, Path]] = None,
    tr: Optional[float] = None,
    representation: Union[Representation, str] = "ts_raw",
    patient_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract features from a 4-D BOLD volume.

    Call either with in-memory arrays (``bold_4d``, ``mask_flat``, ``dt``)
    or with file paths (``bold_4d_path``, ``mask_3d_path``, ``tr``).

    Returns
    -------
    dict with keys ``'features'`` (np.ndarray) and ``'names'`` (np.ndarray of str).
    """
    representation = str(representation).lower()
    patient_path = Path(patient_dir) if patient_dir else None

    if bold_4d_path and mask_3d_path:
        bold_img = nib.load(str(bold_4d_path))
        bold_np = _materialize_bold_float32(bold_img.dataobj)
        mask_img = nib.load(str(mask_3d_path))
        mask_flat_bool = (np.asarray(mask_img.dataobj) > 0).ravel()
        actual_tr = float(tr)
    else:
        bold_np = _materialize_bold_float32(bold_4d)
        mask_flat_bool = np.asarray(mask_flat, dtype=bool).ravel()
        actual_tr = float(dt) if dt is not None else 1.8

    # ---- basic ----
    if representation in ("ts_raw", "ts_normalized"):
        out = _extract_ts_raw(bold_np, mask_flat_bool)
        if representation == "ts_normalized":
            out["features"] = _zscore_rows(out["features"])
        return out

    # ---- ts_preprocessed: detrend + temporal smooth + z-score ----
    if representation == "ts_preprocessed":
        out = _extract_ts_raw(bold_np, mask_flat_bool)
        X = signal.detrend(out["features"], axis=1)
        X = ndimage.gaussian_filter1d(X, sigma=1.0, axis=1)
        out["features"] = _zscore_rows(X.astype(np.float32))
        return out

    # ---- derived features ----
    out_ts = _extract_ts_raw(bold_np, mask_flat_bool)
    X_raw = out_ts["features"]
    sigma_val = 0.5 if representation == "bold_features" else 0.0
    X_filt = ndimage.gaussian_filter1d(X_raw, sigma=sigma_val, axis=1) if sigma_val > 0 else X_raw

    # White matter reference (if available)
    wm_ref = None
    if patient_path:
        wm_mask_path = patient_path / "PREPROCESSING" / "t1" / "wm_mask_in_bold.nii.gz"
        if wm_mask_path.exists():
            wm_mask = nib.load(str(wm_mask_path)).get_fdata().ravel() > 0
            wm_voxels = bold_np.reshape(-1, bold_np.shape[-1])[wm_mask & mask_flat_bool]
            if wm_voxels.size > 0:
                wm_ref = np.mean(wm_voxels, axis=0)

    if representation in ("bold_features", "bold_features_10", "bold_features_17", "full_17"):
        X_b, names_b = calculate_10_bold_features(X_filt, actual_tr, reference_ts=wm_ref)
        return {"features": X_b, "names": names_b}

    raise ValueError(
        f"Unknown representation: {representation!r}. "
        f"Available: ts_raw, ts_normalized, ts_preprocessed, bold_features, "
        f"bold_features_10, bold_features_17, full_17"
    )


def extract_features_compat(
    bold_4d: Any,
    mask_flat: np.ndarray,
    dt: float,
    representation: str,
) -> Dict[str, np.ndarray]:
    """Backward-compatible alias for ``extract_features``."""
    return extract_features(bold_4d=bold_4d, mask_flat=mask_flat, dt=dt, representation=representation)


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class BoldFeatureExtractor(BaseFeatureExtractor):
    """
    Extract BOLD features for all representations.

    Parameters
    ----------
    representation : str
    dt : float
    patient_dir : Path, optional
    config : dict, optional
    """

    def extract(self, bold_4d: Any, mask_flat: np.ndarray) -> np.ndarray:
        """
        Extract the BOLD feature matrix for masked voxels.

        Returns
        -------
        np.ndarray, shape (n_masked_voxels, n_features), float32
        """
        result = extract_features(
            bold_4d=bold_4d,
            mask_flat=mask_flat,
            dt=self.dt,
            representation=self.representation,
            patient_dir=self.patient_dir,
        )
        return result["features"]
