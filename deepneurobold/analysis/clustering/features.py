"""
analysis.clustering.features
=============================
Temporal feature extraction for BOLD-fMRI signals.

Ported from ``deepneurobold/analysis/clustering/features.py``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# _trapz removed in NumPy 2.0; np.trapezoid added in 1.26
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

__all__ = ["extract_bold_features", "extract_bold_timeseries_features"]


def _normalize_template(st: np.ndarray) -> np.ndarray:
    st = np.asarray(st, dtype=np.float32)
    st = st - np.nanmean(st)
    return st / (np.nanstd(st) + 1e-6)


def _rowwise_zscore(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd


def extract_bold_features(
    bold_4d: np.ndarray,
    mask_flat: np.ndarray,
    stim_template: Optional[np.ndarray] = None,
    *,
    percentile_baseline: float = 20.0,
    dt: float = 1.8,
    chunk_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute per-voxel temporal features for voxels where mask_flat is True.

    Features (10 total)
    -------------------
    dip_min, t_min, slope_down, slope_up, auc_neg, auc_pos,
    t_rise, std, corr_tpl, t_min_sec

    Parameters
    ----------
    bold_4d : np.ndarray
        4-D [X, Y, Z, T] or 2-D [V, T] / [T, V].
    mask_flat : np.ndarray, bool
        Flat boolean mask of length V.
    stim_template : np.ndarray, optional
        1-D stimulus template (length T).
    percentile_baseline : float
        Percentile used to estimate the baseline S0.
    dt : float
        TR in seconds.
    chunk_size : int, optional
        Voxel chunk size for memory-efficient processing.

    Returns
    -------
    dict
        ``features``: (N, 10) float32 array.
        ``names``: object array of feature name strings.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    pb = float(percentile_baseline)
    if not (0.0 <= pb <= 100.0):
        raise ValueError("percentile_baseline must be between 0 and 100")

    mask_flat = np.asarray(mask_flat).ravel()
    if mask_flat.dtype != bool:
        raise ValueError("mask_flat must be boolean")

    # Normalise to [V, T]
    if bold_4d.ndim == 4:
        T = int(bold_4d.shape[-1])
        V = int(np.prod(bold_4d.shape[:-1]))
        if mask_flat.shape[0] != V:
            raise ValueError(f"Mask length {mask_flat.shape[0]} != V {V}")
        X_all = bold_4d.reshape(V, T)
    elif bold_4d.ndim == 2:
        r, c = map(int, bold_4d.shape)
        Vm = int(mask_flat.shape[0])
        if r == Vm:
            X_all = np.asarray(bold_4d, dtype=np.float32, order="C")
        elif c == Vm:
            X_all = np.asarray(bold_4d.T, dtype=np.float32, order="C")
        else:
            raise ValueError(f"BOLD 2D shape {bold_4d.shape} incompatible with mask length {Vm}")
    else:
        raise ValueError("bold_4d must be [X,Y,Z,T] or [V,T] (or [T,V])")

    names = np.array(
        ["dip_min", "t_min", "slope_down", "slope_up",
         "auc_neg", "auc_pos", "t_rise", "std", "corr_tpl", "t_min_sec"],
        dtype=object,
    )
    idx = np.flatnonzero(mask_flat)
    N = idx.size

    if N == 0:
        return {"features": np.zeros((0, len(names)), np.float32), "names": names}

    st = None
    if stim_template is not None:
        stim_template = np.asarray(stim_template, dtype=np.float32)
        if stim_template.ndim != 1 or stim_template.shape[0] != X_all.shape[-1]:
            raise ValueError("stim_template length must match T")
        st = _normalize_template(stim_template)

    feats = np.empty((N, len(names)), dtype=np.float32)
    w = max(1, X_all.shape[-1] // 20)
    cs = N if (chunk_size is None or chunk_size <= 0) else chunk_size
    X_all = np.asarray(X_all, dtype=np.float32, order="C")

    cursor = 0
    while cursor < N:
        end = min(cursor + cs, N)
        idc = idx[cursor:end]
        X = X_all[idc, :]

        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        S0 = np.nanpercentile(X, pb, axis=1).astype(np.float32) + 1e-6
        Xn = (X - S0[:, None]) / S0[:, None]

        minv   = Xn.min(axis=1).astype(np.float32)
        argmin = Xn.argmin(axis=1).astype(np.int32)

        auc_neg = _trapz(np.minimum(Xn, 0.0), dx=float(dt), axis=1).astype(np.float32)
        auc_pos = _trapz(np.maximum(Xn, 0.0), dx=float(dt), axis=1).astype(np.float32)

        left  = np.clip(argmin - w, 0, Xn.shape[1] - 1)
        right = np.clip(argmin + w, 0, Xn.shape[1] - 1)
        rows  = np.arange(Xn.shape[0])

        slope_down = (Xn[rows, argmin] - Xn[rows, left])  / (argmin - left  + 1e-6)
        slope_up   = (Xn[rows, right]  - Xn[rows, argmin]) / (right  - argmin + 1e-6)
        std_val    = Xn.std(axis=1).astype(np.float32)

        t_rise = np.zeros(Xn.shape[0], dtype=np.float32)
        for i in range(Xn.shape[0]):
            seg = Xn[i, argmin[i]:]
            back = np.where(seg >= 0)[0]
            t_rise[i] = back[0] * dt if back.size > 0 else np.nan

        corr = (
            (_rowwise_zscore(Xn) @ st.astype(np.float32)) / float(Xn.shape[1])
            if st is not None
            else np.zeros(Xn.shape[0], dtype=np.float32)
        )

        block = np.stack(
            [minv, argmin.astype(np.float32), slope_down.astype(np.float32),
             slope_up.astype(np.float32), auc_neg, auc_pos, t_rise, std_val,
             corr.astype(np.float32), argmin.astype(np.float32) * float(dt)],
            axis=1,
        )
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        feats[cursor:end, :] = block
        cursor = end

    return {"features": np.ascontiguousarray(feats, dtype=np.float32), "names": names}


def extract_bold_timeseries_features(
    bold_4d: np.ndarray,
    mask_flat: np.ndarray,
    dt: float = 1.8,
) -> Dict[str, np.ndarray]:
    """
    Extract row-wise z-scored BOLD time series as features.

    Returns one feature per time point (T features total).
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    mask_flat = np.asarray(mask_flat).ravel()
    if mask_flat.dtype != bool:
        raise ValueError("mask_flat must be boolean")

    if bold_4d.ndim == 4:
        T = int(bold_4d.shape[-1])
        V = int(np.prod(bold_4d.shape[:-1]))
        if mask_flat.shape[0] != V:
            raise ValueError(f"Mask length {mask_flat.shape[0]} != V {V}")
        X_all = bold_4d.reshape(V, T)
    elif bold_4d.ndim == 2:
        r, c = map(int, bold_4d.shape)
        Vm = int(mask_flat.shape[0])
        if r == Vm:
            X_all = np.asarray(bold_4d, dtype=np.float32, order="C")
        elif c == Vm:
            X_all = np.asarray(bold_4d.T, dtype=np.float32, order="C")
        else:
            raise ValueError(f"BOLD 2D shape {bold_4d.shape} incompatible with mask length {Vm}")
    else:
        raise ValueError("bold_4d must be [X,Y,Z,T] or [V,T] (or [T,V])")

    idx = np.flatnonzero(mask_flat)
    feat_names = np.array([f"t{i}" for i in range(X_all.shape[-1])], dtype=object)

    if idx.size == 0:
        return {"features": np.zeros((0, X_all.shape[-1]), dtype=np.float32), "names": feat_names}

    ts = np.asarray(X_all, dtype=np.float32)[idx, :]
    mean = ts.mean(axis=1, keepdims=True)
    std  = ts.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    ts_norm = (ts - mean) / std

    return {"features": np.ascontiguousarray(ts_norm, dtype=np.float32), "names": feat_names}
