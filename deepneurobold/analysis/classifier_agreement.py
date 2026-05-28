"""
analysis.classifier_agreement
=============================
Inter-classifier agreement, calibration, and decision-curve analysis utilities
for ``trial_144_classifier_comparison``.

All functions are pure and side-effect free: they return JSON-serialisable
dicts that the comparison runner writes to disk.

Functions
---------
- ``voxel_agreement(prob_a, prob_b, region_mask)``
    Spearman ρ and Pearson r on probability maps within ``region_mask``.

- ``hotspot_dice(prob_a, prob_b, region_mask, percentiles=[90, 95])``
    Dice coefficient between binary hotspot masks at adaptive percentile
    thresholds computed within ``region_mask``.

- ``calibration_curve_data(y_true, y_score, n_bins=10, strategy='quantile')``
    Reliability-diagram data (mean predicted vs empirical fraction).

- ``decision_curve_data(y_true, y_score, thresholds=None)``
    Decision Curve Analysis (Vickers & Elkin 2006) — net benefit vs threshold.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Voxel-wise agreement on whole-brain probability maps
# ---------------------------------------------------------------------------

def voxel_agreement(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    region_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Spearman ρ and Pearson r between two probability maps within
    ``region_mask`` (boolean, same flat shape as the prob maps).

    Returns a dict with: ``spearman_rho``, ``spearman_p``, ``pearson_r``,
    ``pearson_p``, ``n_voxels``, ``mean_abs_diff``, ``rms_diff``.
    """
    pa = np.asarray(prob_a, dtype=np.float64).ravel()
    pb = np.asarray(prob_b, dtype=np.float64).ravel()
    m = np.asarray(region_mask, dtype=bool).ravel()
    if pa.shape != pb.shape or pa.shape != m.shape:
        raise ValueError("voxel_agreement: prob/mask shape mismatch")

    a, b = pa[m], pb[m]
    n = a.size
    nan = float("nan")
    if n < 3:
        return {
            "spearman_rho": nan, "spearman_p": nan,
            "pearson_r": nan, "pearson_p": nan,
            "n_voxels": int(n), "mean_abs_diff": nan, "rms_diff": nan,
        }
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {
            "spearman_rho": nan, "spearman_p": nan,
            "pearson_r": nan, "pearson_p": nan,
            "n_voxels": int(n),
            "mean_abs_diff": float(np.mean(np.abs(a - b))),
            "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
            "note": "zero-variance",
        }

    rho, p_rho = spearmanr(a, b)
    r, p_r = pearsonr(a, b)
    return {
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "pearson_r": float(r), "pearson_p": float(p_r),
        "n_voxels": int(n),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
    }


# ---------------------------------------------------------------------------
# Hotspot Dice
# ---------------------------------------------------------------------------

def hotspot_dice(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    region_mask: np.ndarray,
    percentiles: Sequence[float] = (90.0, 95.0),
) -> Dict[str, Dict[str, float]]:
    """
    Dice coefficient between hotspot masks at adaptive percentile thresholds.

    For each percentile p in ``percentiles``:
      - Compute threshold_a = percentile(prob_a[region_mask], p)
      - Compute threshold_b = percentile(prob_b[region_mask], p)
      - Binarise each map at its own threshold within region_mask.
      - Dice = 2|A∩B| / (|A|+|B|)

    Returns a dict keyed by percentile (as string) → {dice, threshold_a,
    threshold_b, n_a, n_b}.
    """
    pa = np.asarray(prob_a, dtype=np.float64).ravel()
    pb = np.asarray(prob_b, dtype=np.float64).ravel()
    m = np.asarray(region_mask, dtype=bool).ravel()
    out: Dict[str, Dict[str, float]] = {}
    if not np.any(m):
        for p in percentiles:
            out[f"p{int(p)}"] = {
                "dice": float("nan"), "threshold_a": float("nan"),
                "threshold_b": float("nan"), "n_a": 0, "n_b": 0,
                "note": "empty-region",
            }
        return out

    a_in, b_in = pa[m], pb[m]
    for p in percentiles:
        ta = float(np.percentile(a_in, p))
        tb = float(np.percentile(b_in, p))
        mask_a = m & (pa >= ta)
        mask_b = m & (pb >= tb)
        n_a = int(np.sum(mask_a))
        n_b = int(np.sum(mask_b))
        denom = n_a + n_b
        if denom == 0:
            dice = float("nan")
        else:
            inter = int(np.sum(mask_a & mask_b))
            dice = float(2 * inter / denom)
        out[f"p{int(p)}"] = {
            "dice": dice, "threshold_a": ta, "threshold_b": tb,
            "n_a": n_a, "n_b": n_b,
        }
    return out


# ---------------------------------------------------------------------------
# Calibration (reliability diagram)
# ---------------------------------------------------------------------------

def calibration_curve_data(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Dict[str, List[float]]:
    """
    Bin predicted probabilities and compute empirical positive fraction
    per bin. Output suitable for a reliability-diagram plot.

    Parameters
    ----------
    n_bins : int
    strategy : "quantile" or "uniform"

    Returns
    -------
    dict with: ``bin_centers`` (mean predicted prob per bin),
    ``bin_fractions`` (empirical fraction positive per bin),
    ``bin_counts`` (n samples per bin),
    ``ece`` (expected calibration error, weighted L1),
    ``mce`` (max calibration error).
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    n = y.size
    if n == 0:
        return {"bin_centers": [], "bin_fractions": [], "bin_counts": [],
                "ece": float("nan"), "mce": float("nan"), "n": 0}

    if strategy == "quantile":
        # Equal-frequency bins; use np.quantile for edge robustness
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(s, quantiles))
        # ensure strictly increasing
        if edges.size < 2:
            edges = np.array([s.min(), s.max() + 1e-9])
        bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, edges.size - 2)
    else:  # uniform
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_idx = np.clip(np.digitize(s, edges[1:-1]), 0, n_bins - 1)

    n_used_bins = edges.size - 1 if strategy == "quantile" else n_bins
    centers: List[float] = []
    fracs: List[float] = []
    counts: List[int] = []
    abs_errs: List[float] = []
    for b in range(n_used_bins):
        in_bin = (bin_idx == b)
        cnt = int(np.sum(in_bin))
        if cnt == 0:
            continue
        mean_pred = float(np.mean(s[in_bin]))
        emp_pos = float(np.mean(y[in_bin]))
        centers.append(mean_pred)
        fracs.append(emp_pos)
        counts.append(cnt)
        abs_errs.append(abs(mean_pred - emp_pos))

    if len(counts) == 0:
        return {"bin_centers": [], "bin_fractions": [], "bin_counts": [],
                "ece": float("nan"), "mce": float("nan"), "n": int(n)}

    weights = np.array(counts, dtype=np.float64) / float(n)
    ece = float(np.sum(weights * np.array(abs_errs)))
    mce = float(max(abs_errs))
    return {
        "bin_centers": [float(x) for x in centers],
        "bin_fractions": [float(x) for x in fracs],
        "bin_counts": [int(x) for x in counts],
        "ece": ece, "mce": mce, "n": int(n),
        "strategy": strategy, "n_bins_requested": int(n_bins),
    }


# ---------------------------------------------------------------------------
# Decision Curve Analysis (Vickers & Elkin 2006)
# ---------------------------------------------------------------------------

def decision_curve_data(
    y_true: Sequence[int],
    y_score: Sequence[float],
    thresholds: Sequence[float] = None,
) -> Dict[str, List[float]]:
    """
    Compute net benefit at each threshold p_t:
        NB(p_t) = (TP/n) - (FP/n) * (p_t / (1 - p_t))

    Reference: Vickers AJ, Elkin EB. Med Decis Making. 2006;26(6):565-74.

    Returns
    -------
    dict with: ``thresholds``, ``net_benefit``, ``net_benefit_treat_all``,
    ``net_benefit_treat_none`` (always 0), ``prevalence``.
    """
    y = np.asarray(y_true, dtype=np.int32).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    n = y.size
    if n == 0:
        return {"thresholds": [], "net_benefit": [],
                "net_benefit_treat_all": [], "net_benefit_treat_none": [],
                "prevalence": float("nan"), "n": 0}

    if thresholds is None:
        thresholds = list(np.arange(0.01, 0.99, 0.01))
    thrs = np.asarray(thresholds, dtype=np.float64)
    prevalence = float(np.mean(y))

    nb_model: List[float] = []
    nb_treat_all: List[float] = []
    for pt in thrs:
        pred_pos = s >= pt
        tp = float(np.sum(pred_pos & (y == 1)))
        fp = float(np.sum(pred_pos & (y == 0)))
        if pt >= 1.0:
            nb_model.append(0.0)
            nb_treat_all.append(0.0)
            continue
        weight = pt / (1.0 - pt)
        nb_model.append(tp / n - fp / n * weight)
        # Treat all → every sample is predicted positive → TP=#pos, FP=#neg
        nb_treat_all.append(prevalence - (1.0 - prevalence) * weight)

    return {
        "thresholds": [float(x) for x in thrs],
        "net_benefit": nb_model,
        "net_benefit_treat_all": nb_treat_all,
        "net_benefit_treat_none": [0.0] * len(thrs),
        "prevalence": prevalence,
        "n": int(n),
    }
