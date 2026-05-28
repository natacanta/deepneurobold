"""
experiment.classifier_stats
===========================
Paired statistical tests + effect sizes for classifier comparison
(trial_144_classifier_comparison).

Functions provided
------------------
- ``delong_test(y_true, scores_a, scores_b)``
    Sun & Xu (2014) fast DeLong test for two correlated ROC AUCs.
    Returns (auc_a, auc_b, var_a, var_b, cov_ab, z, p_two_sided).

- ``paired_t_metric(values_a, values_b)``
    Paired t-test on per-fold (or per-patient) metric vectors.

- ``wilcoxon_signed_rank(values_a, values_b)``
    Non-parametric paired comparison (preferred for n=44 patients,
    AUC not normally distributed).

- ``cliffs_delta(values_a, values_b)``
    Non-parametric paired effect size in [-1, 1].

- ``cohen_dz_paired(values_a, values_b)``
    Paired-samples Cohen's d_z (mean diff / SD of diff).

- ``bonferroni_adjust(pvalues, n_tests)``
- ``benjamini_hochberg(pvalues)``

References
----------
- DeLong ER, DeLong DM, Clarke-Pearson DL. Biometrics. 1988.
- Sun X, Xu W. IEEE SPL 2014;21(11):1389-1393. (fast O(n log n) variant)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# DeLong test (Sun & Xu 2014 fast version)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank assignment (average rank for ties). O(n log n)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-indexed mid-rank
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised DeLong for K classifiers and 2-class labels.

    Parameters
    ----------
    predictions_sorted_transposed : np.ndarray, shape (K, N)
        Predictions for each classifier; first ``label_1_count`` columns are
        the positive samples (sorted so labels are [1,1,...,0,0,...]).
    label_1_count : int
        Number of positive samples.

    Returns
    -------
    aucs : np.ndarray, shape (K,)
    delongcov : np.ndarray, shape (K, K) — covariance matrix of AUCs.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative sample.")
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> Dict[str, float]:
    """
    Two-sided DeLong test on correlated AUCs.

    Returns a dict with: ``auc_a``, ``auc_b``, ``auc_diff``,
    ``var_a``, ``var_b``, ``cov_ab``, ``var_diff``, ``z``, ``p_two_sided``,
    plus ``ci95_low``, ``ci95_high`` for the AUC difference.

    NaN-safe: if all scores are identical or one class is empty,
    returns NaNs with a ``note`` key explaining why.
    """
    y = np.asarray(y_true, dtype=np.int32).ravel()
    sa = np.asarray(scores_a, dtype=np.float64).ravel()
    sb = np.asarray(scores_b, dtype=np.float64).ravel()
    if y.size != sa.size or y.size != sb.size:
        raise ValueError("delong_test: input length mismatch")

    # Sort so labels are [1,1,...,0,0,...]
    order = np.argsort(-y)
    y = y[order]
    sa = sa[order]
    sb = sb[order]
    label_1_count = int(np.sum(y == 1))
    if label_1_count == 0 or label_1_count == y.size:
        return _delong_nan("one class is empty")
    preds = np.vstack([sa, sb])

    try:
        aucs, cov = _fast_delong(preds, label_1_count)
    except Exception as e:  # noqa: BLE001
        return _delong_nan(f"delong-failed: {e}")

    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var_a, var_b, cov_ab = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    var_diff = var_a + var_b - 2.0 * cov_ab
    if var_diff <= 0:
        return _delong_nan("var_diff <= 0 (perfectly correlated scores)")
    diff = auc_a - auc_b
    z = diff / float(np.sqrt(var_diff))
    p = 2.0 * float(stats.norm.sf(abs(z)))
    ci_half = 1.959964 * float(np.sqrt(var_diff))
    return {
        "auc_a": auc_a, "auc_b": auc_b, "auc_diff": float(diff),
        "var_a": var_a, "var_b": var_b, "cov_ab": cov_ab,
        "var_diff": float(var_diff), "z": float(z), "p_two_sided": p,
        "ci95_low": float(diff - ci_half), "ci95_high": float(diff + ci_half),
        "n_pos": label_1_count, "n_neg": int(y.size - label_1_count),
    }


def _delong_nan(note: str) -> Dict[str, float]:
    nan = float("nan")
    return {
        "auc_a": nan, "auc_b": nan, "auc_diff": nan,
        "var_a": nan, "var_b": nan, "cov_ab": nan,
        "var_diff": nan, "z": nan, "p_two_sided": nan,
        "ci95_low": nan, "ci95_high": nan, "note": note,
        "n_pos": 0, "n_neg": 0,
    }


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------

def paired_t_metric(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Dict[str, float]:
    """
    Paired t-test on per-fold or per-patient metric vectors.
    NaNs are pairwise-removed.
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 2:
        nan = float("nan")
        return {
            "t": nan, "p_two_sided": nan, "df": float(max(0, n - 1)),
            "mean_diff": nan, "ci95_low": nan, "ci95_high": nan, "n": int(n),
        }
    diff = a - b
    t_stat, p = stats.ttest_rel(a, b)
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1))
    se = sd_d / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return {
        "t": float(t_stat), "p_two_sided": float(p), "df": float(n - 1),
        "mean_diff": mean_d, "sd_diff": sd_d,
        "ci95_low": mean_d - tcrit * se, "ci95_high": mean_d + tcrit * se,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Dict[str, float]:
    """
    Non-parametric paired test. Pairs with zero difference are dropped
    (``zero_method='wilcox'``).
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 2:
        nan = float("nan")
        return {"W": nan, "p_two_sided": nan, "n": int(n), "median_diff": nan}
    try:
        w, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        # All differences zero or other edge case
        return {"W": float("nan"), "p_two_sided": 1.0, "n": int(n),
                "median_diff": float(np.median(a - b)),
                "note": "all-differences-zero-or-degenerate"}
    return {
        "W": float(w), "p_two_sided": float(p), "n": int(n),
        "median_diff": float(np.median(a - b)),
    }


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cliffs_delta(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Dict[str, float]:
    """
    Cliff's delta — non-parametric effect size in [-1, 1].
    Positive means A > B more often than B > A.

    Magnitude thresholds (Romano et al. 2006):
        |δ| < 0.147   negligible
        0.147 - 0.33  small
        0.33  - 0.474 medium
        > 0.474       large
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    mask_a = np.isfinite(a)
    mask_b = np.isfinite(b)
    a, b = a[mask_a], b[mask_b]
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return {"delta": float("nan"), "magnitude": "undefined", "n_a": int(n_a), "n_b": int(n_b)}
    # Sum over all pairs: sign(a_i - b_j)
    # Vectorised pairwise comparison (n_a x n_b broadcast)
    diffs = a[:, None] - b[None, :]
    greater = np.sum(diffs > 0)
    less = np.sum(diffs < 0)
    delta = (greater - less) / (n_a * n_b)
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return {"delta": float(delta), "magnitude": mag, "n_a": int(n_a), "n_b": int(n_b)}


def cohen_dz_paired(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> Dict[str, float]:
    """
    Cohen's d_z for paired samples = mean(a-b) / sd(a-b).

    Magnitude thresholds (Cohen 1988):
        |d_z| < 0.2  negligible
        0.2 - 0.5    small
        0.5 - 0.8    medium
        > 0.8        large
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 2:
        return {"d_z": float("nan"), "magnitude": "undefined", "n": int(n)}
    diff = a - b
    sd_d = float(np.std(diff, ddof=1))
    if sd_d == 0:
        return {"d_z": float("nan"), "magnitude": "zero-variance", "n": int(n)}
    d_z = float(np.mean(diff) / sd_d)
    abs_d = abs(d_z)
    if abs_d < 0.2:
        mag = "negligible"
    elif abs_d < 0.5:
        mag = "small"
    elif abs_d < 0.8:
        mag = "medium"
    else:
        mag = "large"
    return {"d_z": d_z, "magnitude": mag, "n": int(n)}


# ---------------------------------------------------------------------------
# Multiple-testing corrections
# ---------------------------------------------------------------------------

def bonferroni_adjust(pvalues: Sequence[float], n_tests: int) -> List[float]:
    """Bonferroni-adjusted p-values (clipped to [0, 1])."""
    return [float(min(1.0, p * n_tests)) for p in pvalues]


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg (FDR) step-up adjusted p-values."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=np.float64)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        cummin = min(cummin, val)
        adj[i] = cummin
    out = np.empty(n, dtype=np.float64)
    out[order] = adj
    return [float(x) for x in out]
