"""
analysis.bold_quality_classifier
================================
Per-voxel BOLD-quality classifier.

Design goal: learn a data-driven classifier that automatically labels
BOLD acquisitions as good or bad using image-derived features, and
cross-validate it in a leave-one-patient-out setting.

Implementation choices
----------------------
1. **Per-voxel features** (cheap to compute, agnostic to absolute intensity):
       * temporal SNR (mean/std)
       * detrended-DVARS-RMS (frame-to-frame change magnitude)
       * spike count (frames where |z| > 3 inside this voxel's own ts)
       * low-frequency power ratio (DC-removed, < 0.10 Hz vs total)
       * linear drift slope as % of baseline per minute
       * mean amplitude (mean - global brain mean, after normalisation)
       * variance after detrending

2. **Auto-labels** (no manual labelling required). A voxel is labelled:
       * BAD  if BOTH tSNR is in the bottom decile AND
                       (DVARS-RMS is in the top decile  OR  >= 3 spikes  OR
                        |drift_pct_per_min| > 5)
       * GOOD if tSNR is in the top tercile AND DVARS-RMS is in the bottom
              tercile AND zero spikes AND |drift_pct_per_min| < 1
       * unlabeled otherwise (ignored at training time).
   These thresholds are POPULATION-relative (computed across the union of
   the cohort's healthy-brain voxels) so cross-patient calibration is
   automatic. They are exposed as parameters so they can be tuned to match
   any future manually-annotated ground-truth.

3. **Cross-validation**: leave-one-patient-out. The classifier is a Random
   Forest with class_weight='balanced'. Output is a per-voxel quality
   probability map (1 = good, 0 = bad) for each held-out patient.

Public API
----------
    extract_voxel_features(...)
    auto_label_voxels(...)
    train_leave_one_out(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "tsnr",
    "dvars_rms",
    "n_spikes_z3",
    "low_freq_power_ratio",
    "drift_pct_per_min",
    "mean_amplitude_norm",
    "variance_detrended",
)


def extract_voxel_features(
    ts_matrix: np.ndarray,
    tr_sec: float = 1.8,
    low_freq_cut_hz: float = 0.10,
) -> np.ndarray:
    """Compute per-voxel BOLD QA features.

    Parameters
    ----------
    ts_matrix : np.ndarray of shape (N_voxels, T)
        Each row is one voxel's BOLD time-series.
    tr_sec : float
    low_freq_cut_hz : float
        Upper cutoff for the "low-frequency" band used in the power ratio.

    Returns
    -------
    features : np.ndarray of shape (N_voxels, len(FEATURE_NAMES))
        Column order is ``FEATURE_NAMES``.
    """
    ts = np.asarray(ts_matrix, dtype=np.float64)
    n_vox, n_t = ts.shape
    if n_vox == 0 or n_t < 4:
        return np.zeros((n_vox, len(FEATURE_NAMES)), dtype=np.float64)

    mu = ts.mean(axis=1)
    sd = ts.std(axis=1, ddof=0)
    safe_sd = np.where(sd > 1e-9, sd, 1.0)
    tsnr = mu / safe_sd
    tsnr[sd <= 1e-9] = 0.0

    diff = np.diff(ts, axis=1)
    dvars_rms = np.sqrt(np.mean(diff ** 2, axis=1))

    z = (ts - mu[:, None]) / safe_sd[:, None]
    n_spikes = (np.abs(z) > 3.0).sum(axis=1)

    # Linear drift in % per min (slope / baseline * 100 * 60)
    t = np.arange(n_t) * float(tr_sec)
    t_centered = t - t.mean()
    denom = float((t_centered ** 2).sum())
    slope = ((ts - mu[:, None]) * t_centered).sum(axis=1) / max(denom, 1e-9)
    baseline = mu.copy()
    baseline[np.abs(baseline) < 1e-6] = 1.0
    drift_pct_per_min = slope / baseline * 100.0 * 60.0

    # FFT-based low-freq power ratio (DC removed)
    ts_centered = ts - mu[:, None]
    fft = np.fft.rfft(ts_centered, axis=1)
    power = (fft.real ** 2 + fft.imag ** 2)
    freqs = np.fft.rfftfreq(n_t, d=float(tr_sec))
    low_mask = (freqs > 0) & (freqs <= float(low_freq_cut_hz))
    if low_mask.any():
        total_power = power.sum(axis=1) + 1e-12
        low_power = power[:, low_mask].sum(axis=1)
        low_freq_power_ratio = low_power / total_power
    else:
        low_freq_power_ratio = np.zeros(n_vox)

    global_mean = float(mu.mean())
    mean_amplitude_norm = (mu - global_mean) / max(abs(global_mean), 1e-6)

    # Detrended variance (residual after removing linear trend)
    residual = ts_centered - slope[:, None] * t_centered[None, :]
    variance_detrended = (residual ** 2).mean(axis=1)

    features = np.column_stack([
        tsnr,
        dvars_rms,
        n_spikes.astype(np.float64),
        low_freq_power_ratio,
        drift_pct_per_min,
        mean_amplitude_norm,
        variance_detrended,
    ])
    return features


# ---------------------------------------------------------------------------
# Auto-labelling
# ---------------------------------------------------------------------------

@dataclass
class AutoLabelThresholds:
    tsnr_bad_pct: float = 10.0       # bottom decile of tSNR -> candidate bad
    tsnr_good_pct: float = 66.0      # top tercile of tSNR -> candidate good
    dvars_bad_pct: float = 90.0      # top decile of dvars -> candidate bad
    dvars_good_pct: float = 33.0     # bottom tercile -> candidate good
    spikes_bad_threshold: int = 3
    drift_bad_threshold_abs: float = 5.0
    drift_good_threshold_abs: float = 1.0

    def to_dict(self):
        return {k: getattr(self, k) for k in (
            "tsnr_bad_pct", "tsnr_good_pct",
            "dvars_bad_pct", "dvars_good_pct",
            "spikes_bad_threshold",
            "drift_bad_threshold_abs", "drift_good_threshold_abs",
        )}


def auto_label_voxels(
    features: np.ndarray,
    thresholds: AutoLabelThresholds = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Auto-derive good/bad labels from population percentiles of the features.

    Returns
    -------
    labels : np.ndarray of shape (N,) with values in {-1, 0, 1}
        -1 = unlabelled / ambiguous (ignored at training)
         0 = bad
         1 = good
    info : dict with the percentile thresholds actually applied.
    """
    if thresholds is None:
        thresholds = AutoLabelThresholds()
    F = np.asarray(features, dtype=np.float64)
    if F.size == 0:
        return np.array([], dtype=np.int8), {}
    tsnr      = F[:, 0]
    dvars_rms = F[:, 1]
    spikes    = F[:, 2]
    drift     = F[:, 4]

    tsnr_bad   = np.percentile(tsnr, thresholds.tsnr_bad_pct)
    tsnr_good  = np.percentile(tsnr, thresholds.tsnr_good_pct)
    dvars_bad  = np.percentile(dvars_rms, thresholds.dvars_bad_pct)
    dvars_good = np.percentile(dvars_rms, thresholds.dvars_good_pct)

    bad = (
        (tsnr <= tsnr_bad)
        & (
            (dvars_rms >= dvars_bad)
            | (spikes >= thresholds.spikes_bad_threshold)
            | (np.abs(drift) > thresholds.drift_bad_threshold_abs)
        )
    )
    good = (
        (tsnr >= tsnr_good)
        & (dvars_rms <= dvars_good)
        & (spikes == 0)
        & (np.abs(drift) < thresholds.drift_good_threshold_abs)
    )
    labels = np.full(F.shape[0], -1, dtype=np.int8)
    labels[bad]  = 0
    labels[good] = 1

    info = {
        "tsnr_threshold_bad":      float(tsnr_bad),
        "tsnr_threshold_good":     float(tsnr_good),
        "dvars_threshold_bad":     float(dvars_bad),
        "dvars_threshold_good":    float(dvars_good),
        "n_voxels_total":          int(F.shape[0]),
        "n_voxels_labelled_bad":   int(bad.sum()),
        "n_voxels_labelled_good":  int(good.sum()),
        "n_voxels_unlabelled":     int((labels == -1).sum()),
        **thresholds.to_dict(),
    }
    return labels, info


# ---------------------------------------------------------------------------
# Cross-patient leave-one-out training
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    held_out_patient: str
    n_train_voxels: int
    n_train_good: int
    n_train_bad: int
    n_test_voxels: int
    auc_test: float
    accuracy_test: float
    quality_prob_test: np.ndarray = field(default=None, repr=False)
    voxel_coords_test: Optional[np.ndarray] = field(default=None, repr=False)


def train_leave_one_out(
    features_per_patient: Dict[str, np.ndarray],
    labels_per_patient: Dict[str, np.ndarray],
    *,
    coords_per_patient: Optional[Dict[str, np.ndarray]] = None,
    rf_kwargs: Optional[Dict] = None,
) -> List[CVResult]:
    """Train a Random Forest with leave-one-patient-out cross validation.

    Parameters
    ----------
    features_per_patient : dict[patient_id -> (N_i, F) feature matrix]
    labels_per_patient   : dict[patient_id -> (N_i,) labels in {-1,0,1}]
    coords_per_patient   : optional dict[patient_id -> (N_i, 3) voxel coords]
        Used to allow downstream callers to write a per-voxel quality NIfTI.
    rf_kwargs : Random Forest hyperparameters passed to sklearn.

    Returns
    -------
    list of ``CVResult`` (one per held-out patient).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score

    _default_rf = dict(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    if rf_kwargs:
        _default_rf.update(rf_kwargs)
    rf_kwargs = _default_rf

    patient_ids = sorted(features_per_patient.keys())
    results: List[CVResult] = []

    for held in patient_ids:
        X_train_parts = []
        y_train_parts = []
        for pid in patient_ids:
            if pid == held:
                continue
            lab = labels_per_patient[pid]
            sel = lab != -1
            if sel.sum() == 0:
                continue
            X_train_parts.append(features_per_patient[pid][sel])
            y_train_parts.append(lab[sel])
        if not X_train_parts:
            continue
        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)
        # Need at least 2 classes
        if len(np.unique(y_train)) < 2:
            continue

        clf = RandomForestClassifier(**rf_kwargs)
        clf.fit(X_train, y_train)

        X_test = features_per_patient[held]
        lab_test = labels_per_patient[held]
        sel_test = lab_test != -1

        prob = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
        if sel_test.sum() > 0 and len(np.unique(lab_test[sel_test])) >= 2:
            auc = float(roc_auc_score(lab_test[sel_test], prob[sel_test]))
            acc = float(accuracy_score(lab_test[sel_test], (prob[sel_test] >= 0.5).astype(int)))
        else:
            auc = float("nan")
            acc = float("nan")

        results.append(CVResult(
            held_out_patient=held,
            n_train_voxels=int(X_train.shape[0]),
            n_train_good=int((y_train == 1).sum()),
            n_train_bad=int((y_train == 0).sum()),
            n_test_voxels=int(X_test.shape[0]),
            auc_test=auc,
            accuracy_test=acc,
            quality_prob_test=prob.astype(np.float32),
            voxel_coords_test=coords_per_patient.get(held) if coords_per_patient else None,
        ))
    return results
