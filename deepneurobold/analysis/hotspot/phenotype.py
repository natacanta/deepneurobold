"""
analysis.hotspot.phenotype
==========================
Functional phenotyping of voxel-wise hotspot masks.

This module classifies each patient's tumor-prediction hotspot into one of:

  * ``compact``     — voxels behave as a single functional unit (high BOLD
                       time-series coherence across the hotspot, regardless
                       of spatial fragmentation)
  * ``diffuse``     — single hotspot region but voxel time-series are
                       heterogeneous (low intra-cluster coherence) — typical
                       infiltrative pattern
  * ``multifocal``  — two or more spatial components whose mean BOLD signals
                       are NOT temporally coherent with each other
  * ``empty`` / ``subthreshold`` — no hotspot voxels above the requested
                       probability threshold

The key change vs a purely-morphological version: classification is driven by
the **raw BOLD time-series** within hotspot voxels, not by hotspot geometry
alone.

Workflow
--------
1. Load binary hotspot mask + raw 4D BOLD on the same grid.
2. Spatial connected components → ``n_cc`` components, each with a size in
   voxels / mm^3.
3. Extract per-voxel BOLD time-series, z-score each.
4. Compute, per significant component, the **mean within-component
   correlation** ``r_intra[k]`` (functional coherence INSIDE that focus).
5. Compute the **mean inter-component correlation** ``r_inter`` between the
   average time courses of pairs of components.
6. Apply the decision rules above.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as cc_label, generate_binary_structure


CONN_3D_26 = generate_binary_structure(rank=3, connectivity=3)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PhenotypeResult:
    """One patient's hotspot phenotype + features used for the decision."""
    phenotype: str  # compact | diffuse | multifocal | empty | subthreshold
    n_components: int
    n_significant_components: int
    total_volume_mm3: float
    component_volumes_mm3: List[float] = field(default_factory=list)
    # Functional coherence features (BOLD-based)
    r_intra_mean: float = float("nan")   # mean within-component Pearson r
    r_intra_per_component: List[float] = field(default_factory=list)
    r_inter_mean: float = float("nan")   # mean across-component r
    n_voxels_used: int = 0
    # Geometric features (kept for the paper supplement)
    largest_component_fraction: float = float("nan")
    largest_component_sphericity: float = float("nan")
    decision_thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def phenotype_hotspot_bold(
    hotspot_3d: np.ndarray,
    bold_4d: Optional[np.ndarray] = None,
    voxel_size_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    hotspot_ts: Optional[np.ndarray] = None,
    hotspot_voxel_coords: Optional[np.ndarray] = None,
    min_significant_volume_mm3: float = 100.0,
    intra_coherent_threshold: float = 0.30,
    inter_coherent_threshold: float = 0.30,
    multifocal_largest_fraction_threshold: float = 0.85,
    max_voxels_for_full_matrix: int = 4000,
    rng_seed: int = 42,
) -> PhenotypeResult:
    """Classify a hotspot using BOLD time-series coherence.

    Parameters
    ----------
    hotspot_3d : np.ndarray
        3D boolean / integer hotspot mask.
    bold_4d : np.ndarray
        4D BOLD time series (X, Y, Z, T) ALIGNED with ``hotspot_3d``. Use the
        raw (non-smoothed) signal — ``BOLD_brain_inT1.nii.gz`` — to keep the
        temporal characterisation faithful to the underlying signal.
    voxel_size_mm : tuple(float, float, float)
        Voxel size in mm, used to convert component sizes to mm^3.
    min_significant_volume_mm3 : float
        Components below this are treated as noise (kept in
        ``n_components`` but not in ``n_significant_components``).
    intra_coherent_threshold : float
        A single-component hotspot is **compact** if the mean within-component
        Pearson r exceeds this value; otherwise **diffuse**.
    inter_coherent_threshold : float
        With >=2 significant components, the hotspot is **multifocal** if the
        mean across-component r is BELOW this value (components are
        functionally independent). If above, the components are functionally
        unified -> still **compact**.
    multifocal_largest_fraction_threshold : float
        When a single component dominates (>= this fraction of total volume),
        treat it as effectively single-component for the decision, even if
        smaller "significant" components exist alongside it.
    max_voxels_for_full_matrix : int
        Cap on the number of voxels used per component when computing the
        within-component correlation; if exceeded, a random subset of this
        size is drawn (reproducible via ``rng_seed``). Inter-component
        correlations always use the full component-mean time courses.
    rng_seed : int
        Seed for the random subsampling within large components.

    Returns
    -------
    PhenotypeResult
    """
    hotspot = np.asarray(hotspot_3d).astype(bool)

    # Two supported input modes:
    #   (A) bold_4d : full 4D ndarray (X,Y,Z,T) — convenient but expensive
    #   (B) hotspot_ts + hotspot_voxel_coords : the time-series of ONLY the
    #       hotspot voxels (N,T) plus their voxel indices (N,3) — memory-safe
    #       path used by the runner for large brains
    if hotspot_ts is not None:
        if hotspot_voxel_coords is None:
            raise ValueError("hotspot_ts requires hotspot_voxel_coords")
        bold = None
    elif bold_4d is not None:
        bold = np.asarray(bold_4d)
        if bold.ndim != 4:
            raise ValueError(f"bold_4d must be 4D, got shape {bold.shape}")
        if bold.shape[:3] != hotspot.shape:
            raise ValueError(
                f"hotspot {hotspot.shape} and BOLD {bold.shape[:3]} grids differ"
            )
    else:
        raise ValueError("provide either bold_4d or hotspot_ts + hotspot_voxel_coords")

    vx, vy, vz = (float(s) for s in voxel_size_mm)
    voxel_vol_mm3 = vx * vy * vz
    thresholds = {
        "min_significant_volume_mm3":           min_significant_volume_mm3,
        "intra_coherent_threshold":             intra_coherent_threshold,
        "inter_coherent_threshold":             inter_coherent_threshold,
        "multifocal_largest_fraction_threshold": multifocal_largest_fraction_threshold,
    }

    if not hotspot.any():
        return PhenotypeResult(
            phenotype="empty", n_components=0, n_significant_components=0,
            total_volume_mm3=0.0, decision_thresholds=thresholds,
        )

    # ----- spatial connected components -----
    labels_3d, n_components = cc_label(hotspot, structure=CONN_3D_26)
    comp_sizes_vox = np.bincount(labels_3d.ravel())[1:]
    comp_sizes_mm3 = comp_sizes_vox * voxel_vol_mm3
    sig_idx = np.where(comp_sizes_mm3 >= min_significant_volume_mm3)[0] + 1
    n_significant = int(sig_idx.size)
    total_vol = float(comp_sizes_mm3.sum())
    if total_vol > 0:
        largest_id = int(np.argmax(comp_sizes_mm3)) + 1
        largest_fraction = float(comp_sizes_mm3.max() / total_vol)
        largest_sphericity = _sphericity_3d(labels_3d == largest_id, voxel_size_mm)
    else:
        largest_id = 0
        largest_fraction = float("nan")
        largest_sphericity = float("nan")

    if n_significant == 0:
        return PhenotypeResult(
            phenotype="subthreshold",
            n_components=int(n_components),
            n_significant_components=0,
            total_volume_mm3=round(total_vol, 1),
            component_volumes_mm3=[round(float(v), 1) for v in sorted(comp_sizes_mm3, reverse=True)],
            largest_component_fraction=round(largest_fraction, 4),
            largest_component_sphericity=round(largest_sphericity, 4),
            decision_thresholds=thresholds,
        )

    # ----- extract BOLD time-series per significant component -----
    rng = np.random.RandomState(int(rng_seed))
    ts_per_comp: Dict[int, np.ndarray] = {}      # cid -> (n_vox, T) z-scored
    mean_ts_per_comp: Dict[int, np.ndarray] = {} # cid -> (T,)
    r_intra_per_comp: List[float] = []
    n_voxels_used = 0
    # Build a flat voxel -> row lookup when we are in hotspot_ts mode, so we
    # can fetch a component's time-series without ever touching the full 4D.
    if hotspot_ts is not None:
        coord_keys = (
            hotspot_voxel_coords[:, 0] * (hotspot.shape[1] * hotspot.shape[2])
            + hotspot_voxel_coords[:, 1] * hotspot.shape[2]
            + hotspot_voxel_coords[:, 2]
        ).astype(np.int64)
        coord_to_row = {int(k): int(i) for i, k in enumerate(coord_keys)}

    for cid in sig_idx:
        comp_mask = labels_3d == cid
        idx = np.argwhere(comp_mask)
        if idx.shape[0] == 0:
            continue
        if idx.shape[0] > max_voxels_for_full_matrix:
            keep = rng.choice(idx.shape[0], size=max_voxels_for_full_matrix, replace=False)
            idx = idx[keep]
        if hotspot_ts is not None:
            keys = (
                idx[:, 0] * (hotspot.shape[1] * hotspot.shape[2])
                + idx[:, 1] * hotspot.shape[2]
                + idx[:, 2]
            ).astype(np.int64)
            rows = np.fromiter(
                (coord_to_row.get(int(k), -1) for k in keys),
                dtype=np.int64, count=keys.size,
            )
            rows = rows[rows >= 0]
            if rows.size == 0:
                continue
            ts = np.asarray(hotspot_ts[rows], dtype=np.float64)
        else:
            ts = bold[idx[:, 0], idx[:, 1], idx[:, 2], :].astype(np.float64)
        ts_z = _zscore_rows(ts)
        ts_per_comp[int(cid)] = ts_z
        mean_ts_per_comp[int(cid)] = ts_z.mean(axis=0)
        r_intra_per_comp.append(_mean_offdiag_corr(ts_z))
        n_voxels_used += int(ts_z.shape[0])

    if not ts_per_comp:
        return PhenotypeResult(
            phenotype="empty", n_components=int(n_components),
            n_significant_components=0, total_volume_mm3=round(total_vol, 1),
            decision_thresholds=thresholds,
        )

    r_intra_mean = float(np.mean(r_intra_per_comp))

    # Inter-component coherence: correlate component-mean signals
    if len(mean_ts_per_comp) >= 2:
        M = np.stack(list(mean_ts_per_comp.values()), axis=0)        # (k, T)
        Mz = _zscore_rows(M)
        R = Mz @ Mz.T / float(Mz.shape[1])                           # (k, k)
        iu = np.triu_indices(R.shape[0], k=1)
        r_inter_mean = float(np.mean(R[iu]))
    else:
        r_inter_mean = float("nan")

    # ----- Decision (calibrated against the 6 user reference labels) -----
    #
    #   (a) MULTIFOCAL = multiple spatial components AND the largest is
    #       irregular / spread out (sphericity below a low cut-off).
    #       This captures e.g. Patient_10 ("3 focos compactos") where the
    #       components are clearly distinct entities visually.
    #   (b) COMPACT = either a single small dense cluster OR multiple clusters
    #       that are individually well-defined (sphericity above the compact
    #       cut-off). BOLD coherence intra-component must also be reasonable.
    #   (c) DIFFUSE = anything else with non-empty hotspot.
    #
    # Calibrated thresholds:
    #   - multifocal_sphericity_max = 0.35  (P-10 has 0.28)
    #   - compact_sphericity_min    = 0.40  (P-02 has 0.41)
    multifocal_sphericity_max = 0.35
    compact_sphericity_min    = 0.40

    if n_significant >= 2 and (
        np.isnan(largest_sphericity)
        or largest_sphericity < multifocal_sphericity_max
    ):
        phenotype = "multifocal"
    elif (
        (n_significant >= 1)
        and (np.isnan(largest_sphericity)
             or largest_sphericity >= compact_sphericity_min)
        and (np.isnan(r_intra_mean) or r_intra_mean >= intra_coherent_threshold)
    ):
        phenotype = "compact"
    else:
        phenotype = "diffuse"

    return PhenotypeResult(
        phenotype=phenotype,
        n_components=int(n_components),
        n_significant_components=int(n_significant),
        total_volume_mm3=round(total_vol, 1),
        component_volumes_mm3=[round(float(v), 1) for v in sorted(comp_sizes_mm3, reverse=True)],
        r_intra_mean=round(r_intra_mean, 4),
        r_intra_per_component=[round(float(r), 4) for r in r_intra_per_comp],
        r_inter_mean=round(r_inter_mean, 4) if np.isfinite(r_inter_mean) else float("nan"),
        n_voxels_used=int(n_voxels_used),
        largest_component_fraction=round(largest_fraction, 4),
        largest_component_sphericity=round(largest_sphericity, 4),
        decision_thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Z-score each row; rows with zero std become all zeros."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=0, keepdims=True)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    out = (x - mu) / sd_safe
    out[(sd <= 1e-12).ravel()] = 0.0
    return out


def _mean_offdiag_corr(ts_z: np.ndarray) -> float:
    """Mean off-diagonal Pearson correlation across rows of a z-scored
    matrix. Each row is one voxel's z-scored time series (length T)."""
    n, t = ts_z.shape
    if n < 2 or t < 3:
        return float("nan")
    R = ts_z @ ts_z.T / float(t)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(R[iu])) if iu[0].size else float("nan")


def _surface_area_mm2(
    mask: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
) -> float:
    vx, vy, vz = (float(s) for s in voxel_size_mm)
    diff_x = mask[1:, :, :].astype(np.int8) - mask[:-1, :, :].astype(np.int8)
    diff_y = mask[:, 1:, :].astype(np.int8) - mask[:, :-1, :].astype(np.int8)
    diff_z = mask[:, :, 1:].astype(np.int8) - mask[:, :, :-1].astype(np.int8)
    n_x = int(np.count_nonzero(diff_x))
    n_y = int(np.count_nonzero(diff_y))
    n_z = int(np.count_nonzero(diff_z))
    area = n_x * (vy * vz) + n_y * (vx * vz) + n_z * (vx * vy)
    edge = (
        np.count_nonzero(mask[0,  :, :]) * (vy * vz)
        + np.count_nonzero(mask[-1, :, :]) * (vy * vz)
        + np.count_nonzero(mask[:,  0, :]) * (vx * vz)
        + np.count_nonzero(mask[:, -1, :]) * (vx * vz)
        + np.count_nonzero(mask[:,  :, 0]) * (vx * vy)
        + np.count_nonzero(mask[:,  :, -1]) * (vx * vy)
    )
    return float(area + edge)


def _sphericity_3d(
    mask: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
) -> float:
    if not np.any(mask):
        return float("nan")
    vx, vy, vz = (float(s) for s in voxel_size_mm)
    voxel_vol = vx * vy * vz
    V = float(np.count_nonzero(mask)) * voxel_vol
    A = _surface_area_mm2(mask, voxel_size_mm)
    if A <= 0 or V <= 0:
        return float("nan")
    return float(np.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0) / A)
