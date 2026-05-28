"""
analysis.consensus_tumor
========================
Consensus tumor core and consensus edema masks derived from the available
segmentation sources.

Consensus recipe:

  * **Edema-plus** = (union of ALL segmenters' edema/whole-tumor masks)
                     dilated by ~1 cm. Everything OUTSIDE this is "background"
                     for sampling.
  * **Tumor core-plus** = (union of ALL segmenters' tumor-core masks)
                          dilated by a small safety margin. Anything inside
                          this is "tumor" for training.

  The band between core+margin and edema-plus is the "in-between" zone — the
  region where the classifier predictions are clinically meaningful.

Inputs accepted (all optional, mixed-source supported):
  - BRATS-style integer mask in T1 native space (1=NETC/necrosis, 2=edema,
    3=enhancing).
  - Oncohabitats-style mask (same integer convention; only the labels we
    can resolve count).
  - SynthSeg parcellation (used only to flag CSF voxels — never tumor).

A single source is sufficient — the rest are skipped gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_EDEMA_DILATION_MM   = 10.0
DEFAULT_TUMOR_MARGIN_MM     = 2.0
DEFAULT_BRATS_LABEL_MAP     = {"necrosis": (1,), "edema": (2,), "enhancing": (3,)}
DEFAULT_ONCO_LABEL_MAP      = {"necrosis": (1,), "edema": (2,), "enhancing": (3,)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mm_to_iters(mm: float, voxel_size_mm: Tuple[float, float, float]) -> int:
    """Same convention as build_train_masks._mm_to_iters."""
    vmean = float(np.mean([float(v) for v in voxel_size_mm]))
    return max(1, int(round(float(mm) / vmean)))


def _labels_to_mask(
    seg_3d: np.ndarray,
    labels: Iterable[int],
) -> np.ndarray:
    return np.isin(seg_3d, list(labels))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ConsensusResult:
    """Container for the per-source and consensus masks (all flat bool)."""
    tumor_core_consensus_flat: np.ndarray
    tumor_core_with_margin_flat: np.ndarray
    edema_consensus_flat: np.ndarray
    edema_plus_1cm_flat: np.ndarray
    inbetween_flat: np.ndarray
    csf_excluded_flat: np.ndarray
    voxel_counts: Dict[str, int]


def consensus_tumor_masks(
    *,
    brats_seg: Optional[np.ndarray] = None,
    onco_seg: Optional[np.ndarray] = None,
    synthseg_seg: Optional[np.ndarray] = None,
    voxel_size_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    edema_dilation_mm: float = DEFAULT_EDEMA_DILATION_MM,
    tumor_margin_mm: float = DEFAULT_TUMOR_MARGIN_MM,
    brats_label_map: Dict[str, Tuple[int, ...]] = None,
    onco_label_map: Dict[str, Tuple[int, ...]] = None,
) -> ConsensusResult:
    """Combine BRATS, Oncohabitats and SynthSeg masks into a single consensus.

    Outputs ALL share the same flat shape (X*Y*Z,) and dtype ``bool``.

    Definitions
    -----------
    ``tumor_core_consensus`` = (BRATS-necr | BRATS-enh) | (ONCO-necr | ONCO-enh)
        — union of both segmenters' tumor-core labels.
    ``tumor_core_with_margin`` = ``tumor_core_consensus`` dilated by ``tumor_margin_mm``.
    ``edema_consensus``    = ``tumor_core_consensus`` | BRATS-edema | ONCO-edema
        — anything any segmenter marked as tumor or peritumoral edema.
    ``edema_plus_1cm``     = ``edema_consensus`` dilated by ``edema_dilation_mm``.
    ``inbetween``          = ``edema_plus_1cm`` & ~``tumor_core_with_margin``
        — the "interesting" perilesional band where hotspots live.
    ``csf_excluded``       = SynthSeg CSF labels (excluded from positive/neg pools).
    """
    if brats_label_map is None:
        brats_label_map = DEFAULT_BRATS_LABEL_MAP
    if onco_label_map is None:
        onco_label_map = DEFAULT_ONCO_LABEL_MAP

    # Discover the volume shape from whichever input we have.
    for cand in (brats_seg, onco_seg, synthseg_seg):
        if cand is not None:
            shape = tuple(np.asarray(cand).shape)
            break
    else:
        raise ValueError("Provide at least one of brats_seg / onco_seg / synthseg_seg.")

    def _empty():
        return np.zeros(shape, dtype=bool)

    # Per-source tumor core / edema
    sources = {}
    if brats_seg is not None:
        b = np.asarray(brats_seg)
        sources["brats_core"]  = _labels_to_mask(b, brats_label_map["necrosis"] + brats_label_map["enhancing"])
        sources["brats_edema"] = _labels_to_mask(b, brats_label_map["edema"])
    if onco_seg is not None:
        o = np.asarray(onco_seg)
        sources["onco_core"]   = _labels_to_mask(o, onco_label_map["necrosis"] + onco_label_map["enhancing"])
        sources["onco_edema"]  = _labels_to_mask(o, onco_label_map["edema"])
    # SynthSeg: only CSF (to mask out)
    csf = _empty()
    if synthseg_seg is not None:
        from deepneurobold.data.masks import SYNTHSEG_CSF_LABELS
        csf = np.isin(np.asarray(synthseg_seg), list(SYNTHSEG_CSF_LABELS))

    # Build consensus
    core_consensus = _empty()
    for k in ("brats_core", "onco_core"):
        if k in sources:
            core_consensus |= sources[k]
    edema_consensus = core_consensus.copy()
    for k in ("brats_edema", "onco_edema"):
        if k in sources:
            edema_consensus |= sources[k]

    # Remove CSF from BOTH (avoids training inside ventricles)
    if csf.any():
        core_consensus  = core_consensus  & (~csf)
        edema_consensus = edema_consensus & (~csf)

    # Dilations
    tumor_iters = _mm_to_iters(tumor_margin_mm, voxel_size_mm)
    edema_iters = _mm_to_iters(edema_dilation_mm, voxel_size_mm)
    core_with_margin = binary_dilation(core_consensus, iterations=tumor_iters) if tumor_iters else core_consensus.copy()
    edema_plus_1cm   = binary_dilation(edema_consensus, iterations=edema_iters) if edema_iters else edema_consensus.copy()
    if csf.any():
        core_with_margin = core_with_margin & (~csf)
        edema_plus_1cm   = edema_plus_1cm   & (~csf)

    inbetween = edema_plus_1cm & (~core_with_margin)

    counts = {
        "tumor_core_consensus":   int(core_consensus.sum()),
        "tumor_core_with_margin": int(core_with_margin.sum()),
        "edema_consensus":        int(edema_consensus.sum()),
        "edema_plus_1cm":         int(edema_plus_1cm.sum()),
        "inbetween":              int(inbetween.sum()),
        "csf_excluded":           int(csf.sum()),
    }
    for k, v in sources.items():
        counts[k] = int(v.sum())

    return ConsensusResult(
        tumor_core_consensus_flat   = core_consensus.ravel(),
        tumor_core_with_margin_flat = core_with_margin.ravel(),
        edema_consensus_flat        = edema_consensus.ravel(),
        edema_plus_1cm_flat         = edema_plus_1cm.ravel(),
        inbetween_flat              = inbetween.ravel(),
        csf_excluded_flat           = csf.ravel(),
        voxel_counts                = counts,
    )
