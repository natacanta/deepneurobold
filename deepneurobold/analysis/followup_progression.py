"""
analysis.followup_progression
=============================
Track tumor enhancement progression from baseline to follow-up T1CE images
(the only follow-up modality available in this study) and compute
hotspot-vs-progression overlap statistics.

Workflow per patient
--------------------
1. Locate baseline T1CE (data/Patient_XX/PREPROCESSING/t1/t1ce_inT1_ants.nii.gz)
   and all follow-up T1CE (FOLLOW-UP_REGISTERED_NEW/Patient_XX/t1ce_registered/
   T1CE_<DATE>_inT1.nii.gz).
2. Segment "enhancing tumor" on each T1CE by intensity thresholding inside the
   brain, normalised to the median WM signal (or a fallback whole-brain
   median when WM mask is unavailable). Threshold default 1.5 × median.
3. Compute, per follow-up, ``new_enhancement = enh_followup & ~enh_baseline``
   restricted to brain. This gives the spatial map of progression relative to
   the baseline imaging.
4. Compute the overlap statistics between pre-op hotspot mask and the
   new-enhancement maps: voxel counts, Dice, fraction of new enhancement
   that was already predicted by the hotspot.

Outputs (per patient)
---------------------
- ``results/07_followup_progression/Patient_XX/
    enh_baseline.nii.gz``
- ``results/07_followup_progression/Patient_XX/
    enh_<followup_label>.nii.gz``
- ``results/07_followup_progression/Patient_XX/
    new_enh_<followup_label>.nii.gz``
- ``results/07_followup_progression/Patient_XX/progression.json``
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FollowupProgression:
    label: str                          # follow-up acquisition date, dd.mm.yyyy
    n_enh_voxels: int                   # voxels classified as enhancing in the FU
    n_new_enh_voxels: int               # FU enh that were not present at baseline
    fraction_new: float                 # n_new / n_enh
    overlap_with_hotspot_voxels: int    # new_enh ∩ hotspot
    fraction_new_predicted: float       # overlap / n_new
    dice_new_vs_hotspot: float          # Dice between new_enh and hotspot

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FollowupProgressionReport:
    patient_id: str
    baseline_threshold: float
    baseline_normalisation: str         # "median_wm" or "median_brain"
    n_baseline_enh_voxels: int
    n_hotspot_voxels: int
    followups: List[FollowupProgression] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["followups"] = [fu.to_dict() for fu in self.followups]
        return d


def segment_enhancement_t1ce(
    t1ce_3d: np.ndarray,
    brain_mask_3d: np.ndarray,
    *,
    wm_mask_3d: Optional[np.ndarray] = None,
    threshold_factor: float = 1.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Segment "enhancing tumor" voxels on a T1CE volume.

    Voxels are flagged when their intensity exceeds ``threshold_factor`` × the
    median signal in white matter (or the brain if WM unavailable). This is a
    simple, transparent rule — the goal here is progression tracking, not
    surface-perfect segmentation. Outliers and skull are excluded by the
    brain mask.

    Returns the binary 3D mask plus the normalisation diagnostics used.
    """
    t1ce = np.asarray(t1ce_3d, dtype=np.float64)
    brain = np.asarray(brain_mask_3d, dtype=bool)
    if wm_mask_3d is not None and np.any(wm_mask_3d):
        ref_vals = t1ce[(brain & np.asarray(wm_mask_3d, dtype=bool))]
        norm_kind = "median_wm"
    else:
        ref_vals = t1ce[brain]
        norm_kind = "median_brain"
    median = float(np.median(ref_vals)) if ref_vals.size else float("nan")
    if not np.isfinite(median) or median <= 0:
        return np.zeros_like(brain, dtype=bool), {
            "median_reference": median, "threshold": float("nan"), "norm_kind": norm_kind,
        }
    threshold = median * float(threshold_factor)
    mask = (t1ce >= threshold) & brain
    return mask, {
        "median_reference": median,
        "threshold": float(threshold),
        "norm_kind": norm_kind,
    }


def followup_progression_for_patient(
    patient_dir: Path,
    followup_dir: Path,
    hotspot_3d: Optional[np.ndarray] = None,
    *,
    brain_mask_3d: Optional[np.ndarray] = None,
    wm_mask_3d: Optional[np.ndarray] = None,
    threshold_factor: float = 1.5,
) -> Tuple[FollowupProgressionReport, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build the progression report for one patient.

    Parameters
    ----------
    patient_dir : Path
        ``data/Patient_XX/`` directory.
    followup_dir : Path
        ``FOLLOW-UP_REGISTERED_NEW/Patient_XX/t1ce_registered/`` directory.
    hotspot_3d : np.ndarray, optional
        Binary hotspot mask in the patient's T1 grid. If omitted, overlap
        statistics are reported as NaN but enhancement segmentation still
        runs.
    brain_mask_3d : np.ndarray, optional
        Brain mask in T1 native grid; default = read from
        ``PREPROCESSING/t1/t1_brain_mask.nii.gz``.
    wm_mask_3d : np.ndarray, optional
        White-matter mask in T1 native grid (for intensity normalisation).
        Default = ``PREPROCESSING/mask/wm_healthy_pve.nii.gz`` thresholded > 0.5.

    Returns
    -------
    report, baseline_enh_dict, followup_enh_dict
    """
    patient_dir = Path(patient_dir)
    followup_dir = Path(followup_dir)
    pid = patient_dir.name

    base_t1ce_path = patient_dir / "PREPROCESSING" / "t1" / "t1ce_inT1_ants.nii.gz"
    if not base_t1ce_path.exists():
        raise FileNotFoundError(f"baseline T1CE not found: {base_t1ce_path}")
    base_img = nib.load(str(base_t1ce_path))
    base_arr = np.asarray(base_img.get_fdata(), dtype=np.float32)

    if brain_mask_3d is None:
        bmp = patient_dir / "PREPROCESSING" / "t1" / "t1_brain_mask.nii.gz"
        if bmp.exists():
            brain_mask_3d = np.asarray(nib.load(str(bmp)).get_fdata()) > 0
        else:
            brain_mask_3d = base_arr > 0
    if wm_mask_3d is None:
        wmp = patient_dir / "PREPROCESSING" / "mask" / "wm_healthy_pve.nii.gz"
        if wmp.exists():
            wm_mask_3d = np.asarray(nib.load(str(wmp)).get_fdata()) > 0.5

    enh_base, diag = segment_enhancement_t1ce(
        base_arr, brain_mask_3d, wm_mask_3d=wm_mask_3d,
        threshold_factor=threshold_factor,
    )

    n_hot = int(hotspot_3d.sum()) if hotspot_3d is not None else 0
    report = FollowupProgressionReport(
        patient_id=pid,
        baseline_threshold=float(diag.get("threshold", float("nan"))),
        baseline_normalisation=str(diag.get("norm_kind", "?")),
        n_baseline_enh_voxels=int(enh_base.sum()),
        n_hotspot_voxels=n_hot,
    )

    fu_paths = sorted(followup_dir.glob("T1CE_*_inT1.nii.gz"))
    followup_enh_dict: Dict[str, np.ndarray] = {}
    baseline_enh_dict = {"baseline": enh_base}

    for p in fu_paths:
        label = p.stem.replace("T1CE_", "").replace("_inT1", "")
        fu_img = nib.load(str(p))
        # Resample to baseline grid if needed (should already match because of
        # the "_inT1" naming, but be defensive — the cluster has had off-by-one
        # affine drift in some patients).
        if fu_img.shape != base_img.shape or not np.allclose(fu_img.affine, base_img.affine, atol=1e-3):
            fu_img = resample_from_to(fu_img, base_img, order=1)
        fu_arr = np.asarray(fu_img.get_fdata(), dtype=np.float32)
        enh_fu, _ = segment_enhancement_t1ce(
            fu_arr, brain_mask_3d, wm_mask_3d=wm_mask_3d,
            threshold_factor=threshold_factor,
        )
        followup_enh_dict[label] = enh_fu

        new_enh = enh_fu & (~enh_base)
        n_enh = int(enh_fu.sum())
        n_new = int(new_enh.sum())
        frac_new = float(n_new / n_enh) if n_enh else float("nan")

        if hotspot_3d is not None and n_new > 0 and n_hot > 0:
            inter = int((new_enh & hotspot_3d).sum())
            frac_new_pred = float(inter / n_new)
            dice = float(2 * inter / (n_new + n_hot))
        else:
            inter = 0
            frac_new_pred = float("nan")
            dice = float("nan")

        report.followups.append(FollowupProgression(
            label=label,
            n_enh_voxels=n_enh,
            n_new_enh_voxels=n_new,
            fraction_new=round(frac_new, 4) if np.isfinite(frac_new) else float("nan"),
            overlap_with_hotspot_voxels=inter,
            fraction_new_predicted=round(frac_new_pred, 4) if np.isfinite(frac_new_pred) else float("nan"),
            dice_new_vs_hotspot=round(dice, 4) if np.isfinite(dice) else float("nan"),
        ))

    return report, baseline_enh_dict, followup_enh_dict
