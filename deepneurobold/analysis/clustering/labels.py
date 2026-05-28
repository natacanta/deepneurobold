"""
analysis.clustering.labels
==========================
Build region masks from a tumor segmentation NIfTI in T1 space.

Label convention (inside → outside):
    1 = necrosis  (core)
    3 = enhancement (T1CE ring)
    2 = edema     (FLAIR hyperintense outer zone)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

# SciPy is optional — used only for seed erosions
try:
    from scipy.ndimage import binary_erosion, generate_binary_structure
except Exception:
    binary_erosion = None  # type: ignore
    generate_binary_structure = None  # type: ignore

__all__ = ["LABEL_MAP_INSIDE_OUT", "build_masks_from_seg"]

LABEL_MAP_INSIDE_OUT: Dict[int, str] = {
    1: "necro",
    3: "enh",
    2: "edema",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_mask_like(ref_img: nib.Nifti1Image, mask: np.ndarray, path: Path) -> None:
    """Save a boolean mask as uint8, preserving affine/header from *ref_img*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(mask.astype(np.uint8, copy=False), ref_img.affine, ref_img.header)
    out.header.set_data_dtype(np.uint8)
    nib.save(out, str(path))


def _load_bool_nii(p: Path) -> Optional[np.ndarray]:
    """Load a NIfTI and return a boolean array (>0), or None if missing/empty."""
    if not p.exists():
        return None
    img = nib.load(str(p), mmap=True)
    arr = np.asarray(img.dataobj)
    if arr.size == 0:
        return None
    return arr > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_masks_from_seg(
    pdir: Path,
    *,
    seg_name: str = "Segmentation_in_T1.nii.gz",
    brain_mask_name: str = "t1_brain_mask.nii.gz",
    erosion_iters: int = 1,
    save_out: bool = True,
    label_map_inside_out: Optional[Dict[int, str]] = None,
    csf_label: Optional[int] = 4,
    use_brats: bool = False,
) -> Tuple[Dict[str, np.ndarray], nib.Nifti1Image]:
    """
    Build region masks from a segmentation in T1 space.

    Parameters
    ----------
    pdir : Path
        Patient directory (contains PREPROCESSING/ sub-tree).
    seg_name : str
        Filename of the segmentation NIfTI inside PREPROCESSING/mask/.
    brain_mask_name : str
        Filename of the brain mask inside PREPROCESSING/t1/.
    erosion_iters : int
        Number of binary erosion iterations for seed masks (0 = skip).
    save_out : bool
        Whether to write masks and a QC JSON to Analysis/.
    label_map_inside_out : dict, optional
        Override the default {1: "necro", 3: "enh", 2: "edema"} mapping.
    csf_label : int, optional
        Segmentation label for CSF.  Used as fallback if no csf_mask_inT1.nii.gz.
    use_brats : bool
        If True, use PREPROCESSING/mask/BRATS/Segmentation_BRATS_in_T1.nii.gz.

    Returns
    -------
    masks : dict
        Keys always present: necro, enh, edema, tumor, healthy,
        seed_necro, seed_enh, seed_edema.
        Optional key: csf (if found from file or label).
    ref_img : nib.Nifti1Image
        Reference image (affine/header from the segmentation file).
    """
    pdir    = Path(pdir)
    t1_dir  = pdir / "PREPROCESSING" / "t1"
    maskdir = pdir / "PREPROCESSING" / "mask"

    if use_brats:
        seg_p = maskdir / "BRATS" / "Segmentation_BRATS_in_T1.nii.gz"
    else:
        seg_p = maskdir / seg_name

    brain_p = t1_dir  / brain_mask_name
    csf_p   = maskdir / "csf_mask_inT1.nii.gz"

    if not seg_p.exists():
        raise FileNotFoundError(f"Missing segmentation: {seg_p}")
    if not brain_p.exists():
        raise FileNotFoundError(f"Missing brain mask: {brain_p}")

    seg_img = nib.load(str(seg_p), mmap=True)
    seg     = np.asarray(seg_img.dataobj).astype(np.int32, copy=False)
    ref_img = seg_img

    brain_img = nib.load(str(brain_p), mmap=True)
    brain     = np.asarray(brain_img.dataobj) > 0

    if seg.shape != brain.shape:
        raise RuntimeError(f"Shape mismatch: seg{seg.shape} vs brain{brain.shape}")

    label_map = label_map_inside_out or LABEL_MAP_INSIDE_OUT
    masks: Dict[str, np.ndarray] = {}

    # 1) Individual region masks
    for lab, name in label_map.items():
        masks[name] = (seg == int(lab)) & brain

    # 2) Tumor union + healthy
    tumor         = masks["necro"] | masks["enh"] | masks["edema"]
    masks["tumor"] = tumor

    healthy_path = maskdir / "healthy_mask.nii.gz"
    if healthy_path.exists():
        h = _load_bool_nii(healthy_path)
        healthy = (h & brain) if (h is not None and h.shape == brain.shape) else (brain & ~tumor)
    else:
        healthy = brain & ~tumor

    # 3) CSF — prefer external file, fallback to label
    csf_mask: Optional[np.ndarray] = None

    csf_from_file = _load_bool_nii(csf_p)
    if csf_from_file is not None and csf_from_file.shape == brain.shape:
        csf_mask = csf_from_file & brain

    if csf_mask is None and csf_label is not None:
        try:
            candidate = (seg == int(csf_label)) & brain
            if candidate.any():
                csf_mask = candidate
        except Exception:
            csf_mask = None

    if csf_mask is not None:
        masks["csf"] = csf_mask
        healthy = healthy & ~csf_mask

    masks["healthy"] = healthy

    # 4) Seed masks via binary erosion
    iters = max(0, int(erosion_iters))
    if iters > 0 and binary_erosion is not None and generate_binary_structure is not None:
        st = generate_binary_structure(rank=3, connectivity=1)
        for src_key, dst_key in [
            ("necro", "seed_necro"),
            ("enh",   "seed_enh"),
            ("edema", "seed_edema"),
        ]:
            src = masks[src_key]
            masks[dst_key] = (
                binary_erosion(src, structure=st, iterations=iters) if src.any() else src.copy()
            )
    else:
        masks["seed_necro"] = masks["necro"].copy()
        masks["seed_enh"]   = masks["enh"].copy()
        masks["seed_edema"] = masks["edema"].copy()

    # 5) Optionally persist masks + QC JSON
    if save_out:
        outdir = (
            pdir / "Analysis" / "BRATS" / "Masks"
            if use_brats
            else pdir / "Analysis" / "Guided" / "Masks"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        for k, m in masks.items():
            _save_mask_like(ref_img, m, outdir / f"{k}.nii.gz")
        try:
            counts = {k: int(m.sum()) for k, m in masks.items() if not k.startswith("seed_")}
            (outdir / "mask_counts.json").write_text(json.dumps(counts, indent=2))
        except Exception:
            pass

    return masks, ref_img
