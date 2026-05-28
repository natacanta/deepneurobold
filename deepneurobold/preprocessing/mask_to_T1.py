#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_to_T1.py — robust segmentation alignment into native T1 space.

Two paths, selected by an explicit file convention
--------------------------------------------------

1. **Naïve header-based resample** (default, legacy path for all original
   patients):

        Segmentation.nii.gz  --resample_from_to-->  Segmentation_in_T1.nii.gz

   Works whenever the segmentation already shares the native T1 physical
   coordinate system (only shape / voxel size differ). This is the case
   for almost every patient produced by our standard Oncohabitats pipeline.

2. **Image-based registration fallback** (triggered when the user uploads
   an auxiliary reference image):

        Segmentation_oncohabitats_T1.nii.gz  ──dipy MI pyramid──>  T1
                       │
                       └──nearest-neighbour warp──>  Segmentation_in_T1.nii.gz

   Required when Oncohabitats is re-run and the output has a header that
   looks correct (same shape / affine as our T1) but the actual voxel
   content has been rotated/translated by the external pipeline, so
   the naive header math is the identity and the segmentation lands in
   the wrong anatomical location.

Trigger
-------
The trigger is **purely file-based**, deterministic, and explicit:

    if exists(PREPROCESSING/mask/Segmentation_oncohabitats_T1.nii.gz):
        method = "image_based_registration"
    else:
        method = "naive_resample"

No image-similarity heuristics — those cannot reliably distinguish
"misaligned inside the brain" from "correctly aligned" without a tumor-aware
reference image. To force the registration path for a problematic patient,
the operator simply uploads the Oncohabitats T1 to that path.

Post-conditions written to ``PREPROCESSING/mask/Segmentation_in_T1_audit.json``
regardless of which path was taken:
  - shape / affine match the target T1
  - segmentation labels ⊂ {0, 1, 2, 3}
  - each non-zero label has ≥ ``min_voxels_per_label`` voxels (default 50)
  - voxel counts per label
  - method used + final affine
  - (registration path only) final MI metric value
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MIN_VOXELS_PER_LABEL = 50
ALLOWED_LABELS = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MaskToT1:
    """
    Align an Oncohabitats segmentation to the native T1.

    Parameters
    ----------
    interp_order : int
        Interpolation order for the naïve ``resample_from_to`` path.
        ``0`` = nearest-neighbour (preserves labels). Do not change unless
        you know what you are doing.
    min_voxels_per_label : int, default 50
        Post-condition: each non-zero label in the output must have at
        least this many voxels, otherwise a warning is logged.
    """

    def __init__(
        self,
        interp_order: int = 0,
        min_voxels_per_label: int = DEFAULT_MIN_VOXELS_PER_LABEL,
    ) -> None:
        self.interp_order = int(interp_order)
        self.min_voxels_per_label = int(min_voxels_per_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, patient_dir: Path) -> Optional[str]:
        return self._onco_to_t1(Path(patient_dir))

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def _onco_to_t1(self, patient_dir: Path) -> Optional[str]:
        pre = patient_dir / "PREPROCESSING"
        seg_src = pre / "mask" / "Segmentation.nii.gz"
        t1_ref = pre / "t1" / "t1_reoriented.nii.gz"
        seg_out = pre / "mask" / "Segmentation_in_T1.nii.gz"
        audit_out = pre / "mask" / "Segmentation_in_T1_audit.json"
        oh_t1_path = pre / "mask" / "Segmentation_oncohabitats_T1.nii.gz"

        print(f"\n[mask_to_T1] === {patient_dir.name} ===")
        if not seg_src.exists() or not t1_ref.exists():
            print(f"[mask_to_T1] [ERROR] missing required files:")
            print(f"            seg_src   = {seg_src}    exists={seg_src.exists()}")
            print(f"            t1_ref    = {t1_ref}     exists={t1_ref.exists()}")
            return None

        t1_nib = nib.load(str(t1_ref))
        seg_nib = nib.load(str(seg_src))

        oh_t1_provided = oh_t1_path.exists()
        method = "image_based_registration" if oh_t1_provided else "naive_resample"

        audit: Dict[str, Any] = {
            "patient_dir": str(patient_dir),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "input": {
                "segmentation": str(seg_src),
                "t1_ref": str(t1_ref),
                "oncohabitats_t1": str(oh_t1_path) if oh_t1_provided else None,
            },
            "method": method,
            "trigger": {
                "rule": ("Segmentation_oncohabitats_T1.nii.gz "
                         "presence triggers image-based registration"),
                "file_exists": bool(oh_t1_provided),
            },
            "header_comparison": self._compare_headers(seg_nib, t1_nib),
        }
        print(f"[mask_to_T1] method = {method}  "
              f"(OH T1 {'provided' if oh_t1_provided else 'NOT provided'})")

        # -------- Stage: run the selected path --------
        if method == "naive_resample":
            seg_final = self._naive_resample(seg_nib, t1_nib)
            audit["registration_metric_value"] = None
        else:
            seg_final, reg_metric, reg_affine = self._registration_fallback(
                seg_nib=seg_nib, t1_nib=t1_nib, oh_t1_path=oh_t1_path,
            )
            audit["registration_metric_value"] = (
                float(reg_metric) if reg_metric is not None else None
            )
            audit["registration_affine_oh_to_my_t1"] = reg_affine.tolist()

        # -------- Save final result --------
        seg_data = np.asarray(seg_final.get_fdata(), dtype=np.uint8)
        final_nii = nib.Nifti1Image(seg_data, t1_nib.affine, t1_nib.header)
        final_nii.set_data_dtype(np.uint8)
        nib.save(final_nii, str(seg_out))

        # QA border (visual overlay aid)
        self._save_qa_border(seg_data, t1_nib, pre / "mask" / "qa")

        # Post-conditions
        post = self._post_conditions(seg_out, t1_ref, seg_data)
        audit["post_conditions"] = post
        audit["voxel_counts"] = {f"label_{lbl}": int(np.sum(seg_data == lbl))
                                  for lbl in ALLOWED_LABELS}
        audit["final_affine"] = t1_nib.affine.tolist()

        self._write_audit(audit_out, audit)

        # -------- Console summary --------
        print(f"[mask_to_T1] [OK] {method:>26s}  →  {seg_out}")
        for lbl in (1, 2, 3):
            n = int(np.sum(seg_data == lbl))
            warn = "  ⚠ < min" if 0 < n < self.min_voxels_per_label else ""
            print(f"            label {lbl} = {n:>7d} voxels{warn}")
        print(f"            audit:  {audit_out}")
        return str(seg_out)

    # ------------------------------------------------------------------
    # Naïve resample (legacy path — preserves trial_143 byte-identical)
    # ------------------------------------------------------------------
    def _naive_resample(self, seg_nib, t1_nib) -> nib.Nifti1Image:
        seg_ras = nib.as_closest_canonical(seg_nib)
        return resample_from_to(seg_ras, t1_nib, order=self.interp_order)

    # ------------------------------------------------------------------
    # Image-based registration fallback
    # ------------------------------------------------------------------
    def _registration_fallback(
        self,
        seg_nib,
        t1_nib,
        oh_t1_path: Path,
    ) -> Tuple[nib.Nifti1Image, Optional[float], np.ndarray]:
        """
        Register Oncohabitats' own T1 reconstruction onto the patient's T1
        with dipy's MI pyramid, then apply the affine to the segmentation
        with nearest-neighbour interpolation to preserve labels.
        """
        try:
            from dipy.align.imaffine import (
                AffineRegistration, MutualInformationMetric,
                transform_centers_of_mass,
            )
            from dipy.align.transforms import (
                AffineTransform3D, RigidTransform3D, TranslationTransform3D,
            )
        except ImportError as e:
            raise ImportError(
                "Image-based fallback requires dipy. The server environment "
                "<CONDA_ENV>/ includes dipy. "
                f"Original error: {e}"
            )

        oh_t1_nib = nib.load(str(oh_t1_path))
        fixed_nib = nib.as_closest_canonical(t1_nib)
        moving_nib = nib.as_closest_canonical(oh_t1_nib)
        seg_can = nib.as_closest_canonical(seg_nib)

        fixed = np.asarray(fixed_nib.get_fdata(), dtype=np.float32)
        moving = np.asarray(moving_nib.get_fdata(), dtype=np.float32)
        seg_mov = np.asarray(seg_can.get_fdata(), dtype=np.uint8)

        fixed_aff = fixed_nib.affine
        moving_aff = moving_nib.affine

        print(f"[mask_to_T1] [register] fixed shape={fixed.shape}  "
              f"moving shape={moving.shape}")

        # 1. Centre-of-mass init
        c_of_mass = transform_centers_of_mass(
            fixed, fixed_aff, moving, moving_aff,
        )
        starting_affine = c_of_mass.affine

        # 2. Set up MI metric + 3-level pyramid
        metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
        affreg = AffineRegistration(
            metric=metric,
            level_iters=[10000, 1000, 100],
            sigmas=[3.0, 1.0, 0.0],
            factors=[4, 2, 1],
        )

        print("[mask_to_T1] [register] level 1/3 — translation")
        translation = affreg.optimize(
            fixed, moving, TranslationTransform3D(), None,
            fixed_aff, moving_aff, starting_affine=starting_affine,
        )
        print("[mask_to_T1] [register] level 2/3 — rigid")
        rigid = affreg.optimize(
            fixed, moving, RigidTransform3D(), None,
            fixed_aff, moving_aff, starting_affine=translation.affine,
        )
        print("[mask_to_T1] [register] level 3/3 — affine")
        affine_xfm = affreg.optimize(
            fixed, moving, AffineTransform3D(), None,
            fixed_aff, moving_aff, starting_affine=rigid.affine,
        )

        try:
            final_metric = float(affreg.metric.metric_val)
        except Exception:
            final_metric = None

        # 3. Apply transform to the segmentation (NN to preserve labels).
        # dipy's transform method requires float32 input; we cast, apply
        # nearest-neighbour interpolation, then round + cast back to uint8.
        print("[mask_to_T1] [register] applying affine to segmentation (NN)")
        warped_seg = affine_xfm.transform(
            seg_mov.astype(np.float32),
            interpolation="nearest",
            sampling_grid_shape=fixed.shape,
        )
        warped_seg = np.rint(warped_seg).astype(np.uint8)
        # Defensive: clamp to allowed labels
        bad = ~np.isin(warped_seg, ALLOWED_LABELS)
        warped_seg[bad] = 0

        return (
            nib.Nifti1Image(warped_seg, fixed_aff, fixed_nib.header),
            final_metric,
            affine_xfm.affine,
        )

    # ------------------------------------------------------------------
    # QA + audit helpers
    # ------------------------------------------------------------------
    def _save_qa_border(self, seg_arr, t1_nib, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tumor_mask = (seg_arr > 0).astype(np.uint8)
        if not tumor_mask.any():
            return
        border = tumor_mask & (
            ~binary_erosion(tumor_mask, structure=np.ones((3, 3, 3)))
        )
        nib.save(
            nib.Nifti1Image(border.astype(np.uint8), t1_nib.affine, t1_nib.header),
            str(out_dir / "tumor_border_in_T1.nii.gz"),
        )

    def _compare_headers(self, seg_nib, t1_nib) -> Dict[str, Any]:
        def info(img):
            aff = img.affine
            vs = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
            return {
                "shape": tuple(int(s) for s in img.shape[:3]),
                "voxel_size_mm": [round(float(x), 3) for x in vs],
                "origin_mm": [round(float(x), 3) for x in aff[:3, 3]],
                "affine": aff.tolist(),
            }
        return {"segmentation": info(seg_nib), "t1_ref": info(t1_nib)}

    def _post_conditions(
        self, seg_out: Path, t1_ref: Path, seg_data: np.ndarray,
    ) -> Dict[str, Any]:
        out_nib = nib.load(str(seg_out))
        ref_nib = nib.load(str(t1_ref))
        shape_match = bool(out_nib.shape[:3] == ref_nib.shape[:3])
        affine_match = bool(
            np.allclose(out_nib.affine, ref_nib.affine, rtol=0, atol=1e-3)
        )
        unique_labels = sorted(int(x) for x in np.unique(seg_data))
        labels_ok = all(lbl in ALLOWED_LABELS for lbl in unique_labels)
        per_label_ok = all(
            int(np.sum(seg_data == lbl)) >= self.min_voxels_per_label
            for lbl in (1, 2, 3)
        )
        return {
            "shape_match": shape_match,
            "affine_close": affine_match,
            "unique_labels": unique_labels,
            "labels_in_allowed_set": labels_ok,
            "each_label_has_min_voxels": per_label_ok,
        }

    @staticmethod
    def _write_audit(path: Path, audit: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Align Oncohabitats Segmentation to native T1. Falls back "
                    "to dipy image-based registration when "
                    "Segmentation_oncohabitats_T1.nii.gz is present.",
    )
    p.add_argument("--patient-dir", required=True, type=Path)
    args = p.parse_args()
    MaskToT1().run(args.patient_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
