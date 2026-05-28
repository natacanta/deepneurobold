#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepneurobold.preprocessing.mask_to_T1_old
============================================
Legacy ANTs-based segmentation→T1 aligner (kept for backwards compatibility
with the original deepneurobold pipeline). Uses a rigid registration of the
segmentation onto the patient's reoriented T1, applied with nearest-neighbour
interpolation to preserve labels.

The production v2 pipeline uses :mod:`deepneurobold.preprocessing.mask_to_T1`
(coordinate-based resampling, no registration) which is more robust when the
segmentation file already shares the target T1's physical space — the typical
Oncohabitats output. This ANTs-based variant is retained only for cases that
explicitly require registration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ants
import nibabel as nib
import numpy as np


class MaskToT1:
    """
    Align an Oncohabitats segmentation to the native T1 via ANTs rigid
    registration (6 DOF, Mutual-Information cost). Labels preserved via
    nearest-neighbour interpolation.
    """

    def __init__(self, interp: str = "nearestNeighbor") -> None:
        self.interp = interp

    # ------------------------------------------------------------------
    # Oncohabitats segmentation -> T1 (rigid registration)
    # ------------------------------------------------------------------
    def _onco_to_t1_rigid(self, patient_dir: Path):
        patient_dir = Path(patient_dir)
        pre = patient_dir / "PREPROCESSING"
        seg_src = pre / "mask" / "Segmentation.nii.gz"
        seg_out = pre / "mask" / "Segmentation_in_T1.nii.gz"
        t1_ref = pre / "t1" / "t1_reoriented.nii.gz"

        # Persist the transform for debugging.
        transform_dir = pre / "mask" / "transforms"
        transform_dir.mkdir(parents=True, exist_ok=True)
        rigid_mat = transform_dir / "seg_to_t1_rigid.mat"

        print(f"\n[mask_to_T1][ONCO-RIGID] start: {patient_dir.name}")
        print(f"[mask_to_T1][ONCO-RIGID] seg_src = {seg_src}")
        print(f"[mask_to_T1][ONCO-RIGID] seg_out = {seg_out}")
        print(f"[mask_to_T1][ONCO-RIGID] t1_ref  = {t1_ref}")

        # Skip if a valid registration already exists.
        if seg_out.exists() and rigid_mat.exists():
            seg_nii = nib.load(str(seg_out))
            t1_nii = nib.load(str(t1_ref))
            print(
                f"[mask_to_T1][ONCO-RIGID] existing Segmentation_in_T1 shape={seg_nii.shape}, "
                f"T1 shape={t1_nii.shape}"
            )
            if rigid_mat.stat().st_size > 100:
                print("[mask_to_T1][ONCO-RIGID] SKIP: already registered")
                return str(seg_out)
            print("[mask_to_T1][ONCO-RIGID] REBUILD: stored transform looks corrupt")

        if not seg_src.exists():
            print(f"[mask_to_T1][ONCO-RIGID] ERROR: missing {seg_src}")
            return None
        if not t1_ref.exists():
            print(f"[mask_to_T1][ONCO-RIGID] ERROR: missing {t1_ref}")
            return None

        print("[mask_to_T1][ONCO-RIGID] Reading source segmentation and T1 with ANTs...")
        seg_img = ants.image_read(str(seg_src))
        t1_img = ants.image_read(str(t1_ref))
        print(
            f"[mask_to_T1][ONCO-RIGID] seg_src ANTs shape={seg_img.shape}, "
            f"dtype={seg_img.numpy().dtype}"
        )
        print(
            f"[mask_to_T1][ONCO-RIGID] t1_ref ANTs shape={t1_img.shape}, "
            f"dtype={t1_img.numpy().dtype}"
        )

        print("[mask_to_T1][ONCO-RIGID] Running RIGID registration Segmentation -> T1...")
        # Normalise intensities to improve registration robustness.
        seg_norm = ants.iMath(seg_img, "Normalize")
        t1_norm = ants.iMath(t1_img, "Normalize")

        reg = ants.registration(
            fixed=t1_norm,
            moving=seg_norm,
            type_of_transform="Rigid",   # 6 DOF (translation + rotation)
            aff_metric="mattes",         # Mattes Mutual Information
            syn_metric="mattes",
            reg_iterations=(100, 50, 10),
            verbose=True,
        )

        # Persist the forward transform.
        if reg.get("fwdtransforms"):
            import shutil
            shutil.copy(reg["fwdtransforms"][0], str(rigid_mat))
            print(f"[mask_to_T1][ONCO-RIGID] Transform stored at: {rigid_mat}")

        print("[mask_to_T1][ONCO-RIGID] Applying rigid transform to segmentation...")
        seg_registered = ants.apply_transforms(
            fixed=t1_img,
            moving=seg_img,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )

        # Round to integer labels and cast to uint8.
        seg_arr = seg_registered.numpy()
        print(f"[mask_to_T1][ONCO-RIGID] Before rounding: min={seg_arr.min()}, max={seg_arr.max()}")
        seg_arr = np.rint(seg_arr).astype(np.uint8)
        print(f"[mask_to_T1][ONCO-RIGID] After rounding: unique values={np.unique(seg_arr)}")

        # Save with the T1's affine + header.
        t1_nii = nib.load(str(t1_ref))
        seg_nii = nib.Nifti1Image(seg_arr, affine=t1_nii.affine, header=t1_nii.header)
        seg_nii.set_data_dtype(np.uint8)
        nib.save(seg_nii, str(seg_out))
        print(f"[mask_to_T1][ONCO-RIGID] OK: wrote {seg_out} (shape={seg_arr.shape})")

        # QA overlay: write a tumour contour image for visual verification.
        overlay_dir = pre / "mask" / "qa"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        from scipy.ndimage import binary_erosion

        tumor_mask = (seg_arr > 0).astype(np.uint8)
        struct = np.ones((3, 3, 3))
        eroded = binary_erosion(tumor_mask, structure=struct)
        border = tumor_mask & (~eroded)
        border_nii = nib.Nifti1Image(border.astype(np.uint8), affine=t1_nii.affine, header=t1_nii.header)
        nib.save(border_nii, str(overlay_dir / "tumor_border_in_T1.nii.gz"))
        print(
            f"[mask_to_T1][ONCO-RIGID] Border saved at: "
            f"{overlay_dir}/tumor_border_in_T1.nii.gz  (use to verify alignment in ITK-SNAP)"
        )

        return str(seg_out)

    # ------------------------------------------------------------------
    # BraTS segmentation -> T1
    # ------------------------------------------------------------------
    def _brats_to_t1(self, patient_dir: Path):
        pre = patient_dir / "PREPROCESSING"
        t1_ref = pre / "t1" / "t1_reoriented.nii.gz"
        t1_sri = pre / "mask" / "BRATS" / "t1_to_SRI_brain.nii.gz"
        seg_in = pre / "mask" / "BRATS" / "Segmentation_BRATS.nii.gz"
        seg_out = pre / "mask" / "BRATS" / "Segmentation_BRATS_in_T1.nii.gz"
        t1_sri_out = pre / "mask" / "BRATS" / "t1_SRI_in_T1.nii.gz"

        print(f"[mask_to_T1][BRATS] start: {patient_dir.name}")
        print(f"[mask_to_T1][BRATS] t1_ref  = {t1_ref}")
        print(f"[mask_to_T1][BRATS] t1_sri  = {t1_sri}")
        print(f"[mask_to_T1][BRATS] seg_in  = {seg_in}")
        print(f"[mask_to_T1][BRATS] seg_out = {seg_out}")

        if not t1_ref.exists() or not t1_sri.exists() or not seg_in.exists():
            print(
                "[mask_to_T1][BRATS] WARN: missing one of t1_ref / t1_sri / seg_in -> skipping"
            )
            return None

        t1_ref_img = ants.image_read(str(t1_ref))
        t1_sri_img = ants.image_read(str(t1_sri))
        seg_brats = ants.image_read(str(seg_in))

        t1_ref_norm = ants.iMath(t1_ref_img, "Normalize")
        t1_sri_norm = ants.iMath(t1_sri_img, "Normalize")

        print("[mask_to_T1][BRATS] Running RIGID registration T1_SRI -> T1...")
        reg = ants.registration(
            fixed=t1_ref_norm,
            moving=t1_sri_norm,
            type_of_transform="Rigid",
        )

        seg_in_t1 = ants.apply_transforms(
            fixed=t1_ref_img,
            moving=seg_brats,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )
        seg_arr = np.rint(seg_in_t1.numpy()).astype(np.uint8)

        # QA: also map the BraTS atlas T1 into native T1 space.
        t1_sri_in_t1 = ants.apply_transforms(
            fixed=t1_ref_img,
            moving=t1_sri_img,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",
        )

        t1_nii = nib.load(str(t1_ref))

        seg_nii = nib.Nifti1Image(seg_arr, affine=t1_nii.affine, header=t1_nii.header)
        seg_nii.set_data_dtype(np.uint8)
        seg_out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(seg_nii, str(seg_out))
        print(f"[mask_to_T1][BRATS] OK: wrote {seg_out} (shape={seg_arr.shape})")

        t1_sri_nii = nib.Nifti1Image(
            t1_sri_in_t1.numpy().astype(np.float32),
            affine=t1_nii.affine,
            header=t1_nii.header,
        )
        nib.save(t1_sri_nii, str(t1_sri_out))
        print(f"[mask_to_T1][BRATS] OK: wrote {t1_sri_out}")

        return str(seg_out)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, patient_dir: Path, with_brats: bool = False) -> None:
        """Run Oncohabitats rigid registration; optionally also do BraTS."""
        patient_dir = Path(patient_dir)
        self._onco_to_t1_rigid(patient_dir)
        if with_brats:
            self._brats_to_t1(patient_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MaskToT1 with ANTs rigid registration (legacy variant).",
    )
    ap.add_argument("--patient-dir", required=True, type=str, help="Path to Patient_* dir")
    ap.add_argument(
        "--interp",
        type=str,
        default="nearestNeighbor",
        help="Interpolation for the transform application (default nearestNeighbor).",
    )
    ap.add_argument(
        "--with-brats",
        action="store_true",
        help="Also process BraTS segmentation if present.",
    )
    args = ap.parse_args()
    MaskToT1(interp=args.interp).run(Path(args.patient_dir), with_brats=args.with_brats)


if __name__ == "__main__":
    main()
