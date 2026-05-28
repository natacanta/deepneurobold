#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepneurobold.preprocessing.sequences_to_t1
=============================================
Register a tumour segmentation into native T1 space using the ANTs
transform of the modality on which the segmentation was originally drawn
(FLAIR, T1CE, or T2).

This module is the segmentation-aware counterpart of
:mod:`sequences_to_t1_old` (which contains the ``SequencesToT1`` class
used by the v2 pipeline for FLAIR/T1CE/T2 -> T1 registration).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ants
import nibabel as nib
import numpy as np


def find_transform_file(t1_dir: Path, modality: str) -> Path:
    """Return the path to the ANTs transform list for *modality*."""
    modality = modality.lower()
    candidates = [
        t1_dir / f"{modality}2t1_ants_transforms.txt",
        t1_dir / f"{modality}2t1_transforms.txt",
        t1_dir / f"{modality}_to_t1_transforms.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No transform found for modality {modality!r} in {t1_dir}\n"
        f"Tried: {[str(c) for c in candidates]}"
    )


def get_transform_files(transform_txt: Path) -> list[str]:
    """Parse an ANTs transform-list text file and return the forward paths."""
    transforms: list[str] = []
    with open(transform_txt, "r") as f:
        in_forward = False
        for line in f:
            line = line.strip()
            if line.startswith("Forward transforms"):
                in_forward = True
                continue
            if line.startswith("Inverse transforms"):
                break
            if in_forward and line and not line.startswith("#") and not line.startswith("Forward"):
                transform_path = Path(line)
                if transform_path.exists():
                    transforms.append(str(transform_path))
                else:
                    # Fall back to relative resolution alongside the text file.
                    abs_path = transform_txt.parent / transform_path.name
                    if abs_path.exists():
                        transforms.append(str(abs_path))
    return transforms


def find_modality_image(patient_dir: Path, modality: str) -> Path:
    """Locate the native-space NIfTI for the given modality."""
    modality = modality.lower()
    search_paths = [
        patient_dir / modality.upper(),
        patient_dir / modality,
        patient_dir / "SEGMENTATION",
    ]
    for base in search_paths:
        if not base.exists():
            continue
        for ext in (".nii.gz", ".nii"):
            candidates = [
                base / f"{modality}{ext}",
                base / f"{modality.upper()}{ext}",
                base / f"*{modality}*{ext}",
            ]
            for c in candidates:
                if "*" in str(c):
                    from glob import glob
                    for m in glob(str(c)):
                        p = Path(m)
                        if p.exists():
                            return p
                elif c.exists():
                    return c
    raise FileNotFoundError(
        f"No image found for modality {modality!r} in {patient_dir}"
    )


def register_segmentation(
    patient_dir: Path,
    modality: str,
    output_name: str = "Segmentation_in_T1.nii.gz",
    overwrite: bool = False,
) -> Path:
    """
    Register the segmentation into T1 space via the modality's transform.

    Parameters
    ----------
    patient_dir : Path
        Patient directory containing ``PREPROCESSING/``.
    modality : str
        Modality on which the segmentation was drawn (``flair``, ``t1ce``, ``t2``).
    output_name : str, default ``Segmentation_in_T1.nii.gz``
        File name of the resulting NIfTI in ``PREPROCESSING/mask/``.
    overwrite : bool, default False
        If True, regenerate even when the output exists.

    Returns
    -------
    Path
        Path to the registered segmentation NIfTI.
    """
    patient_dir = Path(patient_dir)
    pre = patient_dir / "PREPROCESSING"
    t1_dir = pre / "t1"
    mask_dir = pre / "mask"

    seg_src = mask_dir / "Segmentation.nii.gz"
    t1_ref = t1_dir / "t1_reoriented.nii.gz"
    seg_out = mask_dir / output_name

    qa_dir = mask_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    border_out = qa_dir / "tumor_border_in_T1.nii.gz"

    print(f"\n=== Registering segmentation for {patient_dir.name} ===")
    print(f"Source modality: {modality}")
    print(f"Source segmentation: {seg_src}")
    print(f"Reference T1: {t1_ref}")
    print(f"Output: {seg_out}")

    if not seg_src.exists():
        raise FileNotFoundError(f"Missing segmentation: {seg_src}")
    if not t1_ref.exists():
        raise FileNotFoundError(f"Missing reference T1: {t1_ref}")
    if seg_out.exists() and not overwrite:
        print(f"  SKIP: output already exists ({seg_out}). Use --overwrite to regenerate.")
        return seg_out

    try:
        transform_txt = find_transform_file(t1_dir, modality)
        print(f"Transform file: {transform_txt}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"Available modalities with transforms:")
        for m in ("flair", "t1ce", "t2"):
            try:
                tf = find_transform_file(t1_dir, m)
                print(f"  - {m}: {tf}")
            except FileNotFoundError:
                pass
        raise

    transforms = get_transform_files(transform_txt)
    if not transforms:
        raise RuntimeError(f"Could not read transforms from {transform_txt}")
    print(f"Transforms to apply: {transforms}")

    seg_img = ants.image_read(str(seg_src))
    t1_img = ants.image_read(str(t1_ref))

    print("Applying transforms to segmentation (nearest-neighbour)...")
    seg_registered = ants.apply_transforms(
        fixed=t1_img,
        moving=seg_img,
        transformlist=transforms,
        interpolator="nearestNeighbor",  # CRITICAL: preserve discrete labels
    )

    seg_arr = seg_registered.numpy()
    print(f"Unique values before rounding: {np.unique(seg_arr)[:20]}")
    seg_arr = np.rint(seg_arr).astype(np.uint8)
    print(f"Unique values after rounding: {np.unique(seg_arr)}")

    t1_nii = nib.load(str(t1_ref))
    seg_nii = nib.Nifti1Image(seg_arr, affine=t1_nii.affine, header=t1_nii.header)
    seg_nii.set_data_dtype(np.uint8)
    nib.save(seg_nii, str(seg_out))
    print(f"Wrote segmentation: {seg_out}")

    # QA: write a 1-voxel-thick tumour border for visual alignment check.
    from scipy.ndimage import binary_erosion

    tumor_mask = (seg_arr > 0).astype(bool)
    struct = np.ones((3, 3, 3))
    eroded = binary_erosion(tumor_mask, structure=struct)
    border = tumor_mask & (~eroded)
    border_nii = nib.Nifti1Image(border.astype(np.uint8), affine=t1_nii.affine, header=t1_nii.header)
    nib.save(border_nii, str(border_out))
    print(f"Wrote QA border: {border_out}")

    return seg_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a tumour segmentation into T1 space using a modality transform.",
    )
    parser.add_argument("--patient-dir", required=True, type=str)
    parser.add_argument(
        "--modality",
        required=True,
        choices=["flair", "t1ce", "t2"],
        help="Modality on which the segmentation was originally drawn.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    register_segmentation(
        patient_dir=Path(args.patient_dir),
        modality=args.modality,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
