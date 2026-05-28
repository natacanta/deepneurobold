#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
brats_t1_reg_sweep.py

Register:
  PREPROCESSING/mask/BRATS/t1_to_SRI_brain.nii.gz
to:
  PREPROCESSING/t1/t1_reoriented.nii.gz

Sweep ANTs settings and write outputs in:
  PREPROCESSING/mask/BRATS/
"""

from __future__ import annotations
from pathlib import Path
import argparse
import ants


def run_one_registration(
    t1_ref_img: "ants.ANTsImage",
    t1_sri_img: "ants.ANTsImage",
    out_path: Path,
    type_of_transform: str,
    normalize: bool,
) -> None:
    print()
    print("====================================================")
    print(f"[RUN] type_of_transform={type_of_transform}, normalize={normalize}")
    print(f"[RUN] out_path = {out_path}")

    if normalize:
        print("[RUN] Normalizing intensities...")
        fixed = ants.iMath(t1_ref_img, "Normalize")
        moving = ants.iMath(t1_sri_img, "Normalize")
    else:
        fixed = t1_ref_img
        moving = t1_sri_img

    print("[RUN] Running ants.registration...")
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=type_of_transform,
    )

    print("[RUN] Applying transform to t1_to_SRI_brain...")
    # IMPORTANT: use the SAME fixed used in registration (fixed), not t1_ref_img
    t1_sri_in_t1 = ants.apply_transforms(
        fixed=fixed,
        moving=t1_sri_img,
        transformlist=reg["fwdtransforms"],
        interpolator="linear",
    )

    print("[RUN] Writing output with ANTs (preserves spacing/origin/direction)...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(t1_sri_in_t1, str(out_path))
    print(f"[RUN] Saved: {out_path}")

    # Optional quick stats
    arr = t1_sri_in_t1.numpy()
    print(f"[RUN] Result shape={arr.shape} min={arr.min():.3f} max={arr.max():.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Sweep ANTs registrations between "
            "t1_to_SRI_brain.nii.gz and t1_reoriented.nii.gz "
            "for one Patient_* directory."
        )
    )
    ap.add_argument(
        "--patient-dir",
        required=True,
        type=str,
        help="Path to Patient_* directory (root, not PREPROCESSING/).",
    )
    args = ap.parse_args()

    patient_dir = Path(args.patient_dir)
    pre = patient_dir / "PREPROCESSING"

    t1_ref = pre / "t1" / "t1_reoriented.nii.gz"
    t1_sri = pre / "mask" / "BRATS" / "t1_to_SRI_brain.nii.gz"

    print(f"[INFO] patient_dir = {patient_dir}")
    print(f"[INFO] t1_ref = {t1_ref}")
    print(f"[INFO] t1_sri = {t1_sri}")

    for p in (t1_ref, t1_sri):
        print(f"[CHK] {p} exists={p.exists()}")
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print("[INFO] Reading images with ANTs...")
    t1_ref_img = ants.image_read(str(t1_ref))
    t1_sri_img = ants.image_read(str(t1_sri))

    print("[INFO] t1_ref_img shape:", t1_ref_img.shape)
    print("[INFO] t1_sri_img shape:", t1_sri_img.shape)

    out_dir = pre / "mask" / "BRATS"
    print(f"[INFO] Outputs will be stored in: {out_dir}")

    # 1) Rigid + Normalize
    run_one_registration(
        t1_ref_img=t1_ref_img,
        t1_sri_img=t1_sri_img,
        out_path=out_dir / "t1_SRI_in_T1_rigid_norm.nii.gz",
        type_of_transform="Rigid",
        normalize=True,
    )

    # 2) Affine + Normalize
    run_one_registration(
        t1_ref_img=t1_ref_img,
        t1_sri_img=t1_sri_img,
        out_path=out_dir / "t1_SRI_in_T1_affine_norm.nii.gz",
        type_of_transform="Affine",
        normalize=True,
    )

    # 3) SyN + Normalize
    run_one_registration(
        t1_ref_img=t1_ref_img,
        t1_sri_img=t1_sri_img,
        out_path=out_dir / "t1_SRI_in_T1_syn_norm.nii.gz",
        type_of_transform="SyN",
        normalize=True,
    )

    # 4) Affine + RAW
    run_one_registration(
        t1_ref_img=t1_ref_img,
        t1_sri_img=t1_sri_img,
        out_path=out_dir / "t1_SRI_in_T1_affine_raw.nii.gz",
        type_of_transform="Affine",
        normalize=False,
    )

    print()
    print("[DONE] Registration sweep finished.")


if __name__ == "__main__":
    main()