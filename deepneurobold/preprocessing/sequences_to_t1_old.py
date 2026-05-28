#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import argparse

import numpy as np
import nibabel as nib
import ants


def prep_intensity(img: ants.ANTsImage) -> ants.ANTsImage:
    # Minimal + robust across ANTsPy versions
    return ants.iMath(img, "Normalize")


def find_modality_image(patient_dir: Path, modality: str) -> Optional[Path]:
    pre = patient_dir / "PREPROCESSING"
    t1_dir = pre / "t1"

    m = modality.lower()

    if m == "flair":
        candidates = [
            t1_dir / "FLAIR.nii.gz",
            t1_dir / "FLAIR.nii",
            t1_dir / "flair.nii.gz",
            t1_dir / "flair.nii",
            patient_dir / "FLAIR" / "flair.nii.gz",
            patient_dir / "FLAIR" / "flair.nii",
        ]
    elif m in ("t1ce", "t1_contrast", "t1c"):
        candidates = [
            t1_dir / "T1CE.nii.gz",
            t1_dir / "T1CE.nii",
            t1_dir / "t1ce.nii.gz",
            t1_dir / "t1ce.nii",
            patient_dir / "T1CE" / "t1ce.nii.gz",
            patient_dir / "T1CE" / "t1ce.nii",
        ]
    elif m == "t2":
        candidates = [
            t1_dir / "T2.nii.gz",
            t1_dir / "T2.nii",
            t1_dir / "t2.nii.gz",
            t1_dir / "t2.nii",
            patient_dir / "T2" / "t2.nii.gz",
            patient_dir / "T2" / "t2.nii",
        ]
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    for c in candidates:
        if c.is_file():
            return c

    return None


def build_mask_in_t1(t1_ants: ants.ANTsImage, t1_dir: Path) -> Optional[ants.ANTsImage]:
    t1_brain_mask_path = t1_dir / "t1_brain_mask.nii.gz"
    if t1_brain_mask_path.is_file():
        m = ants.image_read(str(t1_brain_mask_path))
        m = ants.threshold_image(m, 0.5, 1e9, 1, 0)
        return m

    t1_brain_path = t1_dir / "t1_brain.nii.gz"
    if t1_brain_path.is_file():
        t1_brain_ants = ants.image_read(str(t1_brain_path))
        return ants.get_mask(t1_brain_ants)

    try:
        return ants.get_mask(t1_ants)
    except Exception:
        return None


class SequencesToT1:
    """
    Rigid registration FLAIR / T1CE / T2 -> T1 (t1_reoriented.nii.gz).

    Outputs per modality (if found) under PREPROCESSING/t1/:
        <label>_inT1_ants.nii.gz
        <label>_inT1_ants_geo.nii.gz
        <label>_inT1_ants_geo_brain.nii.gz (if mask is available)
        <label>2t1_ants_transforms.txt
    """

    def __init__(
        self,
        threads: Optional[int] = None,
        interp: str = "linear",
        # legacy args (accepted for backward-compatibility)
        tform_type: Optional[str] = None,
        winsorize: Optional[str] = None,
        histmatch: Optional[bool] = None,
        ants_threads: Optional[int] = None,
        **_kwargs,
    ):
        if threads is None and ants_threads is not None:
            threads = ants_threads

        self.threads = threads
        self.interp = interp

        # legacy params ignored intentionally (Rigid only). kept to not break callers.
        self._legacy = {"tform_type": tform_type, "winsorize": winsorize, "histmatch": histmatch}

        if self.threads and self.threads > 0:
            try:
                ants.set_number_of_threads(int(self.threads))
            except Exception:
                pass

    def run(self, patient_dir: Path) -> Dict[str, Dict[str, str]]:
        pdir = Path(patient_dir).resolve()
        pre = pdir / "PREPROCESSING"
        t1_dir = pre / "t1"

        t1_ref = t1_dir / "t1_reoriented.nii.gz"
        if not t1_ref.is_file():
            raise FileNotFoundError(str(t1_ref))

        t1_ants = ants.image_read(str(t1_ref))
        t1_nii = nib.load(str(t1_ref))

        mask_img = build_mask_in_t1(t1_ants, t1_dir)

        outputs: Dict[str, Dict[str, str]] = {}

        for modality in ["flair", "t1ce", "t2"]:
            mov_path = find_modality_image(pdir, modality)
            if mov_path is None:
                continue

            out_dict = self._register_one(
                modality=modality,
                moving_path=mov_path,
                t1_ants=t1_ants,
                t1_nii=t1_nii,
                t1_dir=t1_dir,
                mask_img=mask_img,
            )
            outputs[modality] = out_dict

        return outputs

    def _register_one(
        self,
        *,
        modality: str,
        moving_path: Path,
        t1_ants: ants.ANTsImage,
        t1_nii: nib.Nifti1Image,
        t1_dir: Path,
        mask_img: Optional[ants.ANTsImage] = None,
    ) -> Dict[str, str]:
        label = modality.lower()
        moving_path = Path(moving_path)

        if not moving_path.is_file():
            raise FileNotFoundError(str(moving_path))

        moving_ants = ants.image_read(str(moving_path))

        fixed = prep_intensity(t1_ants)
        moving = prep_intensity(moving_ants)

        # Keep registration call minimal for compatibility across ANTsPy versions
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",
        )

        out: Dict[str, str] = {}

        warped = reg["warpedmovout"]
        warped_arr = warped.numpy().astype(np.float32)
        out_ants = t1_dir / f"{label}_inT1_ants.nii.gz"
        nii_warped = nib.Nifti1Image(warped_arr, t1_nii.affine, t1_nii.header)
        nii_warped.set_data_dtype(np.float32)
        nib.save(nii_warped, str(out_ants))
        out[f"{label}_ants"] = str(out_ants)

        fwd = reg.get("fwdtransforms", [])
        inv = reg.get("invtransforms", [])
        tf_txt = t1_dir / f"{label}2t1_ants_transforms.txt"
        with open(tf_txt, "w") as f:
            f.write("Forward transforms (apply in this order):\n")
            for t in fwd:
                f.write(f"{t}\n")
            f.write("Inverse transforms:\n")
            for t in inv:
                f.write(f"{t}\n")
        out[f"{label}_transforms"] = str(tf_txt)

        resamp = ants.apply_transforms(
            fixed=t1_ants,
            moving=moving_ants,
            transformlist=fwd,
            interpolator=self.interp,
        )
        resamp_arr = resamp.numpy().astype(np.float32)
        out_geo = t1_dir / f"{label}_inT1_ants_geo.nii.gz"
        nii_geo = nib.Nifti1Image(resamp_arr, t1_nii.affine, t1_nii.header)
        nii_geo.set_data_dtype(np.float32)
        nib.save(nii_geo, str(out_geo))
        out[f"{label}_ants_geo"] = str(out_geo)

        if mask_img is not None:
            masked = ants.mask_image(resamp, mask_img, 1)
            masked_arr = masked.numpy().astype(np.float32)
            out_masked = t1_dir / f"{label}_inT1_ants_geo_brain.nii.gz"
            nii_masked = nib.Nifti1Image(masked_arr, t1_nii.affine, t1_nii.header)
            nii_masked.set_data_dtype(np.float32)
            nib.save(nii_masked, str(out_masked))
            out[f"{label}_ants_geo_brain"] = str(out_masked)

        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-dir", required=True, type=str)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--interp", type=str, default="linear")
    args = parser.parse_args()

    if args.threads and args.threads > 0:
        try:
            ants.set_number_of_threads(int(args.threads))
        except Exception:
            pass

    seq = SequencesToT1(
        threads=args.threads,
        interp=args.interp,
    )
    seq.run(Path(args.patient_dir))


if __name__ == "__main__":
    main()