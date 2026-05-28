# deepneurobold/preprocessing/gmwm.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Optional
import os

from deepneurobold.preprocessing.fslenv import run_in_fsl  # your helper

def _ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def derive_healthy_gm_wm_pves(patient_dir: Path,
                              *,
                              t1_brain_rel: str = "PREPROCESSING/t1/t1_brain.nii.gz",
                              brainmask_rel: str = "PREPROCESSING/t1/t1_brain_mask.nii.gz",
                              seg_in_t1_rel: str = "PREPROCESSING/mask/Segmentation_in_T1.nii.gz",
                              out_mask_dir_rel: str = "PREPROCESSING/mask") -> dict:
    """
    Reproduce the bash:
      1) tumor_mask_bin.nii.gz = binarize segmentation
      2) healthy_mask.nii.gz   = brainmask - tumor
      3) t1_brain_healthy.nii.gz = T1brain * healthy_mask
      4) FAST on healthy T1 (GM/WM PVEs)
      5) Copy PVEs to gm_healthy_pve.nii.gz / wm_healthy_pve.nii.gz
    Returns dict with output paths (strings).
    """
    pdir = Path(patient_dir)
    t1b = pdir / t1_brain_rel
    brainmask = pdir / brainmask_rel
    seg = pdir / seg_in_t1_rel
    mdir = pdir / out_mask_dir_rel
    mdir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "tumor_mask_bin": str(mdir / "tumor_mask_bin.nii.gz"),
        "healthy_mask":   str(mdir / "healthy_mask.nii.gz"),
        "t1_brain_healthy": str(mdir / "t1_brain_healthy.nii.gz"),
        "gm_pve":         str(mdir / "gm_healthy_pve.nii.gz"),
        "wm_pve":         str(mdir / "wm_healthy_pve.nii.gz"),
    }

    # sanity
    missing = []
    if not _ok(t1b):       missing.append(str(t1b))
    if not _ok(brainmask): missing.append(str(brainmask))
    if not _ok(seg):       missing.append(str(seg))
    if missing:
        raise FileNotFoundError("Missing inputs:\n  " + "\n  ".join(missing))

    tumor_bin = Path(outputs["tumor_mask_bin"])
    healthy   = Path(outputs["healthy_mask"])
    t1h       = Path(outputs["t1_brain_healthy"])
    gm_pve    = Path(outputs["gm_pve"])
    wm_pve    = Path(outputs["wm_pve"])

    # 1) tumor mask bin
    res = run_in_fsl(f'fslmaths "{seg}" -thr 0.5 -bin "{tumor_bin}"')
    if res.returncode != 0:
        raise RuntimeError(f"fslmaths (tumor bin) failed: {res.stderr}")

    # 2) healthy = brainmask - tumor
    res = run_in_fsl(f'fslmaths "{brainmask}" -sub "{tumor_bin}" -thr 0 -bin "{healthy}"')
    if res.returncode != 0:
        raise RuntimeError(f"fslmaths (healthy) failed: {res.stderr}")

    # 3) apply healthy mask to T1 brain
    res = run_in_fsl(f'fslmaths "{t1b}" -mas "{healthy}" "{t1h}"')
    if res.returncode != 0:
        raise RuntimeError(f"fslmaths (t1_brain_healthy) failed: {res.stderr}")

    # 4) FAST (type=1 = T1)
    #    Outputs: fast_healthy_pve_0/1/2.nii.gz (CSF/GM/WM for 3 classes)
    prefix = mdir / "fast_healthy"
    res = run_in_fsl(
        f'fast -o "{prefix}" -n 3 -H 0.1 -I 4 -l 20.0 -g --type=1 "{t1h}"'
    )
    if res.returncode != 0:
        raise RuntimeError(f"FAST failed: {res.stderr}")

    # 5) copy PVEs (GM=1, WM=2)
    gm_src = mdir / "fast_healthy_pve_1.nii.gz"
    wm_src = mdir / "fast_healthy_pve_2.nii.gz"
    if not _ok(gm_src) or not _ok(wm_src):
        raise RuntimeError("FAST did not produce expected PVE files.")
    os.replace(gm_src, gm_pve)
    os.replace(wm_src, wm_pve)

    # optional: clean FAST extras
    for extra in mdir.glob("fast_healthy_*"):
        if extra.name not in {Path(gm_pve).name, Path(wm_pve).name}:
            try:
                extra.unlink()
            except OSError:
                pass

    return outputs