# deepneurobold/preprocessing/gm_wm_csf.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import os
import argparse

from deepneurobold.preprocessing.fslenv import run_in_fsl  # your helper

def _ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def derive_healthy_gm_wm_csf_pves(
    patient_dir: Path,
    *,
    t1_brain_rel: str = "PREPROCESSING/t1/t1_brain.nii.gz",
    brainmask_rel: str = "PREPROCESSING/t1/t1_brain_mask.nii.gz",
    seg_in_t1_rel: str = "PREPROCESSING/mask/Segmentation_in_T1.nii.gz",
    out_mask_dir_rel: str = "PREPROCESSING/mask",
    csf_thr: float = 0.70,
    do_clean: bool = True,
    restrict_to_healthy: bool = True,
) -> Dict[str, str]:
    """
    Build healthy mask (brain - tumor), run FSL FAST on T1 healthy brain,
    and export GM/WM/CSF partial volume estimates (PVEs) + a binary CSF mask.

    Steps
    -----
    1) tumor_mask_bin.nii.gz = binarize segmentation
    2) healthy_mask.nii.gz   = brainmask - tumor
    3) t1_brain_healthy.nii.gz = T1brain * healthy_mask
    4) FAST on healthy T1 (GM/WM/CSF PVEs)
    5) Copy PVEs to:
         - gm_healthy_pve.nii.gz
         - wm_healthy_pve.nii.gz
         - csf_healthy_pve.nii.gz
    6) CSF binary mask in T1:
         - csf_mask_inT1.nii.gz  (thresholded, cleaned, masked with brainmask
           and optionally with healthy_mask)

    Parameters
    ----------
    csf_thr : float
        Threshold on CSF PVE for binarization (0..1). Typical 0.6–0.8.
    do_clean : bool
        If True, apply -ero followed by -dilM to remove speckles and smooth edges.
    restrict_to_healthy : bool
        If True, final CSF mask is also multiplied by healthy_mask (i.e., excludes tumor region).

    Returns
    -------
    dict
        Paths of produced files (strings).
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
        "csf_pve":        str(mdir / "csf_healthy_pve.nii.gz"),
        "csf_mask":       str(mdir / "csf_mask_inT1.nii.gz"),
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
    csf_pve   = Path(outputs["csf_pve"])
    csf_mask  = Path(outputs["csf_mask"])

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

    # 5) copy PVEs (CSF=0, GM=1, WM=2)
    csf_src = mdir / "fast_healthy_pve_0.nii.gz"
    gm_src  = mdir / "fast_healthy_pve_1.nii.gz"
    wm_src  = mdir / "fast_healthy_pve_2.nii.gz"
    if not (_ok(csf_src) and _ok(gm_src) and _ok(wm_src)):
        raise RuntimeError("FAST did not produce expected PVE files.")
    os.replace(gm_src, gm_pve)
    os.replace(wm_src, wm_pve)
    os.replace(csf_src, csf_pve)

    # 6) CSF binary mask (thr→bin → optional clean → mask with brain/healthy)
    thr = max(0.0, min(1.0, float(csf_thr)))
    cmd = f'fslmaths "{csf_pve}" -thr {thr:.3f} -bin'
    if do_clean:
        cmd += " -ero -dilM"
    # limit to brain
    cmd += f' -mas "{brainmask}"'
    # optionally exclude tumor explicitly (restrict to healthy)
    if restrict_to_healthy:
        cmd += f' -mas "{healthy}"'
    cmd += f' "{csf_mask}"'
    res = run_in_fsl(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"CSF mask creation failed: {res.stderr}")

    # optional: clean FAST extras (keep logs if you prefer)
    for extra in mdir.glob("fast_healthy_*"):
        if extra.name not in {
            Path(gm_pve).name,
            Path(wm_pve).name,
            Path(csf_pve).name,
        }:
            try:
                extra.unlink()
            except OSError:
                pass

    return outputs


# -------------------- CLI --------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("deepneurobold.preprocessing.gmwm")
    ap.add_argument("--patient-dir", type=Path, required=True, help="Patient folder (e.g., Patient_01)")
    ap.add_argument("--csf-thr", type=float, default=0.70, help="Threshold for CSF PVE binarization (0..1)")
    ap.add_argument("--no-clean", action="store_true", help="Disable -ero / -dilM cleanup")
    ap.add_argument("--no-restrict-healthy", action="store_true", help="Do NOT multiply CSF mask by healthy_mask")
    return ap.parse_args()

def main():
    args = _parse_args()
    try:
        outs = derive_healthy_gm_wm_csf_pves(
            args.patient_dir,
            csf_thr=args.csf_thr,
            do_clean=not args.no_clean,
            restrict_to_healthy=not args.no_restrict_healthy,
        )
    except FileNotFoundError as e:
        # Do not crash whole array if this patient is incomplete
        print(f"[WARN] Skipping patient {args.patient_dir} in gm_wm_csf: {e}")
        return

    print("[OK] Generated:")
    for k, v in outs.items():
        print(f"  {k}: {v}")
        
        
if __name__ == "__main__":
    main()