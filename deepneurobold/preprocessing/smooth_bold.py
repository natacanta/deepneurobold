#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import nibabel as nib


# ---------------------------- basic utils ----------------------------

def _echo(msg: str) -> None:
    print(msg, flush=True)


def _assert_img(p: Path, tag: str) -> None:
    if (not p.exists()) or p.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty {tag}: {p}")


def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def _run(cmd: List[str], env: Optional[dict] = None, step: str = "") -> None:
    if step:
        _echo(f"[RUN] {step}: {' '.join(cmd)}")
    else:
        _echo(f"[RUN] {' '.join(cmd)}")

    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        if p.stdout.strip():
            _echo(f"[STDOUT]\n{p.stdout.strip()}")
        if p.stderr.strip():
            _echo(f"[STDERR]\n{p.stderr.strip()}")
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)


def _tmpdir() -> Path:
    """
    Uses DNB_SMOOTH_TMPDIR if set; otherwise falls back to a safe local tmp.
    """
    d = os.environ.get("DNB_SMOOTH_TMPDIR", "").strip()
    if d:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path("/tmp") / f"deepneurobold_smooth_{os.getpid()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_move(src: Path, dst: Path) -> None:
    """
    Atomic move on same filesystem. If dst exists, replace it.
    If src and dst are on different filesystems, fall back to copy+replace.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        src.replace(dst)  # atomic rename if same FS
    except OSError:
        import shutil
        shutil.copy2(str(src), str(dst))
        try:
            src.unlink()
        except Exception:
            pass


def _make_tmp_out(dst: Path, suffix: str = ".tmp") -> Path:
    """
    Create a tmp output path in the tmpdir, then move atomically to dst.
    NOTE: Not used for fslmaths outputs (FSL needs a NIfTI extension at the end).
    """
    dst = Path(dst)
    tmp_name = f"{dst.name}{suffix}.{os.getpid()}"
    return _tmpdir() / tmp_name


def _save_nifti_like(ref_img: nib.Nifti1Image, data3d: np.ndarray, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if data3d.ndim != 3:
        raise ValueError(f"Expected 3D data to save NIfTI, got shape={data3d.shape}")

    out = nib.Nifti1Image(
        data3d.astype(np.float32, copy=False),
        affine=ref_img.affine,
        header=ref_img.header,
    )
    out.set_data_dtype(np.float32)
    nib.save(out, str(out_path))
    _assert_img(out_path, f"nifti:{out_path.name}")


def _resolve_fslmaths(env: Optional[dict]) -> str:
    """
    Return an executable for fslmaths.
    - Prefer absolute path from env if present.
    - Otherwise rely on PATH.
    """
    if env is None:
        env = os.environ

    fsldir = (env.get("FSLDIR") or "").strip()
    if fsldir:
        cand = Path(fsldir) / "bin" / "fslmaths"
        if cand.exists():
            return str(cand)

    w = _which("fslmaths")
    if w is None:
        raise RuntimeError("fslmaths not found. Ensure your wrapper is in PATH or set FSLDIR in env.")
    return w


def _tmp_sibling_keep_ext(dst: Path, tag: str) -> Path:
    """
    Create a temporary output path next to dst, preserving the NIfTI extension
    so FSL does NOT append a second extension (common cause of missing tmp files).
    """
    dst = Path(dst)
    pid = os.getpid()
    name = dst.name

    if name.endswith(".nii.gz"):
        base = name[:-7]
        tmp_name = f"{base}.{tag}.tmp.{pid}.nii.gz"
    elif name.endswith(".nii"):
        base = name[:-4]
        tmp_name = f"{base}.{tag}.tmp.{pid}.nii"
    else:
        tmp_name = f"{name}.{tag}.tmp.{pid}.nii.gz"

    return dst.parent / tmp_name


def _glob_alternatives(p: Path) -> List[Path]:
    """
    If a tool appended an extension unexpectedly, discover it.
    """
    p = Path(p)
    return sorted(p.parent.glob(p.name + "*"))


# ---------------------------- smoothing (NO FFT) ----------------------------

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * float(sigma * sigma))).astype(np.float32)
    k /= float(np.sum(k))
    return k


def _temporal_smooth_direct_block_reflect(X: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Temporal smoothing by direct convolution in time domain (NO FFT).
    Reflect padding reduces edge bias.
    X: (N, T)
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (N,T), got shape={X.shape}")

    N, T = X.shape
    K = int(k.shape[0])
    pad = K // 2
    if pad <= 0:
        return X.astype(np.float32, copy=False)

    Xp = np.pad(X, ((0, 0), (pad, pad)), mode="reflect")
    Y = np.empty((N, T), dtype=np.float32)
    kf = k.astype(np.float32, copy=False)

    # Intentionally direct convolution (no FFT)
    for i in range(N):
        Y[i, :] = np.convolve(Xp[i, :], kf, mode="valid").astype(np.float32, copy=False)

    return Y


def spatial_smooth_fsl(
    in_nii: Path,
    out_nii: Path,
    *,
    fwhm_mm: float = 2.5,
    env: Optional[dict] = None,
) -> Path:
    """
    Spatial Gaussian smoothing using FSL fslmaths.

    Critical correctness:
      - The temporary output MUST end in .nii.gz (or .nii). Otherwise FSL appends
        the outputtype extension and your code will look for the wrong tmp name.
      - We write the tmp in the SAME output folder then rename (atomic within FS).
    """
    in_nii = Path(in_nii)
    out_nii = Path(out_nii)
    _assert_img(in_nii, "input BOLD for spatial smoothing")

    sigma_mm = float(fwhm_mm) / 2.3548
    out_nii.parent.mkdir(parents=True, exist_ok=True)

    fslmaths = _resolve_fslmaths(env)

    _echo(f"[SMOOTH] Spatial: FWHM={fwhm_mm:.3f} mm -> sigma={sigma_mm:.3f} mm")
    _echo(f"[SMOOTH] Using fslmaths={fslmaths}")

    tmp_out = _tmp_sibling_keep_ext(out_nii, "spatial")

    # Clean any leftovers (including if an extension got appended previously)
    for q in _glob_alternatives(tmp_out):
        try:
            q.unlink()
        except Exception:
            pass

    try:
        _run([fslmaths, str(in_nii), "-s", f"{sigma_mm:.6f}", str(tmp_out)], env=env, step="FSL spatial smoothing")

        if (not tmp_out.exists()) or tmp_out.stat().st_size == 0:
            alts = _glob_alternatives(tmp_out)
            raise FileNotFoundError(
                f"Missing/empty spatial_smoothed_tmp: {tmp_out}. "
                f"Alternatives found: {[str(a) for a in alts]}"
            )

        _atomic_move(tmp_out, out_nii)
        _assert_img(out_nii, "spatial_smoothed")
    finally:
        for q in _glob_alternatives(tmp_out):
            if q.exists():
                try:
                    q.unlink()
                except Exception:
                    pass

    return out_nii


def temporal_smooth_gaussian_nofft(
    in_nii: Path,
    out_nii: Path,
    *,
    tr_sec: float,
    sigma_sec: float = 3.0,
    chunk_vox: int = 200_000,
    overwrite: bool = False,
) -> Path:
    """
    Temporal Gaussian smoothing along time axis (NO FFT).
    Blockwise over Z.

    Output safety:
      - Writes final NIfTI to a temp file in the SAME output folder and renames it.
      - Intermediate memmap stays in scratch (DNB_SMOOTH_TMPDIR).
    """
    in_nii = Path(in_nii)
    out_nii = Path(out_nii)
    _assert_img(in_nii, "input BOLD for temporal smoothing")

    if out_nii.exists() and (not overwrite):
        _echo(f"[SMOOTH] Temporal SKIP: exists -> {out_nii.name}")
        return out_nii

    img = nib.load(str(in_nii))
    dataobj = img.dataobj

    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape={img.shape} at {in_nii}")

    X, Y, Z, T = map(int, img.shape)
    tr_sec = float(tr_sec)
    sigma_sec = float(sigma_sec)

    sigma_tp = sigma_sec / tr_sec
    if sigma_tp <= 0:
        raise ValueError("sigma_tp must be > 0")

    radius = int(np.ceil(3.0 * sigma_tp))
    k = _gaussian_kernel_1d(sigma=float(sigma_tp), radius=radius)

    _echo(
        f"[SMOOTH] Temporal (NO FFT): TR={tr_sec:.3f}s sigma={sigma_sec:.3f}s "
        f"-> sigma_tp={sigma_tp:.3f} (radius={radius}, K={k.size})"
    )

    tmp_raw = _tmpdir() / (out_nii.name + f".temporal.memmap.{os.getpid()}.raw")
    out_mm = np.memmap(str(tmp_raw), dtype=np.float32, mode="w+", shape=(X, Y, Z, T))

    vox_per_slice = X * Y
    z_per_block = max(1, int(chunk_vox // max(1, vox_per_slice)))

    for z0 in range(0, Z, z_per_block):
        z1 = min(Z, z0 + z_per_block)

        blk = np.asanyarray(dataobj[:, :, z0:z1, :], dtype=np.float32)
        N = int(blk.shape[0] * blk.shape[1] * blk.shape[2])
        blk2 = blk.reshape((N, T))

        blk2_sm = _temporal_smooth_direct_block_reflect(blk2, k)
        out_mm[:, :, z0:z1, :] = blk2_sm.reshape((X, Y, z1 - z0, T))

        _echo(f"[SMOOTH] Temporal block z={z0}:{z1} done")

    out_mm.flush()

    tmp_final = _tmp_sibling_keep_ext(out_nii, "temporal")

    # Clean leftovers
    for q in _glob_alternatives(tmp_final):
        try:
            q.unlink()
        except Exception:
            pass

    try:
        out_img = nib.Nifti1Image(np.asarray(out_mm), affine=img.affine, header=img.header)
        out_img.set_data_dtype(np.float32)
        nib.save(out_img, str(tmp_final))

        if (not tmp_final.exists()) or tmp_final.stat().st_size == 0:
            alts = _glob_alternatives(tmp_final)
            raise FileNotFoundError(
                f"Missing/empty temporal_smoothed_tmp: {tmp_final}. "
                f"Alternatives found: {[str(a) for a in alts]}"
            )

        _atomic_move(tmp_final, out_nii)
        _assert_img(out_nii, "temporal_smoothed")
    finally:
        try:
            del out_mm
        except Exception:
            pass
        try:
            if tmp_raw.exists():
                tmp_raw.unlink()
        except Exception:
            pass
        for q in _glob_alternatives(tmp_final):
            if q.exists():
                try:
                    q.unlink()
                except Exception:
                    pass

    return out_nii


# ---------------------------- FFT stage (BLOCKWISE; avoids OOM) ----------------------------

def _safe_mask_from_paths(mask_path: Optional[Path], shape3: Tuple[int, int, int]) -> np.ndarray:
    if mask_path is None:
        return np.ones(shape3, dtype=bool)
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    m = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    if m.shape != shape3:
        raise ValueError(f"Mask shape {m.shape} != BOLD spatial shape {shape3}")
    return m > 0.5


def run_fft_stage_to_niftis_blockwise(
    *,
    in_bold_4d: Path,
    tr_sec: float,
    mask_3d: Optional[Path],
    out_bold_dir: Path,
    overwrite: bool = False,
    f_low_hz: float = 0.01,
    f_high_hz: float = 0.10,
    z_block: int = 4,
) -> Dict[str, str]:
    """
    Computes low-frequency FFT features from in_bold_4d (4D) blockwise (over Z) and writes 5x 3D NIfTI maps.
    """
    in_bold_4d = Path(in_bold_4d)
    out_bold_dir = Path(out_bold_dir)
    out_bold_dir.mkdir(parents=True, exist_ok=True)

    outs = {
        "fft_power_low": out_bold_dir / "FFT_power_low_inT1.nii.gz",
        "fft_power_ratio_low": out_bold_dir / "FFT_power_ratio_low_inT1.nii.gz",
        "fft_peak_freq_low": out_bold_dir / "FFT_peak_freq_low_inT1.nii.gz",
        "fft_spectral_entropy_low": out_bold_dir / "FFT_spectral_entropy_low_inT1.nii.gz",
        "fft_spectral_slope_low": out_bold_dir / "FFT_spectral_slope_low_inT1.nii.gz",
    }

    if (not overwrite) and all(p.exists() and p.stat().st_size > 0 for p in outs.values()):
        _echo(f"[FFT] SKIP: all FFT NIfTI maps exist in {out_bold_dir}")
        return {k: str(v) for k, v in outs.items()}

    _echo(f"[FFT] Load header/proxy: {in_bold_4d}")
    ref_img = nib.load(str(in_bold_4d))
    dataobj = ref_img.dataobj

    if len(ref_img.shape) != 4:
        raise ValueError(f"Expected 4D input for FFT stage, got {ref_img.shape}")

    X, Y, Z, T = map(int, ref_img.shape)
    tr_sec = float(tr_sec)

    freqs = np.fft.rfftfreq(T, d=tr_sec)
    band = (freqs >= float(f_low_hz)) & (freqs <= float(f_high_hz))
    if not np.any(band):
        raise ValueError("Low-frequency band is empty. Check f_low_hz/f_high_hz/TR/T.")
    band_freqs = freqs[band].astype(np.float32)
    B = int(band_freqs.size)

    mask3 = _safe_mask_from_paths(mask_3d, (X, Y, Z)).astype(bool)

    power_low_3d = np.zeros((X, Y, Z), dtype=np.float32)
    power_ratio_3d = np.zeros((X, Y, Z), dtype=np.float32)
    peak_freq_3d = np.zeros((X, Y, Z), dtype=np.float32)
    entropy_3d = np.zeros((X, Y, Z), dtype=np.float32)
    slope_3d = np.zeros((X, Y, Z), dtype=np.float32)

    xf = np.log(band_freqs + 1e-12).astype(np.float32)
    x0 = float(xf.mean())
    xd = (xf - x0).astype(np.float32)
    denom = float(np.sum(xd * xd) + 1e-12)

    zb = max(1, int(z_block))

    for z0 in range(0, Z, zb):
        z1 = min(Z, z0 + zb)

        mblk = mask3[:, :, z0:z1]
        if not np.any(mblk):
            continue

        blk = np.asanyarray(dataobj[:, :, z0:z1, :], dtype=np.float32)
        zb_eff = int(z1 - z0)

        V = blk.reshape((-1, T))
        mflat = mblk.reshape((-1,))
        idx = np.where(mflat)[0]
        if idx.size == 0:
            continue

        V = V[idx, :]
        V = V - V.mean(axis=1, keepdims=True)

        S = np.fft.rfft(V, axis=1)
        P = (S.real * S.real + S.imag * S.imag).astype(np.float32)

        P_band = P[:, band]
        P_total = P[:, 1:].sum(axis=1) + 1e-12

        power_low = P_band.sum(axis=1)
        power_ratio = power_low / P_total

        peak_idx = np.argmax(P_band, axis=1)
        peak_freq = band_freqs[peak_idx]

        Pb = P_band + 1e-12
        Pb = Pb / Pb.sum(axis=1, keepdims=True)
        entropy = (-np.sum(Pb * np.log(Pb), axis=1) / np.log(float(B))).astype(np.float32)

        yf = np.log(P_band + 1e-12).astype(np.float32)
        y0 = yf.mean(axis=1, keepdims=True)
        yd = yf - y0
        slope = (yd @ xd / denom).astype(np.float32)

        tmp = np.zeros((X * Y * zb_eff,), dtype=np.float32)

        tmp[idx] = power_low
        power_low_3d[:, :, z0:z1] = tmp.reshape((X, Y, zb_eff))

        tmp[idx] = power_ratio
        power_ratio_3d[:, :, z0:z1] = tmp.reshape((X, Y, zb_eff))

        tmp[idx] = peak_freq
        peak_freq_3d[:, :, z0:z1] = tmp.reshape((X, Y, zb_eff))

        tmp[idx] = entropy
        entropy_3d[:, :, z0:z1] = tmp.reshape((X, Y, zb_eff))

        tmp[idx] = slope
        slope_3d[:, :, z0:z1] = tmp.reshape((X, Y, zb_eff))

        _echo(f"[FFT] z-block {z0}:{z1} done (Nvox={idx.size})")

    _echo("[FFT] Save NIfTI maps...")
    _save_nifti_like(ref_img, power_low_3d, outs["fft_power_low"])
    _save_nifti_like(ref_img, power_ratio_3d, outs["fft_power_ratio_low"])
    _save_nifti_like(ref_img, peak_freq_3d, outs["fft_peak_freq_low"])
    _save_nifti_like(ref_img, entropy_3d, outs["fft_spectral_entropy_low"])
    _save_nifti_like(ref_img, slope_3d, outs["fft_spectral_slope_low"])

    return {k: str(v) for k, v in outs.items()}


# ---------------------------- MAIN PIPELINE API ----------------------------

def run_all(
    *,
    patient_dir: Path,
    in_bold_4d: Path,
    tr_sec: float,
    fwhm_mm: float = 2.5,
    sigma_sec: float = 3.0,
    fsl_env: Optional[dict] = None,
    overwrite: bool = False,
    run_fft: bool = False,
    fft_mask_3d: Optional[Path] = None,
    fft_low_hz: float = 0.01,
    fft_high_hz: float = 0.10,
    fft_z_block: int = 4,
) -> Dict[str, str]:
    patient_dir = Path(patient_dir)
    in_bold_4d = Path(in_bold_4d)

    bold_dir = patient_dir / "PREPROCESSING" / "bold"
    bold_dir.mkdir(parents=True, exist_ok=True)

    out_spatial = bold_dir / "BOLD_brain_inT1_spatial_smooth.nii.gz"
    out_spatiotemporal = bold_dir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz"

    if (not out_spatial.exists()) or overwrite:
        spatial_smooth_fsl(in_bold_4d, out_spatial, fwhm_mm=fwhm_mm, env=fsl_env)
    else:
        _echo(f"[SMOOTH] Spatial exists -> {out_spatial.name}")

    temporal_smooth_gaussian_nofft(
        out_spatial,
        out_spatiotemporal,
        tr_sec=float(tr_sec),
        sigma_sec=float(sigma_sec),
        overwrite=overwrite,
    )

    out: Dict[str, str] = {
        "in_bold": str(in_bold_4d),
        "bold_spatial": str(out_spatial),
        "bold_spatiotemporal": str(out_spatiotemporal),
    }

    if run_fft:
        fft_out = run_fft_stage_to_niftis_blockwise(
            in_bold_4d=out_spatiotemporal,
            tr_sec=float(tr_sec),
            mask_3d=fft_mask_3d,
            out_bold_dir=bold_dir,
            overwrite=overwrite,
            f_low_hz=float(fft_low_hz),
            f_high_hz=float(fft_high_hz),
            z_block=int(fft_z_block),
        )
        out.update(fft_out)

    return out


def run_spatial_and_temporal_smoothing(
    *,
    patient_dir: Path,
    in_bold_4d: Path,
    tr_sec: float,
    fwhm_mm: float = 2.5,
    sigma_sec: float = 3.0,
    fsl_env: Optional[dict] = None,
    overwrite: bool = False,
) -> Dict[str, str]:
    return run_all(
        patient_dir=patient_dir,
        in_bold_4d=in_bold_4d,
        tr_sec=tr_sec,
        fwhm_mm=fwhm_mm,
        sigma_sec=sigma_sec,
        fsl_env=fsl_env,
        overwrite=overwrite,
        run_fft=False,
        fft_mask_3d=None,
    )


# ---------------------------- CLI ----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DeepNeuroBOLD: smoothing + optional FFT stage (single script)")

    p.add_argument("--patient-dir", type=str, required=True)
    p.add_argument("--in-bold", type=str, default=None, help="Default: PREPROCESSING/bold/BOLD_brain_inT1.nii.gz")
    p.add_argument("--tr", type=float, default=1.8)

    p.add_argument("--fwhm-mm", type=float, default=2.5)
    p.add_argument("--sigma-sec", type=float, default=3.0)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--run-fft", action="store_true")
    p.add_argument("--fft-mask", type=str, default=None, help="Optional 3D mask in T1 space")
    p.add_argument("--fft-low-hz", type=float, default=0.01)
    p.add_argument("--fft-high-hz", type=float, default=0.10)
    p.add_argument("--fft-z-block", type=int, default=4, help="Z slices per FFT block (lower -> less RAM)")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    patient_dir = Path(args.patient_dir)
    in_bold = Path(args.in_bold) if args.in_bold else (
        patient_dir / "PREPROCESSING" / "bold" / "BOLD_brain_inT1.nii.gz"
    )
    _assert_img(in_bold, "input BOLD")

    fft_mask = Path(args.fft_mask) if args.fft_mask else None

    out = run_all(
        patient_dir=patient_dir,
        in_bold_4d=in_bold,
        tr_sec=float(args.tr),
        fwhm_mm=float(args.fwhm_mm),
        sigma_sec=float(args.sigma_sec),
        fsl_env=args.__dict__.get("fsl_env", None),  # keep API compatible
        overwrite=bool(args.overwrite),
        run_fft=bool(args.run_fft),
        fft_mask_3d=fft_mask,
        fft_low_hz=float(args.fft_low_hz),
        fft_high_hz=float(args.fft_high_hz),
        fft_z_block=int(args.fft_z_block),
    )

    _echo("[DONE] Outputs:")
    for k, v in out.items():
        _echo(f"  - {k}: {v}")


if __name__ == "__main__":
    main()