#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
import shutil
import traceback
import gzip
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

import nibabel as nib
import numpy as np

from deepneurobold.preprocessing.dicom2nifti import convert_patient_dicom
from deepneurobold.preprocessing.mask_to_T1 import MaskToT1
from deepneurobold.preprocessing.sequences_to_t1_old import SequencesToT1
from deepneurobold.preprocessing.gm_wm_csf import derive_healthy_gm_wm_csf_pves
from deepneurobold.preprocessing.large_vessels import run_large_vessel_mask
from deepneurobold.preprocessing.smooth_bold import (
    run_spatial_and_temporal_smoothing,
    run_fft_stage_to_niftis_blockwise,
)

try:
    from deepneurobold.preprocessing.ensure_gz import ensure_gz
except Exception:
    ensure_gz = None


def _echo(msg: str) -> None:
    print(msg, flush=True)


def _strip_nii_suffix(p: Path) -> Path:
    s = str(p)
    if s.endswith(".nii.gz"):
        return Path(s[:-7])
    if s.endswith(".nii"):
        return Path(s[:-4])
    return p


def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f}{unit}"
        n = int(n / 1024)
    return f"{n}PB"


def _gzip_integrity_ok(p: Path, read_bytes: int = 1024 * 1024) -> bool:
    """
    Fast integrity check for .nii.gz.
    Reads a small chunk (default 1MB) from the decompressor.
    If the file is truncated/corrupted, zlib.error is typically raised early.
    """
    if not p.exists() or p.stat().st_size == 0:
        return False
    if not str(p).endswith(".gz"):
        return True
    try:
        with gzip.open(str(p), "rb") as f:
            _ = f.read(read_bytes)
        return True
    except Exception:
        return False


def _safe_unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def ensure_t1_reoriented(patient_dir: Path) -> Optional[str]:
    t1_dir = Path(patient_dir) / "PREPROCESSING" / "t1"
    src = t1_dir / "t1.nii.gz"
    dst = t1_dir / "t1_reoriented.nii.gz"
    if dst.exists():
        return str(dst)
    if src.exists():
        _echo(f"[{patient_dir.name}] [T1] Reorienting T1 -> t1_reoriented.nii.gz")
        img = nib.load(str(src))
        nib.save(nib.as_closest_canonical(img), str(dst))
        return str(dst)
    return None


BranchName = Literal["nomcst", "mcst"]


class BasePreprocessing:
    """
    Fix implemented for your failure in nomcst:

    You hit:
      zlib.error: invalid literal/length code

    That error happens when nibabel tries to read a corrupted/truncated .nii.gz.
    In your trace, the corrupted file is the spatial smoothing output written to:
      PREPROCESSING/bold/BOLD_brain_inT1_spatial_smooth.nii.gz   (root)

    Why root?
      - Your smooth_bold.run_spatial_and_temporal_smoothing() does NOT support out_dir (legacy signature).
      - So it writes smoothing outputs in the bold root (on the shared filesystem).
      - If an earlier run was interrupted, that file can remain truncated and will be reused when overwrite=False.
      - Then temporal smoothing reads it and zlib explodes.

    This BasePreprocessing now:
      1) Detects whether smooth_bold has out_dir support.
      2) If legacy mode:
           - prints and proactively deletes any existing root smooth outputs + tmp files BEFORE running smoothing
             (even if overwrite=False), to avoid reusing a corrupted file.
           - after smoothing, validates gzip integrity on the outputs.
           - if still broken, cleans and retries ONCE with overwrite=True.
      3) Always moves final smooth outputs into the branch dir and cleans root contamination.
      4) Adds prints that show exactly which file is being used, sizes, and which cleanup path triggered.
    """

    # ---------------- utils ----------------
    def _run(self, cmd: List[str], env: Optional[dict] = None, step: str = "") -> None:
        label = f"[{self.patient_dir.name}]"
        if step:
            _echo(f"{label} [RUN] {step}: {' '.join(cmd)}")
        else:
            _echo(f"{label} [RUN] {' '.join(cmd)}")

        p = subprocess.run(
            cmd,
            check=True,
            env=env or self._fsl_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if p.stdout.strip():
            _echo(f"{label} [STDOUT] {p.stdout.strip()}")
        if p.stderr.strip():
            _echo(f"{label} [STDERR] {p.stderr.strip()}")

    def _assert_img(self, p: Path, tag: str) -> None:
        if (not p.exists()) or (p.is_file() and p.stat().st_size == 0):
            raise FileNotFoundError(f"[{self.patient_dir.name}] Missing/empty {tag}: {p}")

    def _print_file_status(self, p: Path, tag: str) -> None:
        if not p.exists():
            _echo(f"[{self.patient_dir.name}] [DBG] {tag}: MISSING -> {p}")
            return
        sz = p.stat().st_size
        ok = _gzip_integrity_ok(p)
        _echo(
            f"[{self.patient_dir.name}] [DBG] {tag}: EXISTS size={_human_bytes(sz)} gzip_ok={ok} path={p}"
        )

    def _fsl_env(self) -> dict:
        env = os.environ.copy()
        fsldir = self.cfg.get("fsldir") or env.get("FSLDIR")
        if fsldir:
            env["FSLDIR"] = str(fsldir)
            env["PATH"] = f"{fsldir}/bin:" + env.get("PATH", "")

        fslout = (
            self.cfg.get("fsl_outputtype")
            or self.cfg.get("fsl-outputtype")
            or env.get("FSLOUTPUTTYPE")
            or "NIFTI_GZ"
        )
        env["FSLOUTPUTTYPE"] = str(fslout)

        omp = self.cfg.get("omp_threads", 1)
        env["OMP_NUM_THREADS"] = str(omp)
        return env

    def _rewrite_uncompressed_float32(self, in_path: Path) -> Path:
        self._assert_img(in_path, f"input:{in_path.name}")
        img = nib.load(str(in_path))
        data = np.asanyarray(img.dataobj, dtype=np.float32)

        s = str(in_path)
        if s.endswith(".nii.gz"):
            out = Path(s[:-3])  # -> .nii
        elif s.endswith(".nii"):
            out = in_path
        else:
            out = in_path.with_suffix(".nii")

        _echo(f"[{self.patient_dir.name}] [IO] rewrite uncompressed float32: {in_path.name} -> {out.name}")
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(out))
        self._assert_img(out, f"rewrite_uncompressed:{out.name}")
        return out

    def _require_fsl_cmd(self, cmd: str) -> None:
        if _which(cmd) is None:
            fsldir = self.cfg.get("fsldir") or os.environ.get("FSLDIR")
            raise RuntimeError(
                f"Missing FSL command '{cmd}' in PATH. Ensure FSLDIR/bin is exported "
                f"(current FSLDIR={fsldir})."
            )

    def _move_into_branch(self, *, branch_dir: Path, src: Path, dst_name: str, overwrite: bool) -> Path:
        dst = branch_dir / dst_name
        if dst.exists() and dst.stat().st_size > 0 and (not overwrite):
            if src.exists() and src.resolve() != dst.resolve():
                _safe_unlink(src)
            return dst

        if dst.exists():
            _safe_unlink(dst)

        branch_dir.mkdir(parents=True, exist_ok=True)
        _echo(f"[{self.patient_dir.name}] [IO] move -> {dst}")
        shutil.move(str(src), str(dst))
        self._assert_img(dst, f"moved:{dst_name}")
        return dst

    def _enforce_smoothing_in_branch(
        self,
        *,
        branch_dir: Path,
        overwrite: bool,
        clean_root_contamination: bool = True,
    ) -> Dict[str, str]:
        expected = [
            "BOLD_brain_inT1_spatial_smooth.nii.gz",
            "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz",
        ]

        candidates = [branch_dir, self.bold_root_dir]

        found: Dict[str, Path] = {}
        for fn in expected:
            for d in candidates:
                p = d / fn
                if p.exists() and p.stat().st_size > 0:
                    found[fn] = p
                    break

        missing = [fn for fn in expected if fn not in found]
        if missing:
            raise FileNotFoundError(
                f"[{self.patient_dir.name}] Missing smoothing outputs after smoothing stage: {missing}. "
                f"Searched in: {[str(x) for x in candidates]}"
            )

        out_paths: Dict[str, str] = {}
        for fn in expected:
            src = found[fn]
            if src.parent.resolve() != branch_dir.resolve():
                dst = self._move_into_branch(branch_dir=branch_dir, src=src, dst_name=fn, overwrite=overwrite)
                out_paths[fn] = str(dst)
            else:
                out_paths[fn] = str(src)

        if clean_root_contamination:
            for fn in expected:
                p = self.bold_root_dir / fn
                if p.exists():
                    _echo(f"[{self.patient_dir.name}] [CLEAN] removing contaminated root file: {p}")
                    _safe_unlink(p)

            for p in self.bold_root_dir.glob("BOLD_brain_inT1_spatiotemporal_smooth.nii.gz.tmp*"):
                _echo(f"[{self.patient_dir.name}] [CLEAN] removing contaminated root tmp file: {p}")
                _safe_unlink(p)

        return out_paths

    def _set_smooth_bold_tmpdir(self) -> None:
        existing = os.environ.get("DNB_SMOOTH_TMPDIR", "").strip()
        if existing:
            Path(existing).mkdir(parents=True, exist_ok=True)
            return

        user = os.environ.get("USER", "user")
        job = os.environ.get("SLURM_JOB_ID", "nojob")
        base = os.environ.get("SLURM_TMPDIR", "").strip()

        if base:
            tmpdir = Path(base) / f"deepneurobold_smooth_{user}_{job}"
        else:
            tmpdir = Path("/scratch") / user / f"deepneurobold_smooth_{job}"

        tmpdir.mkdir(parents=True, exist_ok=True)
        os.environ["DNB_SMOOTH_TMPDIR"] = str(tmpdir)
        _echo(f"[{self.patient_dir.name}] [ENV] DNB_SMOOTH_TMPDIR={tmpdir}")

    def _smooth_bold_supports_out_dir(self) -> bool:
        import inspect
        try:
            sig = inspect.signature(run_spatial_and_temporal_smoothing)
            return "out_dir" in sig.parameters
        except Exception:
            return False

    def _legacy_smooth_root_paths(self) -> Dict[str, Path]:
        return {
            "spatial": self.bold_root_dir / "BOLD_brain_inT1_spatial_smooth.nii.gz",
            "spatiotemporal": self.bold_root_dir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz",
        }

    def _cleanup_legacy_smooth_root(self) -> None:
        paths = self._legacy_smooth_root_paths()
        _echo(f"[{self.patient_dir.name}] [CLEAN] legacy smooth outputs in root (pre-run)")
        for k, p in paths.items():
            self._print_file_status(p, f"root_smooth_{k}")
            if p.exists():
                _echo(f"[{self.patient_dir.name}] [CLEAN] deleting root file to avoid reuse: {p}")
                _safe_unlink(p)

        for p in self.bold_root_dir.glob("BOLD_brain_inT1_spatiotemporal_smooth.nii.gz.tmp*"):
            _echo(f"[{self.patient_dir.name}] [CLEAN] deleting root tmp: {p}")
            _safe_unlink(p)

    def _validate_smooth_outputs_or_raise(self, spatial: Path, spatiotemporal: Path) -> None:
        self._print_file_status(spatial, "smooth_spatial")
        self._print_file_status(spatiotemporal, "smooth_spatiotemporal")

        if not _gzip_integrity_ok(spatial):
            raise RuntimeError(f"[{self.patient_dir.name}] Corrupted gzip: {spatial}")
        if not _gzip_integrity_ok(spatiotemporal):
            raise RuntimeError(f"[{self.patient_dir.name}] Corrupted gzip: {spatiotemporal}")

        try:
            _ = nib.load(str(spatial)).shape
            _ = nib.load(str(spatiotemporal)).shape
        except Exception as e:
            raise RuntimeError(f"[{self.patient_dir.name}] nibabel could not read smooth outputs: {e}")

    # ---------------- init ----------------
    def __init__(self, patient_dir: Path, *, cfg: Optional[dict] = None):
        self.patient_dir = Path(patient_dir)
        self.cfg = cfg or {}

        self.preproc_dir = self.patient_dir / "PREPROCESSING"
        self.bold_root_dir = self.preproc_dir / "bold"
        self.t1_dir = self.preproc_dir / "t1"
        self.mask_dir = self.preproc_dir / "mask"

        self.bold_root_dir.mkdir(parents=True, exist_ok=True)
        self.t1_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        ensure_t1_reoriented(self.patient_dir)

        self.bold4d_raw = self.bold_root_dir / "BOLD.nii.gz"
        self.t1vol = self.t1_dir / "t1_reoriented.nii.gz"

        self.seg_src = self.mask_dir / "Segmentation.nii.gz"
        self.seg_in_t1 = self.mask_dir / "Segmentation_in_T1.nii.gz"

        self.mask_to_t1 = MaskToT1(interp_order=0)
        self.seq_reg = SequencesToT1(
            tform_type=self.cfg.get("ants_transform", "SyN"),
            interp=self.cfg.get("ants_interp", "linear"),
            winsorize=self.cfg.get("ants_winsorize", "0.005,0.995"),
            histmatch=self.cfg.get("ants_histmatch", True),
            threads=self.cfg.get("ants_threads", None),
        )

    def _branch_dir(self, branch: BranchName) -> Path:
        d = self.bold_root_dir / f"branch_{branch}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------------- Segmentation -> T1 (shape check) ----------------
    def _ensure_seg_in_t1(self) -> Optional[str]:
        if not self.seg_src.exists():
            _echo(f"[{self.patient_dir.name}] [MASK] WARN: missing {self.seg_src.name}")
            return None
        if not self.t1vol.exists():
            _echo(f"[{self.patient_dir.name}] [MASK] WARN: missing t1_reoriented.nii.gz")
            return None

        t1_shape = nib.load(str(self.t1vol)).shape
        needs_build = True

        if self.seg_in_t1.exists():
            try:
                if nib.load(str(self.seg_in_t1)).shape == t1_shape:
                    needs_build = False
                    _echo(f"[{self.patient_dir.name}] [MASK] Segmentation_in_T1 OK (shape matches)")
                else:
                    _echo(f"[{self.patient_dir.name}] [MASK] shape != T1 -> rebuilding")
            except Exception:
                _echo(f"[{self.patient_dir.name}] [MASK] unreadable -> rebuilding")

        if needs_build:
            _echo(f"[{self.patient_dir.name}] [MASK] Running MaskToT1 (rebuild)")
            self.mask_to_t1.run(self.patient_dir)

        if self.seg_in_t1.exists():
            return str(self.seg_in_t1)
        return None

    # ---------------- T1 brain mask ----------------
    def _ensure_t1_brain_mask(self) -> Path:
        env = self._fsl_env()
        self._assert_img(self.t1vol, "t1_reoriented")

        t1_brain = self.t1_dir / "t1_brain.nii.gz"
        t1_mask = self.t1_dir / "t1_brain_mask.nii.gz"

        if t1_mask.exists() and t1_mask.stat().st_size > 0:
            _echo(f"[{self.patient_dir.name}] [T1] brain mask exists -> {t1_mask.name}")
            return t1_mask

        bet_cmd = os.environ.get("DNB_BET_CMD", "bet2")
        if _which(bet_cmd) is None:
            bet_cmd = "bet"
        if _which(bet_cmd) is None:
            raise RuntimeError("Neither bet2 nor bet found in PATH (FSL not loaded?)")

        out_prefix = str(_strip_nii_suffix(t1_brain))
        _echo(f"[{self.patient_dir.name}] [T1] BET cmd={bet_cmd}, out_prefix={out_prefix}")

        self._run(
            [bet_cmd, str(self.t1vol), out_prefix, "-f", str(self.cfg.get("bet_frac", 0.35))],
            env,
            step="BET T1",
        )

        self._run(["fslmaths", str(t1_brain), "-bin", "-fillh", str(t1_mask)], env, step="Build t1_brain_mask")
        try:
            self._run(["fslcpgeom", str(self.t1vol), str(t1_mask)], env, step="Copy geom to mask")
        except Exception:
            pass

        self._assert_img(t1_mask, "t1_brain_mask")
        return t1_mask

    # ---------------- MC/ST (branch-local) ----------------
    def _bold_motion_and_slice_timing_branch(self, *, branch_dir: Path) -> Dict[str, str]:
        env = self._fsl_env()
        self._assert_img(self.bold4d_raw, "BOLD 4D (raw)")

        self._require_fsl_cmd("mcflirt")
        self._require_fsl_cmd("slicetimer")

        bold_mc = branch_dir / "BOLD_mc.nii.gz"
        bold_st = branch_dir / "BOLD_st.nii.gz"

        if (not (bold_mc.exists() and bold_mc.stat().st_size > 0)) or bool(self.cfg.get("mc_overwrite", False)):
            _echo(f"[{self.patient_dir.name}] [BOLD][{branch_dir.name}] Motion correction (mcflirt)")
            mcflirt_args = ["mcflirt", "-in", str(self.bold4d_raw), "-out", str(bold_mc), "-plots"]
            if bool(self.cfg.get("mc_spline_final", True)):
                mcflirt_args += ["-spline_final"]
            refvol = self.cfg.get("mc_refvol", None)
            if refvol is not None:
                mcflirt_args += ["-refvol", str(int(refvol))]
            self._run(mcflirt_args, env, step="MCFLIRT")
        self._assert_img(bold_mc, "BOLD_mc")

        if (not (bold_st.exists() and bold_st.stat().st_size > 0)) or bool(self.cfg.get("st_overwrite", False)):
            _echo(f"[{self.patient_dir.name}] [BOLD][{branch_dir.name}] Slice timing correction (slicetimer)")
            st_args = ["slicetimer", "-i", str(bold_mc), "-o", str(bold_st)]

            st_mode = self.cfg.get("st_mode", "odd")
            if st_mode == "odd":
                st_args += ["--odd"]
            elif st_mode == "down":
                st_args += ["--down"]
            elif st_mode == "up":
                st_args += ["--up"]

            st_custom = self.cfg.get("st_custom", None)
            if st_custom:
                st_args += ["--tcustom", str(st_custom)]

            st_tr = self.cfg.get("tr_sec", None)
            if st_tr is not None:
                st_args += ["-r", str(float(st_tr))]

            self._run(st_args, env, step="SLICETIMER")
        self._assert_img(bold_st, "BOLD_st")

        return {"bold_mc": str(bold_mc), "bold_st": str(bold_st)}

    # ---------------- BOLD->T1 + mask (branch-local) ----------------
    def _bold_to_t1_branch(self, *, in_bold_4d: Path, branch_dir: Path) -> Dict[str, str]:
        env = self._fsl_env()

        in_bold_4d = Path(in_bold_4d)
        self._assert_img(in_bold_4d, f"BOLD 4D input ({in_bold_4d.name})")
        self._assert_img(self.t1vol, "T1 reoriented")

        mask = self.t1_dir / "t1_brain_mask.nii.gz"
        self._assert_img(mask, "t1_brain_mask")

        T1 = str(self.t1vol)
        B4 = str(in_bold_4d)

        bold_mean = branch_dir / "BOLD_mean.nii.gz"
        bold2t1_mat = branch_dir / "bold2t1.mat"
        bold_mean_inT1_gz = branch_dir / "BOLD_mean_inT1.nii.gz"
        bold_inT1_gz = branch_dir / "BOLD_inT1.nii.gz"
        bold_brain_inT1_gz = branch_dir / "BOLD_brain_inT1.nii.gz"
        bold_mean_brain_T1_gz = branch_dir / "BOLD_mean_brain_inT1.nii.gz"

        if (not bold_mean.exists()) or bool(self.cfg.get("mean_overwrite", False)):
            self._run(["fslmaths", B4, "-Tmean", str(bold_mean)], env, step="BOLD Tmean")
        self._assert_img(bold_mean, "BOLD_mean")

        if (not bold_mean_inT1_gz.exists()) or (not bold2t1_mat.exists()) or bool(self.cfg.get("flirt_overwrite", False)):
            self._run(
                [
                    "flirt",
                    "-in", str(bold_mean),
                    "-ref", T1,
                    "-omat", str(bold2t1_mat),
                    "-out", str(bold_mean_inT1_gz),
                    "-dof", "6",
                    "-cost", "normmi",
                    "-interp", "trilinear",
                ],
                env,
                step="FLIRT mean->T1",
            )
        self._assert_img(bold_mean_inT1_gz, "BOLD_mean_inT1")
        self._assert_img(bold2t1_mat, "bold2t1.mat")

        if (not bold_inT1_gz.exists()) or bool(self.cfg.get("flirt_overwrite", False)):
            self._run(
                [
                    "flirt",
                    "-in", B4,
                    "-ref", T1,
                    "-applyxfm",
                    "-init", str(bold2t1_mat),
                    "-out", str(bold_inT1_gz),
                    "-interp", "trilinear",
                ],
                env,
                step="FLIRT 4D->T1",
            )
        self._assert_img(bold_inT1_gz, "BOLD_inT1.gz")

        bold_inT1_nu = self._rewrite_uncompressed_float32(bold_inT1_gz)
        mean_inT1_nu = self._rewrite_uncompressed_float32(bold_mean_inT1_gz)

        self._run(
            ["fslmaths", str(bold_inT1_nu), "-mas", str(mask), str(bold_brain_inT1_gz)],
            env,
            step="Mask BOLD 4D",
        )
        self._run(
            ["fslmaths", str(mean_inT1_nu), "-mas", str(mask), str(bold_mean_brain_T1_gz)],
            env,
            step="Mask BOLD mean",
        )

        self._assert_img(bold_brain_inT1_gz, "BOLD_brain_inT1")
        self._assert_img(bold_mean_brain_T1_gz, "BOLD_mean_brain_inT1")

        return {
            "bold_input_used": str(in_bold_4d),
            "bold_mean": str(bold_mean),
            "bold2t1_mat": str(bold2t1_mat),
            "bold_mean_in_t1": str(bold_mean_inT1_gz),
            "bold_in_t1": str(bold_inT1_gz),
            "bold_brain_in_t1": str(bold_brain_inT1_gz),
            "bold_mean_brain_in_t1": str(bold_mean_brain_T1_gz),
        }

    # ---------------- vessel mask (branch-local) ----------------
    def _optional_large_vessel_mask_branch(self, *, branch_dir: Path) -> Optional[Dict[str, Any]]:
        if not bool(self.cfg.get("vessel_mask_enable", False)):
            return None

        bold_brain_in_t1 = branch_dir / "BOLD_brain_inT1.nii.gz"
        t1_brain_mask = self.t1_dir / "t1_brain_mask.nii.gz"
        self._assert_img(bold_brain_in_t1, f"{branch_dir.name}/BOLD_brain_inT1")
        self._assert_img(t1_brain_mask, "t1_brain_mask")

        _echo(f"[{self.patient_dir.name}] [VESSEL][{branch_dir.name}] enabled. Calling run_large_vessel_mask...")

        out_dir = branch_dir / "Vessel_removal"
        result = run_large_vessel_mask(bold_brain_in_t1, t1_brain_mask, out_dir, self.cfg)

        if not result or not result.get("enabled"):
            _echo(f"[{self.patient_dir.name}] [VESSEL][{branch_dir.name}] SKIPPED or FAILED")
            return None

        overwrite = bool(self.cfg.get("vessel_overwrite_bold_brain_in_t1", True))
        bold_cleaned_path = Path(result["bold_cleaned"])
        final_bold_path = bold_cleaned_path

        if overwrite:
            _echo(f"[{self.patient_dir.name}] [VESSEL][{branch_dir.name}] Overwriting branch BOLD_brain_inT1 with cleaned data.")
            backup = branch_dir / "BOLD_brain_inT1__pre_vesselclean.nii.gz"

            if not backup.exists():
                bold_brain_in_t1.rename(backup)
                self._assert_img(backup, f"{branch_dir.name}/backup")

            if bold_cleaned_path.exists() and bold_cleaned_path.stat().st_size > 0:
                if bold_brain_in_t1.exists():
                    _safe_unlink(bold_brain_in_t1)
                bold_cleaned_path.rename(bold_brain_in_t1)
                final_bold_path = bold_brain_in_t1
            else:
                raise FileNotFoundError(f"[{self.patient_dir.name}] vessel-cleaned BOLD missing: {bold_cleaned_path}")

            self._assert_img(bold_brain_in_t1, f"{branch_dir.name}/BOLD_brain_inT1 (vesselclean)")

        result["overwrite"] = overwrite
        result["final_bold_path"] = str(final_bold_path)
        return result

    # ---------------- one branch ----------------
    def _run_one_branch(
        self,
        *,
        branch: BranchName,
        run_bold_to_t1: bool,
        run_large_vessel_mask_flag: bool,
        run_motion_correction: bool,
        run_slice_timing: bool,
    ) -> Dict[str, Any]:
        branch_dir = self._branch_dir(branch)

        _echo(f"[{self.patient_dir.name}] [DBG] branch_dir={branch_dir}")
        _echo(f"[{self.patient_dir.name}] [DBG] bold_raw={self.bold4d_raw}")
        _echo(f"[{self.patient_dir.name}] [DBG] t1={self.t1vol}")
        _echo(f"[{self.patient_dir.name}] [DBG] smooth_supports_out_dir={self._smooth_bold_supports_out_dir()}")

        bold_for_reg = self.bold4d_raw
        mcst_info: Optional[Dict[str, str]] = None

        if branch == "mcst":
            _echo(f"[{self.patient_dir.name}] [BRANCH {branch}] MC/ST stage")
            if run_motion_correction:
                mcst_info = self._bold_motion_and_slice_timing_branch(branch_dir=branch_dir)
                if run_slice_timing:
                    bold_for_reg = Path(mcst_info["bold_st"])
                else:
                    bold_for_reg = Path(mcst_info["bold_mc"])
            else:
                _echo(f"[{self.patient_dir.name}] [BRANCH {branch}] MC disabled -> using RAW")
                bold_for_reg = self.bold4d_raw
        else:
            _echo(f"[{self.patient_dir.name}] [BRANCH {branch}] using RAW (no MC/ST)")
            bold_for_reg = self.bold4d_raw

        if run_bold_to_t1:
            _echo(f"[{self.patient_dir.name}] [BRANCH {branch}] BOLD->T1 + mask")
            bold_to_t1_info = self._bold_to_t1_branch(in_bold_4d=bold_for_reg, branch_dir=branch_dir)
        else:
            bold_to_t1_info = {"bold_input_used": str(bold_for_reg)}

        vessel_info = None
        active_bold_in_t1 = branch_dir / "BOLD_brain_inT1.nii.gz"

        if run_large_vessel_mask_flag:
            _echo(f"[{self.patient_dir.name}] [BRANCH {branch}] Vessel removal (optional)")
            try:
                vessel_info = self._optional_large_vessel_mask_branch(branch_dir=branch_dir)
            except Exception as e:
                _echo(f"[{self.patient_dir.name}] [WARN][{branch}] vessel masking failed: {e}")
                traceback.print_exc()

        if vessel_info and vessel_info.get("final_bold_path"):
            active_bold_in_t1 = Path(vessel_info["final_bold_path"])

        self._assert_img(active_bold_in_t1, f"{branch_dir.name}/active_bold_in_t1")
        self._print_file_status(active_bold_in_t1, "active_bold_in_t1")

        # ------------------- SMOOTHING ALWAYS -------------------
        self._set_smooth_bold_tmpdir()

        _echo(f"[{self.patient_dir.name}] [BRANCH {branch}] Spatio-temporal smoothing (ALWAYS)")
        tr_sec = float(self.cfg.get("tr_sec", 1.8))
        fwhm_mm = float(self.cfg.get("smooth_fwhm_mm", 2.5))
        sigma_sec = float(self.cfg.get("smooth_sigma_sec", 3.0))
        overwrite = bool(self.cfg.get("smooth_overwrite", False))

        _echo(f"[{self.patient_dir.name}] [DBG] smoothing params: TR={tr_sec} fwhm_mm={fwhm_mm} sigma_sec={sigma_sec} overwrite={overwrite}")
        _echo(f"[{self.patient_dir.name}] [DBG] DNB_SMOOTH_TMPDIR={os.environ.get('DNB_SMOOTH_TMPDIR','')}")

        supports_out_dir = self._smooth_bold_supports_out_dir()

        if not supports_out_dir:
            self._cleanup_legacy_smooth_root()

        smooth_info: Dict[str, Any] = {}
        attempted_retry = False

        while True:
            try:
                if supports_out_dir:
                    smooth_info = run_spatial_and_temporal_smoothing(
                        patient_dir=self.patient_dir,
                        in_bold_4d=Path(active_bold_in_t1),
                        tr_sec=tr_sec,
                        fwhm_mm=fwhm_mm,
                        sigma_sec=sigma_sec,
                        fsl_env=self._fsl_env(),
                        overwrite=overwrite,
                        out_dir=branch_dir,
                    )
                else:
                    _echo(f"[{self.patient_dir.name}] [DBG] smooth_bold legacy signature: outputs go to bold root and will be moved.")
                    smooth_info = run_spatial_and_temporal_smoothing(
                        patient_dir=self.patient_dir,
                        in_bold_4d=Path(active_bold_in_t1),
                        tr_sec=tr_sec,
                        fwhm_mm=fwhm_mm,
                        sigma_sec=sigma_sec,
                        fsl_env=self._fsl_env(),
                        overwrite=overwrite,
                    )

                if supports_out_dir:
                    sp = branch_dir / "BOLD_brain_inT1_spatial_smooth.nii.gz"
                    st = branch_dir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz"
                else:
                    sp = self.bold_root_dir / "BOLD_brain_inT1_spatial_smooth.nii.gz"
                    st = self.bold_root_dir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz"

                self._validate_smooth_outputs_or_raise(sp, st)
                break

            except Exception as e:
                _echo(f"[{self.patient_dir.name}] [ERR] smoothing failed: {e}")
                traceback.print_exc()

                if attempted_retry:
                    raise

                attempted_retry = True
                _echo(f"[{self.patient_dir.name}] [FIX] retry smoothing once: cleaning outputs + forcing overwrite=True")

                for p in [
                    branch_dir / "BOLD_brain_inT1_spatial_smooth.nii.gz",
                    branch_dir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz",
                    self.bold_root_dir / "BOLD_brain_inT1_spatial_smooth.nii.gz",
                    self.bold_root_dir / "BOLD_brain_inT1_spatiotemporal_smooth.nii.gz",
                ]:
                    _safe_unlink(p)

                for p in self.bold_root_dir.glob("BOLD_brain_inT1_spatiotemporal_smooth.nii.gz.tmp*"):
                    _safe_unlink(p)

                overwrite = True
                continue

        moved = self._enforce_smoothing_in_branch(
            branch_dir=branch_dir,
            overwrite=overwrite,
            clean_root_contamination=True,
        )

        bold_sp = Path(moved["BOLD_brain_inT1_spatial_smooth.nii.gz"])
        bold_st = Path(moved["BOLD_brain_inT1_spatiotemporal_smooth.nii.gz"])

        self._assert_img(bold_sp, f"{branch_dir.name}/BOLD spatial smooth")
        self._assert_img(bold_st, f"{branch_dir.name}/BOLD spatiotemporal smooth")
        self._validate_smooth_outputs_or_raise(bold_sp, bold_st)

        # ------------------- FFT ALWAYS (unless explicitly disabled) -------------------
        fft_enable = bool(self.cfg.get("fft_enable", True))
        fft_info = None

        _echo(f"[{self.patient_dir.name}] [DBG] fft_enable={fft_enable}")

        if fft_enable:
            _echo(f"[{self.patient_dir.name}] [BRANCH {branch}] FFT low-freq maps (blockwise) (ALWAYS)")

            fft_mask_path = self.cfg.get("fft_mask_3d", None)
            fft_mask_3d = Path(fft_mask_path) if fft_mask_path else None
            if fft_mask_3d is not None:
                self._print_file_status(fft_mask_3d, "fft_mask_3d")

            fft_overwrite = bool(self.cfg.get("fft_overwrite", False))
            fft_low_hz = float(self.cfg.get("fft_low_hz", 0.01))
            fft_high_hz = float(self.cfg.get("fft_high_hz", 0.10))
            fft_z_block = int(self.cfg.get("fft_z_block", 2))

            _echo(
                f"[{self.patient_dir.name}] [DBG] fft params: overwrite={fft_overwrite} "
                f"low_hz={fft_low_hz} high_hz={fft_high_hz} z_block={fft_z_block} in={bold_st}"
            )

            fft_info = run_fft_stage_to_niftis_blockwise(
                in_bold_4d=bold_st,
                tr_sec=float(tr_sec),
                mask_3d=fft_mask_3d,
                out_bold_dir=branch_dir,
                overwrite=fft_overwrite,
                f_low_hz=float(fft_low_hz),
                f_high_hz=float(fft_high_hz),
                z_block=int(fft_z_block),
            )

            for f in [
                "FFT_power_low_inT1.nii.gz",
                "FFT_power_ratio_low_inT1.nii.gz",
                "FFT_peak_freq_low_inT1.nii.gz",
                "FFT_spectral_entropy_low_inT1.nii.gz",
                "FFT_spectral_slope_low_inT1.nii.gz",
            ]:
                self._assert_img(branch_dir / f, f"{branch_dir.name}/{f}")
                self._print_file_status(branch_dir / f, f"fft_out:{f}")

        out: Dict[str, Any] = {
            "branch": branch,
            "branch_dir": str(branch_dir),
            "bold_4d_raw": str(self.bold4d_raw),
            "bold_4d_for_reg": str(bold_for_reg),
            "t1": str(self.t1vol),
            "t1_brain_mask": str(self.t1_dir / "t1_brain_mask.nii.gz"),
            "bold_to_t1": bold_to_t1_info,
            "active_bold_in_t1": str(active_bold_in_t1),
            "smoothing": smooth_info,
            "smoothing_outputs": {
                "spatial": str(bold_sp),
                "spatiotemporal": str(bold_st),
            },
            "fft_enable": bool(fft_enable),
            "motion_correction_enabled": bool(run_motion_correction) and (branch == "mcst"),
            "slice_timing_enabled": bool(run_slice_timing) and (branch == "mcst"),
            "smooth_tmpdir": os.environ.get("DNB_SMOOTH_TMPDIR", ""),
            "smooth_supports_out_dir": supports_out_dir,
            "smoothing_retry_happened": attempted_retry,
        }

        if mcst_info is not None:
            out["motion_slice_timing"] = mcst_info
        if vessel_info is not None:
            out["vessel_masking"] = vessel_info
        if fft_info is not None:
            out["fft"] = fft_info

        return out

    # ---------------- pipeline ----------------
    def run(
        self,
        *,
        run_dicom: bool = True,
        run_bold_to_t1: bool = True,
        run_large_vessel_mask: bool = True,
        run_ensure_gz: bool = True,
        run_mask_to_t1: bool = True,
        run_gm_wm_pves: bool = True,
        run_sequences_to_t1: bool = True,
        run_motion_correction: bool = False,
        run_slice_timing: bool = False,
        run_branches: Literal["nomcst", "mcst", "both"] = "both",
    ) -> Dict[str, Any]:
        _echo(f"[{self.patient_dir.name}] ===== PREPROCESS START =====")

        # 0) Ensure smoothing tempdir is safe early
        self._set_smooth_bold_tmpdir()

        # 1) DICOM -> NIfTI (shared raw)
        if run_dicom:
            _echo(f"[{self.patient_dir.name}] [STEP 1] DICOM->NIfTI")
            if not (self.bold4d_raw.exists() and (self.t1_dir / "t1.nii.gz").exists()):
                convert_patient_dicom(self.patient_dir)
                ensure_t1_reoriented(self.patient_dir)
            else:
                _echo(f"[{self.patient_dir.name}] [STEP 1] SKIP (already have BOLD + T1)")

        self._assert_img(self.bold4d_raw, "BOLD.nii.gz (raw)")
        self._assert_img(self.t1vol, "t1_reoriented.nii.gz")

        # 2) T1 brain mask (shared)
        _echo(f"[{self.patient_dir.name}] [STEP 2] T1 brain mask")
        t1_mask = self._ensure_t1_brain_mask()
        self._assert_img(Path(t1_mask), "t1_brain_mask")

        # 3) Segmentation -> T1 (shape check)
        if run_mask_to_t1:
            _echo(f"[{self.patient_dir.name}] [STEP 3] Segmentation->T1 (shape check)")
            try:
                self._ensure_seg_in_t1()
            except Exception as e:
                _echo(f"[{self.patient_dir.name}] [WARN] mask_to_T1 failed: {e}")
                traceback.print_exc()

        # 4) GM/WM/CSF PVEs
        if run_gm_wm_pves:
            _echo(f"[{self.patient_dir.name}] [STEP 4] FAST PVEs")
            try:
                derive_healthy_gm_wm_csf_pves(self.patient_dir)
            except Exception as e:
                _echo(f"[{self.patient_dir.name}] [WARN] gm_wm_csf failed: {e}")
                traceback.print_exc()

        # 5) Sequences -> T1
        if run_sequences_to_t1:
            _echo(f"[{self.patient_dir.name}] [STEP 5] ANTs sequences->T1")
            try:
                self.seq_reg.run(self.patient_dir)
            except Exception as e:
                _echo(f"[{self.patient_dir.name}] [WARN] sequences_to_t1 failed: {e}")
                traceback.print_exc()

        # Branch selection
        branches: List[BranchName]
        if run_branches == "both":
            branches = ["nomcst", "mcst"]
        elif run_branches == "nomcst":
            branches = ["nomcst"]
        else:
            branches = ["mcst"]

        out: Dict[str, Any] = {
            "patient_dir": str(self.patient_dir),
            "t1": str(self.t1vol),
            "t1_brain_mask": str(self.t1_dir / "t1_brain_mask.nii.gz"),
            "bold_4d_raw": str(self.bold4d_raw),
            "branches": {},
            "smooth_tmpdir": os.environ.get("DNB_SMOOTH_TMPDIR", ""),
        }

        for b in branches:
            _echo(f"[{self.patient_dir.name}] ===== BRANCH {b} START =====")
            branch_out = self._run_one_branch(
                branch=b,
                run_bold_to_t1=run_bold_to_t1,
                run_large_vessel_mask_flag=run_large_vessel_mask,
                run_motion_correction=run_motion_correction if b == "mcst" else False,
                run_slice_timing=run_slice_timing if b == "mcst" else False,
            )
            out["branches"][b] = branch_out
            _echo(f"[{self.patient_dir.name}] ===== BRANCH {b} DONE =====")

        # ensure_gz (optional)
        if run_ensure_gz and ensure_gz is not None:
            _echo(f"[{self.patient_dir.name}] [STEP 6] ensure_gz (optional)")
            try:
                ensure_gz(self.patient_dir)
            except Exception as e:
                _echo(f"[{self.patient_dir.name}] [WARN] ensure_gz failed: {e}")
                traceback.print_exc()

        _echo(f"[{self.patient_dir.name}] ===== PREPROCESS DONE =====")
        return out