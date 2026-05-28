#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
from pathlib import Path
from collections import defaultdict
import subprocess
from subprocess import run, CalledProcessError

# third-party
import nibabel as nib  # for T1 reorientation without FSL


# -----------------------------
# FSL / dcm2niix from environment
# -----------------------------
FSLDIR = os.getenv("FSLDIR", "<FSLDIR>")
DCM2NIIX_PATH = os.getenv("DCM2NIIX", str(Path(FSLDIR) / "bin" / "dcm2niix"))


def run_in_fsl(cmd_str: str) -> subprocess.CompletedProcess:
    """Run a shell command in a bash subshell with FSL fully configured."""
    bash_line = (
        f'export FSLDIR="{FSLDIR}"; '
        f'[[ -f "$FSLDIR/etc/fslconf/fsl.sh" ]] && source "$FSLDIR/etc/fslconf/fsl.sh"; '
        f'export PATH="$FSLDIR/bin:$PATH"; '
        # ---- environment knobs to avoid OOM on large series ----
        f'export TMPDIR="/scratch/$USER/tmp"; '
        f'mkdir -p "$TMPDIR"; '
        f'export OMP_NUM_THREADS=1; '
        f'export FSLOUTPUTTYPE=NIFTI_GZ; '
        # ---------------------------------
        f'{cmd_str}'
    )
    return run(["bash", "-lc", bash_line], check=False, capture_output=True, text=True)


def reorient_to_ras(input_nii: str, output_nii: str) -> None:
    """Reorient NIfTI to canonical RAS using nibabel (no FSL)."""
    img = nib.load(input_nii)
    ras_img = nib.as_closest_canonical(img)
    nib.save(ras_img, output_nii)


class DicomToNiftiConverter:
    """Internal worker for a single modality folder."""

    def __init__(self, input_dir, output_dir, dcm2niix_path, modality="BOLD"):
        self.input_dir = str(input_dir)
        self.output_dir = str(output_dir)
        self.dcm2niix_path = str(dcm2niix_path)
        self.modality = modality.upper()
        os.makedirs(self.output_dir, exist_ok=True)

    def convert(self):
        if self.modality == "BOLD":
            parts = self._convert_bold()
            return self._merge_bold(parts) if parts else None
        if self.modality == "T1":
            return self._convert_t1()
        print(f"[ERR] Unknown modality: {self.modality}")
        return None

    # ---------------- BOLD ----------------
    def _find_dicom_groups(self):
        """Group DICOMs by series number extracted from filename (pattern: MR<digits>)."""
        files = glob.glob(os.path.join(self.input_dir, "*.dcm"))
        groups = defaultdict(list)
        for f in files:
            m = re.search(r"MR\D*(\d{4,})", os.path.basename(f))
            if m:
                groups[m.group(1)].append(f)
        return {k: v for k, v in groups.items() if len(v) == 50}

    def _convert_bold(self):
        grouped = self._find_dicom_groups()
        if not grouped:
            print(f"[BOLD] No valid groups of 50 DICOMs in {self.input_dir}")
            return None

        nifti_parts = []
        for series_no, files in grouped.items():
            tmp_dir = os.path.join(self.output_dir, f"tmp_{series_no}")
            os.makedirs(tmp_dir, exist_ok=True)

            # Symlink DICOMs (do not move originals)
            for src in files:
                link_path = os.path.join(tmp_dir, os.path.basename(src))
                try:
                    os.symlink(src, link_path)
                except FileExistsError:
                    pass

            cmd = [
                self.dcm2niix_path,
                "-o", self.output_dir,
                "-f", f"BOLD_{series_no}",
                "-b", "n",
                "-z", "y",
                "-m", "y",
                "-x", "y",
                tmp_dir,
            ]
            try:
                run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except CalledProcessError as e:
                print(f"[BOLD] dcm2niix failed for series {series_no}: {e}")
                self._cleanup_dir(tmp_dir)
                return None

            produced = sorted(
                glob.glob(os.path.join(self.output_dir, f"BOLD_{series_no}*.nii.gz"))
            )
            if produced:
                nifti_parts.append(produced[0])
            else:
                print(f"[BOLD] Warning: no NIfTI produced for series {series_no}")

            self._cleanup_dir(tmp_dir)

        return nifti_parts

    def _merge_bold(self, nifti_files):
        if not nifti_files:
            print("[BOLD] Nothing to merge.")
            return None

        out_path = os.path.join(self.output_dir, "BOLD.nii.gz")
        parts_quoted = " ".join(f'"{p}"' for p in nifti_files)
        res = run_in_fsl(f'fslmerge -t "{out_path}" {parts_quoted}')

        print("[BOLD] fslmerge rc=", res.returncode)
        if res.stdout:
            print("[BOLD] fslmerge stdout:\n", res.stdout)
        if res.stderr:
            print("[BOLD] fslmerge stderr:\n", res.stderr)

        if res.returncode != 0 or not os.path.exists(out_path):
            print(f"[BOLD] Merge failed or output missing: {out_path}")
            return None

        # Convert to float32 to reduce memory footprint in downstream stages
        tmp_f32 = out_path.replace(".nii.gz", "_f32.nii.gz")
        res2 = run_in_fsl(
            f'fslmaths "{out_path}" -odt float "{tmp_f32}" && mv "{tmp_f32}" "{out_path}"'
        )
        if res2.returncode != 0:
            print("[BOLD] Warning: float32 conversion failed:\n", res2.stderr)

        for p in nifti_files:
            try:
                os.remove(p)
            except OSError:
                pass

        print(f"[BOLD] 4D merged (float32): {out_path}")
        return out_path

    # ---------------- T1 ----------------
    def _convert_t1(self):
        """Convert T1 DICOMs and reorient to canonical RAS."""
        print(f"[T1] Converting DICOMs from: {self.input_dir}")
        cmd = [
            self.dcm2niix_path,
            "-o",
            self.output_dir,
            "-f",
            "T1",
            "-b",
            "n",
            "-z",
            "y",
            "-m",
            "n",
            "-x",
            "n",
            self.input_dir,
        ]
        conv = run(cmd, check=False, capture_output=True, text=True)
        print("[T1] dcm2niix rc=", conv.returncode)
        if conv.stdout:
            print("[T1] dcm2niix stdout:\n", conv.stdout)
        if conv.stderr:
            print("[T1] dcm2niix stderr:\n", conv.stderr)
        if conv.returncode != 0:
            print("[T1] dcm2niix failed.")
            return None

        produced_list = sorted(glob.glob(os.path.join(self.output_dir, "T1*.nii*")))
        if not produced_list:
            print("[T1] No NIfTI produced by dcm2niix.")
            return None

        produced = produced_list[0]
        final_path = os.path.join(self.output_dir, "t1.nii.gz")
        print(f"[T1] Reorient to RAS (nibabel): {produced} -> {final_path}")
        try:
            reorient_to_ras(produced, final_path)
        except Exception as e:
            print(f"[T1] Reorientation failed: {e}")
            return None

        if os.path.exists(final_path):
            print(f"[T1] ✅ Final file: {final_path}")
        else:
            print(f"[T1] ❌ Final file missing: {final_path}")
            return None

        if os.path.abspath(produced) != os.path.abspath(final_path) and os.path.exists(produced):
            try:
                os.remove(produced)
                print(f"[T1] Removed intermediate: {produced}")
            except OSError:
                pass

        return final_path

    # ---------------- utils ----------------
    @staticmethod
    def _cleanup_dir(path: str):
        try:
            for p in glob.glob(os.path.join(path, "*")):
                try:
                    os.remove(p)
                except OSError:
                    pass
            os.rmdir(path)
        except OSError:
            pass


# ---------------- High-level API (per patient) ----------------
def convert_patient_dicom(patient_dir: Path, dcm2niix_path: str = DCM2NIIX_PATH) -> dict:
    """Convert DICOMs for one patient into PREPROCESSING/{bold,t1}."""
    patient_dir = Path(patient_dir)
    bold_dcm = patient_dir / "BOLD"
    t1_dcm = patient_dir / "T1"

    out_bold = patient_dir / "PREPROCESSING" / "bold"
    out_t1 = patient_dir / "PREPROCESSING" / "t1"
    out_bold.mkdir(parents=True, exist_ok=True)
    out_t1.mkdir(parents=True, exist_ok=True)

    out = {"bold_4d": None, "t1": None}

    if bold_dcm.exists():
        print("=== BOLD: convert & merge ===")
        b = DicomToNiftiConverter(bold_dcm, out_bold, dcm2niix_path, modality="BOLD")
        out["bold_4d"] = b.convert()

    if t1_dcm.exists():
        print("\n=== T1: convert & reorient (nibabel) ===")
        t = DicomToNiftiConverter(t1_dcm, out_t1, dcm2niix_path, modality="T1")
        out["t1"] = t.convert()

    return out