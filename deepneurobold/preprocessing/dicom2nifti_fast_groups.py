#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepneurobold.preprocessing.dicom2nifti_fast_groups
=====================================================
Fast DICOM → NIfTI conversion in fixed-size groups:

- BOLD: convert in blocks (e.g. 50 DICOMs at a time) with ``dcm2niix``,
  then concatenate the per-block NIfTIs along the time axis with ``fslmerge``.
- T1: convert once and reorient to canonical RAS with
  ``nibabel.as_closest_canonical``.

Requirements
------------
- ``dcm2niix`` reachable (default ``$FSLDIR/bin/dcm2niix``; configurable
  via the ``DCM2NIIX`` environment variable).
- FSL utilities (``fslmerge``, ``fslmaths``) on ``PATH`` (set via ``$FSLDIR``).

Expected per-patient inputs (when present)
------------------------------------------
- ``<patient_dir>/BOLD/*.dcm``  (or any DICOM extension recognised by dcm2niix)
- ``<patient_dir>/T1/*.dcm``

Outputs
-------
- ``<patient_dir>/PREPROCESSING/bold/BOLD.nii.gz``
- ``<patient_dir>/PREPROCESSING/t1/t1.nii.gz``   (reoriented to RAS)
"""

from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path
from subprocess import CalledProcessError, run

import nibabel as nib

# ---------------------------------------------------------------------------
# FSL / dcm2niix environment
# ---------------------------------------------------------------------------
FSLDIR = os.getenv("FSLDIR", "<FSLDIR>")
DCM2NIIX_PATH = os.getenv("DCM2NIIX", str(Path(FSLDIR) / "bin" / "dcm2niix"))


def _run_bash_in_fsl(cmd_str: str):
    """Run a bash command line inside a clean FSL-configured shell.

    Returns the :class:`subprocess.CompletedProcess` with stdout/stderr captured.
    """
    bash_line = (
        f'export FSLDIR="{FSLDIR}"; '
        f'[[ -f "$FSLDIR/etc/fslconf/fsl.sh" ]] && source "$FSLDIR/etc/fslconf/fsl.sh"; '
        f'export PATH="$FSLDIR/bin:$PATH"; '
        f'export OMP_NUM_THREADS=1; '
        f'export FSLOUTPUTTYPE=NIFTI_GZ; '
        f'export TMPDIR="${{SLURM_TMPDIR:-/scratch/$USER/tmp}}"; mkdir -p "$TMPDIR"; '
        f'{cmd_str}'
    )
    return run(["bash", "-lc", bash_line], check=False, capture_output=True, text=True)


def _reorient_to_ras(input_nii: str, output_nii: str) -> None:
    img = nib.load(input_nii)
    ras_img = nib.as_closest_canonical(img)
    nib.save(ras_img, output_nii)


def _list_dicoms(dirname: Path) -> list[str]:
    """List DICOM files in *dirname* tolerantly.

    Accepts ``.dcm``, ``.ima``, and extension-less files (common in PACS dumps).
    Filters out obvious metadata such as ``.json``, ``.bval``, ``.bvec``.
    """
    if not dirname.is_dir():
        return []
    patterns = [
        str(dirname / "*.dcm"),
        str(dirname / "*.DCM"),
        str(dirname / "*.ima"),
        str(dirname / "*.IMA"),
        str(dirname / "*"),
    ]
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    out: list[str] = []
    for f in sorted(set(files)):
        p = Path(f)
        if not p.is_file():
            continue
        if p.suffix.lower() in {".json", ".bval", ".bvec"}:
            continue
        out.append(str(p))
    return out


class DicomToNiftiConverter:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        dcm2niix_path: str,
        modality: str = "BOLD",
        group_size: int = 50,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.dcm2niix_path = str(dcm2niix_path)
        self.modality = modality.upper()
        self.group_size = int(os.getenv("DNB_DICOM_GROUP", group_size))
        os.makedirs(self.output_dir, exist_ok=True)

    def convert(self):
        if self.modality == "BOLD":
            parts = self._convert_bold_groups()
            return self._merge_bold(parts)
        if self.modality == "T1":
            return self._convert_t1()
        raise ValueError(f"Unknown modality: {self.modality}")

    # ---------------- BOLD (group-wise) ----------------

    def _find_dicom_groups(self) -> list[list[str]]:
        files = _list_dicoms(self.input_dir)
        if not files:
            print(f"[BOLD] No DICOMs in {self.input_dir}")
            return []
        g = self.group_size if self.group_size > 0 else 50
        return [files[i:i + g] for i in range(0, len(files), g)]

    def _convert_bold_groups(self) -> list[str] | None:
        if not Path(self.dcm2niix_path).exists():
            print(f"[BOLD] dcm2niix not found at {self.dcm2niix_path}")
            return None

        groups = self._find_dicom_groups()
        if not groups:
            return None

        produced: list[str] = []
        for idx, group in enumerate(groups, 1):
            print(f"[BOLD] Converting group {idx}/{len(groups)} ({len(group)} files)")

            with tempfile.TemporaryDirectory(dir=self.output_dir) as tmp_dir:
                # Stage the current group's DICOMs into a private dir so
                # dcm2niix sees only the right subset (it does not accept
                # a file list on stdin).
                grp_dir = Path(tmp_dir) / f"group_{idx:03d}"
                grp_dir.mkdir(parents=True, exist_ok=True)
                for f in group:
                    try:
                        # Prefer hard link (instant, no copy); fall back to
                        # copy across filesystems.
                        os.link(f, grp_dir / Path(f).name)
                    except OSError:
                        import shutil
                        shutil.copy2(f, grp_dir / Path(f).name)

                try:
                    cmd = [
                        self.dcm2niix_path,
                        "-o", str(self.output_dir),
                        "-f", f"BOLD_{idx:03d}",
                        "-b", "n",
                        "-z", "y",
                        "-m", "n",
                        "-x", "n",
                        str(grp_dir),
                    ]
                    run(cmd, check=True, capture_output=True, text=True)
                except CalledProcessError as e:
                    print(f"[BOLD] dcm2niix failed for group {idx}: {e.stderr.strip()}")
                    continue

                out = sorted(glob.glob(str(self.output_dir / f"BOLD_{idx:03d}*.nii.gz")))
                if out:
                    produced.append(out[0])

        return produced

    def _merge_bold(self, nifti_files: list[str] | None) -> str | None:
        if not nifti_files:
            print("[BOLD] Nothing to merge.")
            return None

        out_path = str(self.output_dir / "BOLD.nii.gz")
        print(f"[BOLD] Merging {len(nifti_files)} files -> {out_path}")
        parts_quoted = " ".join(f'"{p}"' for p in nifti_files)
        res = _run_bash_in_fsl(f'fslmerge -t "{out_path}" {parts_quoted}')

        if res.returncode != 0:
            print("[BOLD] Merge failed:")
            print(res.stderr.strip())
            return None

        # Force float32 dtype for downstream consistency.
        _run_bash_in_fsl(f'fslmaths "{out_path}" -odt float "{out_path}"')

        # Clean up per-group partials.
        for f in nifti_files:
            try:
                os.remove(f)
            except OSError:
                pass

        print(f"[BOLD] Final merged (float32): {out_path}")
        return out_path

    # ---------------- T1 (single pass) ----------------

    def _convert_t1(self) -> str | None:
        if not Path(self.dcm2niix_path).exists():
            print(f"[T1] dcm2niix not found at {self.dcm2niix_path}")
            return None

        print(f"[T1] Converting DICOMs from {self.input_dir}")
        cmd = [
            self.dcm2niix_path,
            "-o", str(self.output_dir),
            "-f", "T1",
            "-b", "n",
            "-z", "y",
            "-m", "n",
            "-x", "n",
            str(self.input_dir),
        ]
        conv = run(cmd, check=False, capture_output=True, text=True)
        if conv.returncode != 0:
            print("[T1] dcm2niix failed:")
            print(conv.stderr.strip())
            return None

        out = sorted(glob.glob(str(self.output_dir / "T1*.nii.gz")))
        if not out:
            print("[T1] No NIfTI produced.")
            return None

        produced = out[0]
        final_path = self.output_dir / "t1.nii.gz"
        _reorient_to_ras(produced, str(final_path))
        if Path(produced) != final_path:
            try:
                os.remove(produced)
            except OSError:
                pass
        print(f"[T1] Reoriented: {final_path}")
        return str(final_path)


# ---------------------------------------------------------------------------
# Per-patient high-level API
# ---------------------------------------------------------------------------

def convert_patient_dicom(patient_dir: Path | str):
    """Convert BOLD and T1 for a patient if the source DICOM folders exist.

    Returns a dict with output paths for ``bold_4d`` and ``t1`` (``None``
    if the corresponding input was missing or conversion failed).
    """
    patient_dir = Path(patient_dir)
    bold_dcm = patient_dir / "BOLD"
    t1_dcm = patient_dir / "T1"
    out_bold = patient_dir / "PREPROCESSING" / "bold"
    out_t1 = patient_dir / "PREPROCESSING" / "t1"
    out_bold.mkdir(parents=True, exist_ok=True)
    out_t1.mkdir(parents=True, exist_ok=True)

    out: dict[str, str | None] = {"bold_4d": None, "t1": None}

    if bold_dcm.exists():
        b = DicomToNiftiConverter(bold_dcm, out_bold, DCM2NIIX_PATH, modality="BOLD")
        out["bold_4d"] = b.convert()
    else:
        print(f"[INFO] No BOLD dir: {bold_dcm}")

    if t1_dcm.exists():
        t = DicomToNiftiConverter(t1_dcm, out_t1, DCM2NIIX_PATH, modality="T1")
        out["t1"] = t.convert()
    else:
        print(f"[INFO] No T1 dir: {t1_dcm}")

    return out


# ---------------------------------------------------------------------------
# Simple CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Fast DICOM->NIfTI (group-wise) for a single patient directory",
    )
    ap.add_argument("patient_dir", help="Patient directory (containing BOLD/ and/or T1/)")
    ap.add_argument(
        "--group-size",
        type=int,
        default=int(os.getenv("DNB_DICOM_GROUP", "50")),
        help="BOLD DICOM group size (default 50)",
    )
    args = ap.parse_args()

    # Allow CLI override of group size via the env var read by the converter.
    os.environ["DNB_DICOM_GROUP"] = str(max(1, args.group_size))
    convert_patient_dicom(args.patient_dir)
