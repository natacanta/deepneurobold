#!/usr/bin/env bash
set -euo pipefail
set -x

patient_name="${1:?Usage: $0 PATIENT_NAME}"

# ============================================================================
# FSL: container-backed wrappers MUST already be in PATH (from your SLURM job)
#   - Do NOT set FSLDIR=/data/${USER}/fsl (that is your old native FSL path)
#   - This script expects: flirt, fslmaths, bet2 (or bet), fslmerge, fslsplit,
#     fslcpgeom, fslchfiletype (all provided by the apptainer wrappers)
# ============================================================================

# Work uncompressed in critical steps; recompress at the end
export FSLOUTPUTTYPE="NIFTI"
NCORES="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${NCORES}"

# Sanity: fail fast if wrappers not found
command -v flirt >/dev/null 2>&1 || { echo "❌ flirt not found in PATH (wrappers not loaded)"; exit 2; }
command -v fslmaths >/dev/null 2>&1 || { echo "❌ fslmaths not found in PATH (wrappers not loaded)"; exit 2; }
if command -v bet2 >/dev/null 2>&1; then
  BETCMD="bet2"
elif command -v bet >/dev/null 2>&1; then
  BETCMD="bet"
else
  echo "❌ bet2/bet not found in PATH (wrappers not loaded)"
  exit 2
fi

# --- paths
PAT="${DNB_DATA_ROOT%/*}/${patient_name}"
BOLD_DIR="${PAT}/PREPROCESSING/bold"
T1_DIR="${PAT}/PREPROCESSING/t1"
mkdir -p "$BOLD_DIR" "$T1_DIR"

T1="${T1_DIR}/t1_reoriented.nii.gz"
MASK="${T1_DIR}/t1_brain_mask.nii.gz"
BOLD4D="${BOLD_DIR}/BOLD.nii.gz"

[ -f "$T1" ] || { echo "❌ Missing $T1"; exit 1; }
[ -f "$BOLD4D" ] || { echo "❌ Missing $BOLD4D"; exit 1; }

# 0) Brain mask on T1 (create if missing)
if [ ! -f "$MASK" ]; then
  $BETCMD "$T1" "${T1_DIR}/t1_brain" -f 0.35
  fslmaths "${T1_DIR}/t1_brain.nii.gz" -bin -fillh "$MASK"
  fslcpgeom "$T1" "$MASK" || true
fi
[ -f "$MASK" ] || { echo "❌ Mask not created: $MASK"; exit 1; }

# 1) Tmean (write as .nii because FSLOUTPUTTYPE=NIFTI)
fslmaths "$BOLD4D" -Tmean "${BOLD_DIR}/BOLD_mean.nii"

# 2) Register mean-BOLD → T1
flirt -in "${BOLD_DIR}/BOLD_mean.nii" -ref "$T1" \
      -omat "${BOLD_DIR}/bold2t1.mat" \
      -out "${BOLD_DIR}/BOLD_mean_inT1.nii" \
      -dof 6 -cost normmi -interp trilinear
[ -s "${BOLD_DIR}/bold2t1.mat" ] || { echo "❌ bold2t1.mat not created"; exit 1; }

# 3) Apply transform to full 4D
FALLBACK=0
if command -v applyxfm4D >/dev/null 2>&1; then
  applyxfm4D "$BOLD4D" "$T1" "${BOLD_DIR}/BOLD_inT1.nii" "${BOLD_DIR}/bold2t1.mat" -singlematrix || FALLBACK=1
else
  FALLBACK=1
fi

# Fallback: split → per-volume flirt in parallel → merge
if [ "$FALLBACK" = "1" ]; then
  fslsplit "$BOLD4D" "${BOLD_DIR}/vol_" -t

  # Parallel per-volume registration
  ls "${BOLD_DIR}"/vol_*.nii.gz | xargs -I{} -P "${NCORES}" bash -lc '
    in="$1"
    out="${in%.nii.gz}_inT1.nii"
    flirt -in "$in" -ref "'"$T1"'" -applyxfm -init "'"${BOLD_DIR}"'/bold2t1.mat" -out "$out" -interp trilinear
  ' _ {}

  fslmerge -t "${BOLD_DIR}/BOLD_inT1.nii" "${BOLD_DIR}"/vol_*_inT1.nii

  # cleanup
  rm -f "${BOLD_DIR}"/vol_*.nii* "${BOLD_DIR}"/vol_*_inT1.nii
fi

# 4) Masking (work in .nii)
fslmaths "${BOLD_DIR}/BOLD_inT1.nii" -mas "$MASK" "${BOLD_DIR}/BOLD_brain_inT1.nii"
fslmaths "${BOLD_DIR}/BOLD_mean_inT1.nii" -mas "$MASK" "${BOLD_DIR}/BOLD_mean_brain_inT1.nii"

# 5) Recompress key outputs to .nii.gz
fslchfiletype NIFTI_GZ "${BOLD_DIR}/BOLD_inT1.nii"
fslchfiletype NIFTI_GZ "${BOLD_DIR}/BOLD_brain_inT1.nii"
fslchfiletype NIFTI_GZ "${BOLD_DIR}/BOLD_mean_inT1.nii"
fslchfiletype NIFTI_GZ "${BOLD_DIR}/BOLD_mean_brain_inT1.nii"

echo "✅ DONE: ${patient_name}"