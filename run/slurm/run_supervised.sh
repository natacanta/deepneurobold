#!/usr/bin/env bash
# =============================================================================
# run_supervised.sh — SLURM submission script for DeepNeuroBOLD v2
# =============================================================================
#
# Usage:
#   sbatch run/slurm/run_supervised.sh <patient_id> <trial_name> [branch]
#
# Example:
#   sbatch run/slurm/run_supervised.sh Patient_01 trial_001 nomcst
#
# =============================================================================
#SBATCH --job-name=dnb2_supervised
#SBATCH --partition=standard
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j-%x.out
#SBATCH --error=logs/slurm-%j-%x.err

set -euo pipefail

# ---- Arguments ----
PATIENT_ID="${1:?Usage: sbatch run_supervised.sh <patient_id> <trial_name> [branch]}"
TRIAL_NAME="${2:?Usage: sbatch run_supervised.sh <patient_id> <trial_name> [branch]}"
BRANCH="${3:-nomcst}"

# ---- Paths (edit to match your server layout) ----
BASE="${DNB_SHARES_ROOT}"
REPO="${BASE}/deepneurobold"
PATIENTS_DIR="${BASE}/<DATA_ROOT>"
CONDA_ENV="${BASE}/<CONDA_ENV_NAME>"
OUTPUT_DIR="${BASE}/results/${TRIAL_NAME}/${PATIENT_ID}"

# ---- Environment ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" logs/

echo "========================================"
echo "DeepNeuroBOLD v2 — Supervised Experiment"
echo "========================================"
echo "Patient   : ${PATIENT_ID}"
echo "Trial     : ${TRIAL_NAME}"
echo "Branch    : ${BRANCH}"
echo "Output    : ${OUTPUT_DIR}"
echo "Node      : $(hostname)"
echo "CPUs      : ${SLURM_CPUS_PER_TASK}"
echo "========================================"

python "${REPO}/run/run_experiment.py" \
    --patient-dir  "${PATIENTS_DIR}/${PATIENT_ID}" \
    --output-dir   "${OUTPUT_DIR}" \
    --branch       "${BRANCH}" \
    --positive     enhancing \
    --classifier   random_forest \
    --cv-folds     3

echo "Done: ${PATIENT_ID} / ${TRIAL_NAME}"
