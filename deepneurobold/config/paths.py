"""Resolve project paths from environment variables.

All filesystem locations are resolved at runtime from environment
variables that each site sets locally.

Required
--------
DNB_DATA_ROOT
    Root of the patient data tree. Expected layout:
        <DNB_DATA_ROOT>/Patient_<ID>/PREPROCESSING/
        <DNB_DATA_ROOT>/Patient_<ID>/T1CE/
        <DNB_DATA_ROOT>/Patient_<ID>/Analysis/

DNB_OUTPUT_ROOT
    Root of the pipeline output tree. All results, logs and figures
    are written under this path.

Optional
--------
DNB_FOLLOWUP_ROOT
    Root of registered post-operative follow-up T1CE volumes.
    Falls back to ``<DNB_OUTPUT_ROOT>/followup`` if not set.

DNB_ENV_PYTHON
    Absolute path to the Python interpreter used by SBATCH launchers.
    Defaults to ``python`` (i.e. the interpreter on PATH).

DNB_SURGERY_DATES_CSV
    CSV of per-patient surgery dates and clinical notes (kept outside
    this repository). See ``run/pick_followup_timepoints.py`` for the
    expected schema.

Setup example
-------------
    export DNB_DATA_ROOT=/path/to/patient/data
    export DNB_OUTPUT_ROOT=/path/to/pipeline/output
    export DNB_FOLLOWUP_ROOT=/path/to/registered/followups
    export DNB_ENV_PYTHON=/opt/envs/dnb/bin/python
    export DNB_SURGERY_DATES_CSV=/path/to/surgery_dates.csv
"""
from __future__ import annotations

import os
from pathlib import Path


def _require_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Environment variable {name} is not set. "
            f"See deepneurobold/config/paths.py for setup instructions."
        )
    return Path(value)


def _optional_env(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value)
    return default


DATA_ROOT = _require_env("DNB_DATA_ROOT")
OUTPUT_ROOT = _require_env("DNB_OUTPUT_ROOT")
FOLLOWUP_ROOT = _optional_env("DNB_FOLLOWUP_ROOT", OUTPUT_ROOT / "followup")
PYTHON_BIN = os.environ.get("DNB_ENV_PYTHON", "python")
SURGERY_DATES_CSV = _optional_env("DNB_SURGERY_DATES_CSV")
