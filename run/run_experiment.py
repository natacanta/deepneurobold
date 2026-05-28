#!/usr/bin/env python3
"""
run_experiment.py
=================
Command-line entry point for running a single supervised experiment.

Usage
-----
.. code-block:: bash

    python run/run_experiment.py \\
        --patient-dir /path/to/Patient_01 \\
        --output-dir  /path/to/results/trial_001 \\
        --branch      nomcst \\
        --positive    enhancing \\
        --classifier  random_forest \\
        --cv-folds    3

SLURM example
-------------
.. code-block:: bash

    sbatch run/slurm/run_supervised.sh Patient_01 trial_001

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepNeuroBOLD v2 — supervised infiltration mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient-dir",  required=True, type=Path, help="Patient data directory")
    parser.add_argument("--output-dir",   required=True, type=Path, help="Output directory for trial results")
    parser.add_argument("--branch",       default="nomcst", choices=["nomcst", "mcst"], help="BOLD processing branch")
    parser.add_argument("--positive",     default="enhancing", help="Positive tissue class")
    parser.add_argument("--classifier",   default="random_forest", choices=["random_forest", "logreg", "svm"])
    parser.add_argument("--representation", default="ts_preprocessed")
    parser.add_argument("--cv-folds",     default=3, type=int)
    parser.add_argument("--seed",         default=42, type=int)
    parser.add_argument("--config",       default=None, type=Path, help="Path to config.py (optional)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from run.config import config_file
    cfg = config_file()

    from deepneurobold.experiment.spec import ExperimentSpec
    from deepneurobold.experiment.runner import ExperimentRunner

    spec = ExperimentSpec(
        positive=args.positive,
        classifier=args.classifier,
        representation=args.representation,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )

    runner = ExperimentRunner(
        patient_dir=args.patient_dir,
        output_dir=args.output_dir,
        spec=spec,
        bold_branch=args.branch,
        config=cfg,
    )
    runner.execute()
    return 0


if __name__ == "__main__":
    sys.exit(main())
