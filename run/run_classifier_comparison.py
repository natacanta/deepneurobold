#!/usr/bin/env python3
"""
run_classifier_comparison.py
============================
CLI entry point for trial_144 (3-classifier head-to-head comparison).

Mirrors run/run_experiment.py: same CLI arguments, except --classifier
is ignored (all three classifiers always run on identical splits).

Usage
-----
.. code-block:: bash

    python run/run_classifier_comparison.py \\
        --patient-dir <DATA_ROOT_PARENT>/Patient_13 \\
        --output-dir  <OUTPUT_ROOT>/runs/Patient_13/trial_144_classifier_comparison/run_003 \\
        --branch      nomcst \\
        --positive    enhancing_plus_necrosis \\
        --cv-folds    4
"""

from __future__ import annotations

from deepneurobold.config.paths import OUTPUT_ROOT
import argparse
import sys
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepNeuroBOLD v2 — trial_144 classifier comparison (RF + SVM + LR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--patient-dir",    required=True, type=Path)
    parser.add_argument("--output-dir",     required=True, type=Path,
                        help="Run directory for ONE channel (e.g., .../run_003)")
    parser.add_argument("--branch",         default="nomcst", choices=["nomcst", "mcst"])
    parser.add_argument("--positive",       default="enhancing_plus_necrosis",
                        help="Positive tissue class")
    parser.add_argument("--negative",       default="healthy_strict",
                        choices=["healthy_strict", "healthy_controlled"])
    parser.add_argument("--representation", default="ts_preprocessed")
    parser.add_argument("--sampling",       default="grid_block:block_mm=20.0:buffer_mm=30.0")
    parser.add_argument("--cv-folds",       default=4, type=int)
    parser.add_argument("--seed",           default=42, type=int)
    parser.add_argument("--classifier",     default="rf",
                        help="IGNORED — all three classifiers always run.")
    parser.add_argument("--config",         default=None, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from deepneurobold.experiment.spec import ExperimentSpec
    from deepneurobold.experiment.classifier_comparison import run_classifier_comparison

    spec = ExperimentSpec(
        positive=args.positive,
        negative=args.negative,
        representation=args.representation,
        sampling=args.sampling,
        classifier="rf",  # placeholder; ignored by run_classifier_comparison
        seed=args.seed,
        cv_folds=args.cv_folds,
        notes=f"trial_144 {args.positive}",
    )

    # Config from --config or inline dict
    cfg = {}
    if args.config is not None:
        try:
            from run.config import config_file
            cfg = config_file(args.config)
        except Exception:
            cfg = {}

    run_classifier_comparison(
        spec=spec,
        patient_dir=args.patient_dir,
        run_dir=args.output_dir,
        bold_branch=args.branch,
        config=cfg,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
