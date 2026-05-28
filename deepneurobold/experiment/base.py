"""
experiment.base
===============
Abstract base class for experiment runners.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepneurobold.core.base import BaseExperiment


class BaseRunner(BaseExperiment):
    """
    Abstract experiment runner: loads data, trains classifiers, saves outputs.

    Subclasses implement :meth:`setup` and :meth:`run` for specific
    experimental designs (cross-validation, grid search, single-fold, etc.).

    Parameters
    ----------
    patient_dir : Path
        Patient data directory.
    output_dir : Path
        Root directory for experiment outputs.
    spec : ExperimentSpec
        Experiment configuration (classifier, representation, sampling, etc.).
    bold_branch : str
        BOLD processing branch: ``"nomcst"`` or ``"mcst"``.
    config : dict, optional
        Runtime configuration (paths, thread counts, flags).
    """

    def __init__(
        self,
        patient_dir: Path,
        output_dir: Path,
        spec: Any,
        bold_branch: str = "nomcst",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            name=f"Runner[{getattr(spec, 'classifier', 'unknown')}]",
            output_dir=output_dir,
            config=config,
        )
        self.patient_dir = Path(patient_dir)
        self.spec = spec
        self.bold_branch = bold_branch

    @abstractmethod
    def setup(self) -> None:
        """Load BOLD data, masks, and prepare the experiment."""

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute training, inference, and save probability maps."""

    def validate(self) -> bool:
        if not self.patient_dir.exists():
            raise ValueError(f"Patient directory not found: {self.patient_dir}")
        return True
