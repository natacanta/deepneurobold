"""
analysis.base
=============
Abstract base class for all analysis pipelines.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from deepneurobold.core.base import BaseComponent


class BaseAnalysis(BaseComponent):
    """
    Abstract interface for post-classification analysis modules.

    All analysis steps (hotspot detection, clustering, proximity visualization)
    inherit from this class and implement :meth:`analyze`.

    Parameters
    ----------
    trial_dir : Path
        Directory containing the outputs of a completed experiment trial.
    config : dict, optional
        Runtime configuration.
    """

    def __init__(self, trial_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=self.__class__.__name__, config=config)
        self.trial_dir = Path(trial_dir)

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Run the analysis and return results.

        Returns
        -------
        dict
            Analysis outputs (paths to saved files, metrics, DataFrames).
        """

    def validate(self) -> bool:
        if not self.trial_dir.exists():
            raise ValueError(f"Trial directory not found: {self.trial_dir}")
        return True
