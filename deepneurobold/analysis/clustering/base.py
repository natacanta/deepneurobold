"""
analysis.clustering.base
========================
Abstract base class for unsupervised BOLD clustering pipelines.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

from deepneurobold.analysis.base import BaseAnalysis


class BaseClustering(BaseAnalysis):
    """
    Abstract base for unsupervised BOLD fMRI clustering strategies.

    Parameters
    ----------
    trial_dir : Path
        Trial output directory.
    config : dict, optional
        Runtime configuration.
    """

    @abstractmethod
    def cluster(self) -> Dict[str, Any]:
        """Run the clustering algorithm and return cluster assignments."""

    def analyze(self) -> Dict[str, Any]:
        return self.cluster()
