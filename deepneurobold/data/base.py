"""
data.base
=========
Abstract base class for all data-loading components.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from deepneurobold.core.base import BaseComponent


class BaseDataLoader(BaseComponent):
    """
    Abstract interface for loading neuroimaging data.

    All data loaders in DeepNeuroBOLD v2 (BOLD volumes, segmentation masks,
    feature matrices) inherit from this class and implement :meth:`load`.

    Parameters
    ----------
    patient_dir : Path
        Root directory for the patient whose data is being loaded.
    config : dict, optional
        Runtime configuration.
    """

    def __init__(self, patient_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=self.__class__.__name__, config=config)
        self.patient_dir = Path(patient_dir)

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """
        Load and return the data.

        Returns
        -------
        dict
            Loaded data arrays and metadata (keys depend on subclass).
        """

    def validate(self) -> bool:
        if not self.patient_dir.exists():
            raise ValueError(f"Patient directory not found: {self.patient_dir}")
        return True
