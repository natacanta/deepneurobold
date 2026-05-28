"""
core.base
=========
Top-level abstract base classes shared across all DeepNeuroBOLD v2 modules.

Every major component (preprocessor, classifier, feature extractor, runner)
inherits from :class:`BaseComponent`, which enforces a minimal logging and
configuration interface.  :class:`BaseExperiment` adds experiment lifecycle
hooks (setup / run / teardown).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class BaseComponent(ABC):
    """
    Minimal interface shared by all DeepNeuroBOLD v2 components.

    Attributes
    ----------
    name : str
        Human-readable identifier used in logs and output filenames.
    config : dict
        Runtime configuration dictionary (paths, hyperparameters, flags).
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.config: Dict[str, Any] = config or {}
        self._logger = logging.getLogger(f"deepneurobold.{name}")

    def log(self, message: str, level: str = "info") -> None:
        """Emit a log message at the requested level."""
        getattr(self._logger, level.lower(), self._logger.info)(f"[{self.name}] {message}")

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that the component is correctly configured.

        Returns
        -------
        bool
            True if all required inputs / parameters are present.

        Raises
        ------
        ValueError
            If a required configuration key is missing or invalid.
        """

    def get_config(self, key: str, default: Any = None) -> Any:
        """Safely retrieve a configuration value by dotted key path."""
        parts = key.split(".")
        cur: Any = self.config
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


class BaseExperiment(BaseComponent):
    """
    Abstract experiment lifecycle: setup → run → teardown.

    Subclasses implement :meth:`setup`, :meth:`run`, and :meth:`teardown`.
    The :meth:`execute` method calls them in order and guarantees teardown
    even if run raises an exception.

    Parameters
    ----------
    name : str
        Experiment identifier (used for output directory naming).
    output_dir : Path
        Root directory where all experiment outputs are written.
    config : dict, optional
        Runtime configuration dictionary.
    """

    def __init__(
        self,
        name: str,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def setup(self) -> None:
        """Prepare resources before the main computation."""

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the experiment.

        Returns
        -------
        dict
            Metrics and output paths produced by the experiment.
        """

    def teardown(self) -> None:
        """Release resources after the experiment (override if needed)."""

    def execute(self) -> Dict[str, Any]:
        """Run the full lifecycle: setup → run → teardown."""
        self.log("Starting experiment setup.")
        self.setup()
        try:
            self.log("Running experiment.")
            results = self.run()
            self.log("Experiment completed successfully.")
            return results
        except Exception:
            self.log("Experiment failed — see traceback above.", level="error")
            raise
        finally:
            self.teardown()

    def validate(self) -> bool:
        return self.output_dir.exists()
