"""
experiment.spec
===============
Experiment specification dataclass — defines a single experimental configuration.

The ``sampling`` field accepts either a simple mode name (``"patch_holdout"``)
or an encoded parameter string: ``"grid_block:block_mm=20:buffer_mm=5"``.
This mirrors the original deepneurobold API exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Types (mirroring deepneurobold API)
# ---------------------------------------------------------------------------
Positive = Literal["enhancing_only", "enhancing_plus_necrosis"]
Representation = Literal[
    "ts_raw",
    "ts_normalized",
    "ts_preprocessed",
    "bold_features",
    "full_17",
]
Negative = Literal["healthy_strict", "healthy_controlled"]

# sampling is a free string — grid parameters encoded in it:
#   "patch_holdout"
#   "grid_block:block_mm=20:buffer_mm=5"
Sampling = str

Classifier = Literal["logreg", "svm", "random_forest"]


@dataclass(frozen=True)
class ExperimentSpec:
    """
    Complete specification for a single supervised experiment.

    Parameters
    ----------
    positive : str
        Tissue class used as positive training examples.
    representation : str
        BOLD feature representation.
    sampling : str
        Sampling mode — ``"patch_holdout"`` or ``"grid_block:block_mm=…:buffer_mm=…"``.
    negative : str
        Tissue class used as negative examples.
    classifier : str
        Classifier to train.
    seed : int
        Random seed.
    cv_folds : int
        Number of cross-validation folds.
    grid_block_mm : float, optional
        Informational — runner parses ``sampling`` string directly.
    grid_buffer_mm : float, optional
        Informational — runner parses ``sampling`` string directly.
    notes : str
        Free-text description.
    """

    positive: Positive
    representation: Representation
    sampling: Sampling
    negative: Negative
    classifier: Classifier

    seed: int = 0
    cv_folds: int = 1

    grid_block_mm: Optional[float] = None
    grid_buffer_mm: Optional[float] = None

    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "positive": self.positive,
            "representation": self.representation,
            "sampling": self.sampling,
            "negative": self.negative,
            "classifier": self.classifier,
            "seed": int(self.seed),
            "cv_folds": int(self.cv_folds),
            "grid_block_mm": (None if self.grid_block_mm is None else float(self.grid_block_mm)),
            "grid_buffer_mm": (None if self.grid_buffer_mm is None else float(self.grid_buffer_mm)),
            "notes": self.notes,
        }

    def key(self) -> Tuple[str, str, str, str, str, int, int]:
        """Unique tuple key for deduplication."""
        return (
            str(self.positive),
            str(self.representation),
            str(self.sampling),
            str(self.negative),
            str(self.classifier),
            int(self.seed),
            int(self.cv_folds),
        )

    def run_name(self) -> str:
        """Directory-safe string identifier for this experiment."""
        return (
            f"pos-{self.positive}"
            f"__rep-{self.representation}"
            f"__samp-{self.sampling}"
            f"__neg-{self.negative}"
            f"__clf-{self.classifier}"
            f"__seed-{int(self.seed)}"
            f"__cv-{int(self.cv_folds)}"
        )
