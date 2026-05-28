"""
experiment.registry
===================
Registry of named experiment presets.

Usage
-----
>>> from deepneurobold.experiment.registry import get_preset, presets
>>> specs = get_preset("only_ts_preprocessed_gridblock_random_forest")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .spec import ExperimentSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_REPRESENTATIONS: Tuple[str, ...] = (
    "ts_raw",
    "ts_normalized",
    "ts_preprocessed",
    "bold_features",
    "full_17",
)

VALID_CLASSIFIERS: Tuple[str, ...] = (
    "svm",
    "logreg",
    "random_forest",
)

VALID_SAMPLING_BASE: Tuple[str, ...] = (
    "patch_holdout",
    "grid_block",
)

_DEFAULT_GRID_BLOCK_MM = 20.0
_DEFAULT_GRID_BUFFER_MM = 5.0

_REGISTRY: Dict[str, List[ExperimentSpec]] = {}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _sampling_base(sampling: str) -> str:
    s = str(sampling).strip()
    return s.split(":", 1)[0].strip().lower() if ":" in s else s.lower()


def _sampling_get_float(sampling: str, key: str) -> Optional[float]:
    s = str(sampling).strip()
    if ":" not in s:
        return None
    for p in s.split(":")[1:]:
        p = p.strip()
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if k.strip().lower() == key.lower():
            return float(v.strip())
    return None


def _make_grid_sampling(block_mm: float, buffer_mm: float) -> str:
    return f"grid_block:block_mm={float(block_mm)}:buffer_mm={float(buffer_mm)}"


def _grid_params_or_defaults(sampling: str) -> Tuple[float, float]:
    block = _sampling_get_float(sampling, "block_mm") or _DEFAULT_GRID_BLOCK_MM
    buff  = _sampling_get_float(sampling, "buffer_mm") or _DEFAULT_GRID_BUFFER_MM
    return float(block), float(buff)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(spec: ExperimentSpec) -> None:
    if not isinstance(spec, ExperimentSpec):
        raise TypeError(f"Expected ExperimentSpec, got {type(spec)}")
    if spec.representation not in VALID_REPRESENTATIONS:
        raise ValueError(f"Invalid representation '{spec.representation}'")
    if spec.classifier not in VALID_CLASSIFIERS:
        raise ValueError(f"Invalid classifier '{spec.classifier}'")
    if spec.positive not in ("enhancing_only", "enhancing_plus_necrosis"):
        raise ValueError(f"Invalid positive: {spec.positive}")
    if spec.negative not in ("healthy_strict", "healthy_controlled"):
        raise ValueError(f"Invalid negative: {spec.negative}")
    if not spec.sampling:
        raise ValueError("ExperimentSpec.sampling is empty")
    base = _sampling_base(str(spec.sampling))
    if base not in VALID_SAMPLING_BASE:
        raise ValueError(f"Invalid sampling '{spec.sampling}'")
    if base == "grid_block":
        block_mm, buffer_mm = _grid_params_or_defaults(str(spec.sampling))
        if block_mm <= 0:
            raise ValueError("grid_block:block_mm must be > 0")
        if buffer_mm < 0:
            raise ValueError("grid_block:buffer_mm must be >= 0")


def _add(name: str, specs: List[ExperimentSpec]) -> None:
    for s in specs:
        _validate(s)
    _REGISTRY[name] = specs


# ---------------------------------------------------------------------------
# Individual presets — patch_holdout
# ---------------------------------------------------------------------------
for _r in VALID_REPRESENTATIONS:
    for _c in VALID_CLASSIFIERS:
        _add(
            f"{_r}_{_c}",
            [ExperimentSpec(
                representation=_r, classifier=_c,
                sampling="patch_holdout",
                positive="enhancing_plus_necrosis",
                negative="healthy_strict", seed=42,
                notes=f"Single run: {_r} with {_c}",
            )],
        )

# ---------------------------------------------------------------------------
# Individual presets — grid_block
# ---------------------------------------------------------------------------
for _r in VALID_REPRESENTATIONS:
    for _c in VALID_CLASSIFIERS:
        _add(
            f"{_r}_{_c}_gridblock",
            [ExperimentSpec(
                representation=_r, classifier=_c,
                sampling=_make_grid_sampling(_DEFAULT_GRID_BLOCK_MM, _DEFAULT_GRID_BUFFER_MM),
                positive="enhancing_plus_necrosis",
                negative="healthy_strict", seed=42,
                notes=f"Single run (grid_block): {_r} with {_c}",
            )],
        )

# ---------------------------------------------------------------------------
# Canonical presets
# ---------------------------------------------------------------------------
_add(
    "only_ts_preprocessed_patchholdout_random_forest",
    [ExperimentSpec(
        representation="ts_preprocessed", classifier="random_forest",
        sampling="patch_holdout",
        positive="enhancing_plus_necrosis", negative="healthy_strict", seed=42,
        notes="Only ts_preprocessed + RandomForest",
    )],
)

_add(
    "only_ts_preprocessed_gridblock_random_forest",
    [ExperimentSpec(
        representation="ts_preprocessed", classifier="random_forest",
        sampling=_make_grid_sampling(_DEFAULT_GRID_BLOCK_MM, _DEFAULT_GRID_BUFFER_MM),
        positive="enhancing_plus_necrosis", negative="healthy_strict", seed=42,
        notes="Only ts_preprocessed + RandomForest (grid_block)",
    )],
)

# ---------------------------------------------------------------------------
# Massive presets — patch_holdout
# ---------------------------------------------------------------------------
for _clf_name, _clf_key in [("svm", "svm"), ("logreg", "logreg"), ("random_forest", "random_forest")]:
    _add(
        f"all_reps_patchholdout_{_clf_key}",
        [ExperimentSpec(
            representation=_r, classifier=_clf_key,
            sampling="patch_holdout",
            positive="enhancing_plus_necrosis", negative="healthy_strict", seed=42,
        ) for _r in VALID_REPRESENTATIONS],
    )

_add(
    "all_reps_patchholdout_svm_logreg_random_forest",
    _REGISTRY["all_reps_patchholdout_svm"]
    + _REGISTRY["all_reps_patchholdout_logreg"]
    + _REGISTRY["all_reps_patchholdout_random_forest"],
)

# ---------------------------------------------------------------------------
# Massive presets — grid_block
# ---------------------------------------------------------------------------
for _clf_key in ("svm", "logreg", "random_forest"):
    _add(
        f"all_reps_gridblock_{_clf_key}",
        [ExperimentSpec(
            representation=_r, classifier=_clf_key,
            sampling=_make_grid_sampling(_DEFAULT_GRID_BLOCK_MM, _DEFAULT_GRID_BUFFER_MM),
            positive="enhancing_plus_necrosis", negative="healthy_strict", seed=42,
        ) for _r in VALID_REPRESENTATIONS],
    )

_add(
    "all_reps_gridblock_svm_logreg_random_forest",
    _REGISTRY["all_reps_gridblock_svm"]
    + _REGISTRY["all_reps_gridblock_logreg"]
    + _REGISTRY["all_reps_gridblock_random_forest"],
)

# ---------------------------------------------------------------------------
# Custom preset (matches bash usage)
# ---------------------------------------------------------------------------
_add(
    "rf_grid_small",
    [ExperimentSpec(
        representation="ts_preprocessed", classifier="random_forest",
        sampling=_make_grid_sampling(12.0, 2.0),
        positive="enhancing_plus_necrosis", negative="healthy_strict", seed=42,
        notes="Custom RF small grid-block",
    )],
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def presets() -> Dict[str, List[ExperimentSpec]]:
    """Return all registered presets."""
    return {k: list(v) for k, v in _REGISTRY.items()}


def get_preset(name: str) -> List[ExperimentSpec]:
    """
    Return the list of :class:`ExperimentSpec` for a named preset.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        available = "\n".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available presets:\n{available}")
    return list(_REGISTRY[name])


REGISTRY = _REGISTRY
