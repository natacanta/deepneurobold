"""
experiment.runner
=================
Supervised experiment runner — orchestrates data loading, training, inference,
output saving, and hotspot detection for a single patient.

Entry point
-----------
>>> from deepneurobold.experiment.runner import run_one_experiment
>>> run_one_experiment(spec, patient_dir, run_dir, bold_branch="nomcst", config=cfg)
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.manifold import MDS

from .base import BaseRunner
from .spec import ExperimentSpec

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from deepneurobold.data.loader import load_bold_and_brain_mask
from deepneurobold.data.masks import load_segmentation_masks, build_train_masks
from deepneurobold.data.sampling import make_split
from deepneurobold.features.bold_features import extract_features
from deepneurobold.classifiers.random_forest import train_random_forest, predict_proba as predict_proba_rf, get_feature_importance, get_sample_proximity
from deepneurobold.classifiers.logistic_regression import train_logreg, predict_proba as predict_proba_logreg
from deepneurobold.classifiers.svm import train_svm_linear, predict_proba as predict_proba_svm

# MDS threshold above which t-SNE replaces MDS (MDS is O(n^2) memory)
_MDS_MAX_SAMPLES = 2000

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config_obj(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, (str, Path)):
        p = Path(cfg)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    if isinstance(cfg, dict):
        return dict(cfg)
    return {}


def _cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    def _first(v: Any) -> Any:
        return v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v

    if key in cfg:
        return _first(cfg[key])
    if "." not in key:
        return default
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return _first(cur)


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

def _normalize_classifier_name(x: Any) -> str:
    c = str(x).lower().strip()
    if c in {"rf", "random_forest", "randomforest"}:
        return "rf"
    if c in {"logreg", "logistic", "logistic_regression"}:
        return "logreg"
    if c in {"svm", "svc", "svm_linear", "linear_svm"}:
        return "svm"
    raise ValueError(f"Unknown classifier: {x}")


def _train_model_any(
    *,
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_scaler: bool,
    seed: int,
    rf_n_estimators: int = 500,
    rf_max_depth: Optional[int] = None,
    rf_min_samples_leaf: int = 1,
    rf_max_features: str = "sqrt",
) -> Any:
    c = _normalize_classifier_name(classifier_name)
    if c == "rf":
        return train_random_forest(
            X_train=X_train, y_train=y_train, use_scaler=use_scaler, seed=seed,
            n_estimators=rf_n_estimators, max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf, max_features=rf_max_features,
        )
    if c == "logreg":
        return train_logreg(X_train=X_train, y_train=y_train, use_scaler=use_scaler, seed=seed)
    if c == "svm":
        return train_svm_linear(X_train=X_train, y_train=y_train, use_scaler=use_scaler, seed=seed)
    raise ValueError(f"Unknown classifier: {classifier_name}")


def _predict_proba_any(classifier_name: str, model: Any, X: np.ndarray) -> np.ndarray:
    c = _normalize_classifier_name(classifier_name)
    if c == "logreg":
        return predict_proba_logreg(model, X)
    if c == "svm":
        return predict_proba_svm(model, X)
    if c == "rf":
        return predict_proba_rf(model, X)
    raise ValueError(f"Unknown classifier: {classifier_name}")


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _normalize_sampling_mode(sampling: Any) -> Tuple[str, str]:
    sampling_raw = str(sampling).strip().lower()
    sampling_mode = sampling_raw.split(":", 1)[0].strip()
    return sampling_raw, sampling_mode


def _sampling_get_float(sampling: str, key: str) -> Optional[float]:
    if ":" not in sampling:
        return None
    for p in sampling.split(":")[1:]:
        p = p.strip()
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if k.strip().lower() == key.lower():
            return float(v.strip())
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_val_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    p_pos = np.asarray(p_pos, dtype=np.float32).ravel()
    nan = float("nan")
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return dict(val_auc=nan, val_ap=nan, val_bal_acc=nan, val_brier=nan,
                    val_mean_prob_pos=nan, val_mean_prob_neg=nan)
    return {
        "val_auc":           float(roc_auc_score(y_true, p_pos)),
        "val_ap":            float(average_precision_score(y_true, p_pos)),
        "val_bal_acc":       float(balanced_accuracy_score(y_true, (p_pos >= 0.5).astype(np.int64))),
        "val_brier":         float(brier_score_loss(y_true, p_pos)),
        "val_mean_prob_pos": float(np.mean(p_pos[y_true == 1])),
        "val_mean_prob_neg": float(np.mean(p_pos[y_true == 0])),
    }


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def _save_mask_nifti(
    *,
    out_path: Path,
    mask_flat_bool: np.ndarray,
    vol_shape_3d: Tuple[int, int, int],
    ref_img: Any,
    dtype: type = np.uint8,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.asarray(mask_flat_bool, dtype=bool).ravel().reshape(vol_shape_3d).astype(dtype, copy=False)
    nib.save(nib.Nifti1Image(vol, ref_img.affine, ref_img.header), str(out_path))
    print(f"[mask-nifti] wrote: {out_path}  true={int(np.count_nonzero(vol))}")


def _save_float_nifti(
    *,
    out_path: Path,
    data_flat: np.ndarray,
    vol_shape_3d: Tuple[int, int, int],
    ref_img: Any,
    dtype: type = np.float32,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vol = np.asarray(data_flat).ravel().astype(dtype, copy=False).reshape(vol_shape_3d)
    nib.save(nib.Nifti1Image(vol, ref_img.affine, ref_img.header), str(out_path))
    print(f"[float-nifti] wrote: {out_path}")


def _save_prob_map(
    run_dir: Path,
    prob_flat: np.ndarray,
    vol_shape_3d: Tuple[int, int, int],
    ref_img: Any,
    name: str = "prob_tumor_like.nii.gz",
) -> Path:
    """Save a probability volume as NIfTI."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / name
    _save_float_nifti(out_path=out_path, data_flat=prob_flat,
                      vol_shape_3d=vol_shape_3d, ref_img=ref_img)
    return out_path


# ---------------------------------------------------------------------------
# Split caching
# ---------------------------------------------------------------------------

def _split_cache_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()  # noqa: S324


def _maybe_load_or_create_split(
    *,
    cache_dir: Path,
    split_def: Dict[str, Any],
    make_split_fn: Any,
    make_split_kwargs: Dict[str, Any],
) -> Tuple[Any, str]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _split_cache_key(split_def)
    npz_path = cache_dir / f"split_{key}.npz"
    meta_path = cache_dir / f"split_{key}.json"

    if npz_path.exists() and meta_path.exists():
        data = np.load(npz_path, allow_pickle=False)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        class _Split:
            pass

        split = _Split()
        split.train_idx = data["train_idx"]
        split.val_idx   = data["val_idx"]
        split.train_y   = data["train_y"]
        split.val_y     = data["val_y"]
        split.meta      = meta.get("split_meta", {})
        print(f"\n[split-cache] loaded: {npz_path.name}")
        return split, "cached"

    split = make_split_fn(**make_split_kwargs)
    np.savez_compressed(
        npz_path,
        train_idx=np.asarray(split.train_idx, dtype=np.int64).ravel(),
        val_idx=np.asarray(split.val_idx,   dtype=np.int64).ravel(),
        train_y=np.asarray(split.train_y,   dtype=np.int64).ravel(),
        val_y=np.asarray(split.val_y,       dtype=np.int64).ravel(),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "split_def": split_def,
            "split_meta": (split.meta if hasattr(split, "meta") and isinstance(split.meta, dict) else {}),
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, sort_keys=True)
    print(f"\n[split-cache] created: {npz_path.name}")
    return split, "fresh"


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _append_csv_row(path: Path, header: List[str], row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def _safe_str(v: Any) -> str:
    """Convert any value to a safe string for CSV output."""
    if v is None:
        return ""
    return str(v)


def _trial_name_from_run_dir(run_dir: Path) -> str:
    """Extract trial name from run directory path (last two path components)."""
    run_dir = Path(run_dir)
    parts = run_dir.parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else run_dir.name


def _write_run_summary(
    *,
    run_dir: Path,
    patient_dir: Path,
    spec: Any,
    method_name: str,
    bold_branch: str,
    bold_source: str,
    sampling_raw: str,
    sampling_mode_requested: str,
    sampling_mode_effective: str,
    split_source: str,
    n_per_class: int,
    holdout_frac: float,
    patch_half_width: int,
    grid_block_mm: float,
    grid_buffer_mm: float,
    fold_id: int,
    fold_seed: int,
    metrics: Dict[str, Any],
) -> None:
    """
    Append one row per fold to RunSummary.csv.

    Mirrors original deepneurobold runner output format.
    """
    out_csv = Path(run_dir) / "RunSummary.csv"
    trial = _trial_name_from_run_dir(run_dir)
    patient = Path(patient_dir).name

    header = [
        "trial", "patient", "representation", "method",
        "fold_id", "fold_seed", "bold_branch", "bold_source",
        "sampling_raw", "sampling_mode_requested", "sampling_mode_effective",
        "split_source", "n_per_class", "holdout_frac", "patch_half_width",
        "grid_block_mm", "grid_buffer_mm",
        "n_train_samples", "n_val_samples",
        "val_auc", "val_ap", "val_bal_acc@0p5", "val_brier",
        "val_mean_prob_pos", "val_mean_prob_neg",
        "edema_n_voxels", "edema_mean_prob", "edema_std_prob",
        "edema_pct_tumor_like_0.5", "edema_pct_healthy_like_0.5",
        "created_at", "run_dir",
    ]

    row: Dict[str, Any] = {
        "trial":                     trial,
        "patient":                   patient,
        "representation":            _safe_str(spec.representation),
        "method":                    method_name,
        "fold_id":                   int(fold_id),
        "fold_seed":                 int(fold_seed),
        "bold_branch":               _safe_str(bold_branch),
        "bold_source":               _safe_str(bold_source),
        "sampling_raw":              _safe_str(sampling_raw),
        "sampling_mode_requested":   _safe_str(sampling_mode_requested),
        "sampling_mode_effective":   _safe_str(sampling_mode_effective),
        "split_source":              _safe_str(split_source),
        "n_per_class":               int(n_per_class),
        "holdout_frac":              float(holdout_frac),
        "patch_half_width":          int(patch_half_width),
        "grid_block_mm":             float(grid_block_mm),
        "grid_buffer_mm":            float(grid_buffer_mm),
        "n_train_samples":           float(metrics.get("n_train_samples", float("nan"))),
        "n_val_samples":             float(metrics.get("n_val_samples", float("nan"))),
        "val_auc":                   float(metrics.get("val_auc", float("nan"))),
        "val_ap":                    float(metrics.get("val_ap", float("nan"))),
        "val_bal_acc@0p5":           float(metrics.get("val_bal_acc", float("nan"))),
        "val_brier":                 float(metrics.get("val_brier", float("nan"))),
        "val_mean_prob_pos":         float(metrics.get("val_mean_prob_pos", float("nan"))),
        "val_mean_prob_neg":         float(metrics.get("val_mean_prob_neg", float("nan"))),
        "edema_n_voxels":            float(metrics.get("edema_n_voxels", float("nan"))),
        "edema_mean_prob":           float(metrics.get("edema_mean_prob", float("nan"))),
        "edema_std_prob":            float(metrics.get("edema_std_prob", float("nan"))),
        "edema_pct_tumor_like_0.5":  float(metrics.get("edema_pct_tumor_like_0.5", float("nan"))),
        "edema_pct_healthy_like_0.5": float(metrics.get("edema_pct_healthy_like_0.5", float("nan"))),
        "created_at":                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir":                   str(Path(run_dir).resolve()),
    }
    _append_csv_row(out_csv, header, row)
    print(f"[csv] RunSummary row appended: fold_id={fold_id}  val_auc={row['val_auc']:.4f}")


def _write_run_aggregate(
    *,
    run_dir: Path,
    patient_dir: Path,
    spec: Any,
    method_name: str,
    cv_folds: int,
    sampling_mode_requested: str,
    fold_metrics: List[Dict[str, Any]],
) -> None:
    """
    Write aggregated metrics (mean ± std across folds) to RunAggregate.csv.

    Includes val_brier_mean / val_brier_std.
    """
    out_csv = Path(run_dir) / "RunAggregate.csv"
    trial = _trial_name_from_run_dir(run_dir)
    patient = Path(patient_dir).name

    def _nanmean(key: str) -> float:
        vals = [float(m[key]) for m in fold_metrics if key in m and not np.isnan(float(m.get(key, float("nan"))))]
        return float(np.nanmean(vals)) if vals else float("nan")

    def _nanstd(key: str) -> float:
        vals = [float(m[key]) for m in fold_metrics if key in m and not np.isnan(float(m.get(key, float("nan"))))]
        return float(np.nanstd(vals)) if vals else float("nan")

    header = [
        "trial", "patient", "representation", "method",
        "cv_folds", "sampling_mode_requested",
        "val_auc_mean", "val_auc_std",
        "val_ap_mean", "val_ap_std",
        "val_bal_acc@0p5_mean", "val_bal_acc@0p5_std",
        "val_brier_mean", "val_brier_std",
        "edema_n_voxels_mean", "edema_n_voxels_std",
        "edema_mean_prob_mean", "edema_mean_prob_std",
        "edema_pct_tumor_like_0.5_mean", "edema_pct_tumor_like_0.5_std",
        "edema_pct_healthy_like_0.5_mean", "edema_pct_healthy_like_0.5_std",
        "created_at", "run_dir",
    ]

    row: Dict[str, Any] = {
        "trial":                          trial,
        "patient":                        patient,
        "representation":                 _safe_str(spec.representation),
        "method":                         method_name,
        "cv_folds":                       int(cv_folds),
        "sampling_mode_requested":        _safe_str(sampling_mode_requested),
        "val_auc_mean":                   _nanmean("val_auc"),
        "val_auc_std":                    _nanstd("val_auc"),
        "val_ap_mean":                    _nanmean("val_ap"),
        "val_ap_std":                     _nanstd("val_ap"),
        "val_bal_acc@0p5_mean":           _nanmean("val_bal_acc"),
        "val_bal_acc@0p5_std":            _nanstd("val_bal_acc"),
        "val_brier_mean":                 _nanmean("val_brier"),
        "val_brier_std":                  _nanstd("val_brier"),
        "edema_n_voxels_mean":            _nanmean("edema_n_voxels"),
        "edema_n_voxels_std":             _nanstd("edema_n_voxels"),
        "edema_mean_prob_mean":           _nanmean("edema_mean_prob"),
        "edema_mean_prob_std":            _nanstd("edema_mean_prob"),
        "edema_pct_tumor_like_0.5_mean":  _nanmean("edema_pct_tumor_like_0.5"),
        "edema_pct_tumor_like_0.5_std":   _nanstd("edema_pct_tumor_like_0.5"),
        "edema_pct_healthy_like_0.5_mean": _nanmean("edema_pct_healthy_like_0.5"),
        "edema_pct_healthy_like_0.5_std":  _nanstd("edema_pct_healthy_like_0.5"),
        "created_at":                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir":                        str(Path(run_dir).resolve()),
    }
    _append_csv_row(out_csv, header, row)
    print(f"[csv] RunAggregate written: val_auc={row['val_auc_mean']:.4f} ± {row['val_auc_std']:.4f}")


def _save_split_audit_files(
    *,
    fold_dir: Path,
    n_vox: int,
    split: Any,
    train_mask_flat: np.ndarray,
    val_mask_flat: np.ndarray,
    brain_flat: np.ndarray,
) -> None:
    """
    Save split index and mask arrays for full traceability.

    Mirrors original deepneurobold runner behaviour.
    """
    fold_dir = Path(fold_dir)
    train_idx = np.asarray(split.train_idx, dtype=np.int64).ravel()
    val_idx   = np.asarray(split.val_idx,   dtype=np.int64).ravel()
    np.save(fold_dir / "train_indices_flat.npy", train_idx)
    np.save(fold_dir / "val_indices_flat.npy",   val_idx)
    np.save(fold_dir / "train_mask_flat.npy", np.asarray(train_mask_flat, dtype=bool).ravel())
    np.save(fold_dir / "val_mask_flat.npy",   np.asarray(val_mask_flat,   dtype=bool).ravel())
    np.save(fold_dir / "brain_flat.npy",      np.asarray(brain_flat,      dtype=bool).ravel())
    y_train = np.asarray(split.train_y, dtype=np.int64).ravel()
    y_val   = np.asarray(split.val_y,   dtype=np.int64).ravel()
    np.save(fold_dir / "y_train.npy", y_train)
    np.save(fold_dir / "y_val.npy",   y_val)
    print(
        f"[split-audit] train_idx={train_idx.shape}  val_idx={val_idx.shape}  "
        f"y_train pos={int(y_train.sum())} neg={int(y_train.size - y_train.sum())}  "
        f"y_val pos={int(y_val.sum())} neg={int(y_val.size - y_val.sum())}"
    )


# ---------------------------------------------------------------------------
# Branch env
# ---------------------------------------------------------------------------

def _set_branch_env(branch: str) -> str:
    branch = str(branch).strip().lower()
    if branch not in {"nomcst", "mcst"}:
        raise ValueError(f"Invalid bold branch: {branch}. Expected 'nomcst' or 'mcst'.")
    os.environ["DNB_BOLD_BRANCH"] = branch
    print(f"[branch] DNB_BOLD_BRANCH set to '{branch}'")
    return branch


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _mask_from_idx(n: int, idx: np.ndarray) -> np.ndarray:
    m = np.zeros((n,), dtype=bool)
    if idx.size > 0:
        m[idx.astype(np.int64, copy=False)] = True
    return m


# ---------------------------------------------------------------------------
# Adaptive split helpers
# ---------------------------------------------------------------------------

def _runtime_error_is_empty_val_pool(e: BaseException) -> bool:
    s = str(e).lower()
    return ("empty validation pool" in s) or ("produced empty validation pool" in s)


def _runtime_error_is_empty_class_pool(e: BaseException) -> bool:
    s = str(e).lower()
    return ("empty class pool" in s) or ("cannot sample: empty class pool" in s)


def _adaptive_make_split_cached(
    *,
    cache_dir: Path,
    split_def_base: Dict[str, Any],
    make_split_fn: Any,
    make_split_kwargs_base: Dict[str, Any],
    sampling_mode: str,
    voxel_size_mm: Tuple[float, float, float],
    grid_block_mm_init: float,
    grid_buffer_mm_init: float,
    holdout_frac_init: float,
    patch_half_width_init: int,
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Attempt to create a train/val split with automatic fallback.

    For grid_block: tries progressively smaller block_mm and buffer_mm
    combinations. If all fail, falls back to patch_holdout with multiple
    holdout_frac / patch_half_width combinations.

    Returns (split, source, used_params_dict).
    """
    used: Dict[str, Any] = {"final_mode": sampling_mode}

    def _try_once(split_def: Dict[str, Any], make_kwargs: Dict[str, Any]) -> Tuple[Any, str]:
        return _maybe_load_or_create_split(
            cache_dir=cache_dir,
            split_def=split_def,
            make_split_fn=make_split_fn,
            make_split_kwargs=make_kwargs,
        )

    if sampling_mode == "grid_block":
        block_cands: List[float] = []
        for v in [grid_block_mm_init, 18.0, 15.0, 12.0, 10.0, 8.0]:
            if float(v) not in block_cands:
                block_cands.append(float(v))

        buffer_cands: List[float] = []
        for v in [grid_buffer_mm_init, 4.0, 3.0, 2.0, 1.0, 0.0]:
            if float(v) not in buffer_cands:
                buffer_cands.append(float(v))

        last_err: Optional[Exception] = None
        for block_mm in block_cands:
            for buffer_mm in buffer_cands:
                sd = {**split_def_base,
                      "sampling_mode": "grid_block",
                      "grid_block_mm": float(block_mm),
                      "grid_buffer_mm": float(buffer_mm),
                      "voxel_size_mm": list(voxel_size_mm)}
                mk = {**make_split_kwargs_base,
                      "mode": "grid_block",
                      "voxel_size_mm": voxel_size_mm,
                      "grid_block_mm": float(block_mm),
                      "grid_buffer_mm": float(buffer_mm)}
                print(f"\n[split-adapt] trying grid_block: block_mm={block_mm}  buffer_mm={buffer_mm}")
                try:
                    split, src = _try_once(sd, mk)
                    used.update({"final_mode": "grid_block",
                                 "grid_block_mm": block_mm,
                                 "grid_buffer_mm": buffer_mm})
                    return split, src, used
                except RuntimeError as e:
                    last_err = e
                    if _runtime_error_is_empty_val_pool(e):
                        print(f"[split-adapt] empty val pool — continuing...")
                        continue
                    if _runtime_error_is_empty_class_pool(e):
                        raise
                    print(f"[split-adapt] RuntimeError: {e} — continuing...")
                    continue

        print("\n[split-adapt] grid_block exhausted — fallback to patch_holdout.")
        if last_err is not None:
            print(f"[split-adapt] last error: {last_err}")
        sampling_mode = "patch_holdout"

    # patch_holdout fallback
    frac_cands: List[float] = []
    for v in [holdout_frac_init, 0.02, 0.05, 0.08, 0.10]:
        if float(v) not in frac_cands:
            frac_cands.append(float(v))

    phw_cands: List[int] = []
    for v in [patch_half_width_init, 1, 0]:
        if int(v) not in phw_cands:
            phw_cands.append(int(v))

    last_err2: Optional[Exception] = None
    for frac in frac_cands:
        for phw in phw_cands:
            sd = {**split_def_base,
                  "sampling_mode": "patch_holdout",
                  "holdout_frac": float(frac),
                  "patch_half_width": int(phw)}
            mk = {**make_split_kwargs_base,
                  "mode": "patch_holdout",
                  "holdout_frac": float(frac),
                  "patch_half_width": int(phw)}
            print(f"\n[split-adapt] trying patch_holdout: holdout_frac={frac}  patch_half_width={phw}")
            try:
                split, src = _try_once(sd, mk)
                used.update({"final_mode": "patch_holdout",
                             "holdout_frac": frac,
                             "patch_half_width": phw})
                return split, src, used
            except RuntimeError as e:
                last_err2 = e
                if _runtime_error_is_empty_val_pool(e) or _runtime_error_is_empty_class_pool(e):
                    print(f"[split-adapt] patch_holdout failed: {e} — continuing...")
                    continue
                raise

    raise RuntimeError(
        f"All adaptive sampling attempts failed. Last error: {last_err2}"
    )


# ---------------------------------------------------------------------------
# EPI ghost detection
# ---------------------------------------------------------------------------

def _write_epi_ghost_outputs(
    run_dir: Path,
    report: Dict[str, Any],
    mean_profile: Optional[np.ndarray],
) -> None:
    """Save EPI ghost JSON report and signal-profile PNG."""
    run_dir = Path(run_dir)

    def _jsonify(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report_path = run_dir / "epi_ghost_report.json"
    safe_report = {k: _jsonify(v) for k, v in report.items()}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=2, sort_keys=True)
    report["report_path"] = str(report_path)

    plot_path = run_dir / "epi_ghost_profile.png"
    if mean_profile is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            n_slices = len(mean_profile)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(n_slices), mean_profile, color="steelblue", linewidth=1.5, label="Mean signal profile")
            threshold = report.get("detection_threshold")
            if threshold is not None:
                ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
                           label=f"Ghost threshold ({threshold:.1f})")
            for idx in report.get("band_indices", []):
                ax.axvline(idx, color="orange", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.set_xlabel(f"Slice index (axis {report.get('phase_encode_axis', '?')})")
            ax.set_ylabel("Mean BOLD signal")
            ax.set_title(
                f"EPI ghost profile — {len(report.get('band_indices', []))} band(s) detected\n"
                f"patient: {Path(report.get('patient_dir', '')).name}"
            )
            ax.legend(fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"[epi-ghost] plot skipped: {e}")
    report["plot_path"] = str(plot_path)
    print(f"[epi-ghost] report: {report_path}")
    print(f"[epi-ghost] profile plot: {plot_path}")


def _detect_epi_ghost_artifacts(
    *,
    patient_dir: Path,
    run_dir: Path,
    bold_4d: Any,
    ref_img: Any,
    phase_encode_axis: int = 1,
    ghost_band_halfwidth: int = 3,
    n_top_bands: int = 5,
) -> Dict[str, Any]:
    """
    Detect EPI Nyquist ghost artifacts in the mean BOLD volume.

    EPI ghost artifacts appear as bands of elevated signal displaced by N/2
    voxels from the true anatomy along the phase-encode direction. This function:

      1. Computes the temporal mean volume from the loaded BOLD data.
      2. Computes the mean signal profile along the phase-encode axis.
      3. Detects bands whose mean intensity exceeds mean + 2*std (heuristic
         marker of ghosting).
      4. Saves epi_ghost_report.json and epi_ghost_profile.png.

    This is a detection-only step: it does not modify training or inference.

    Parameters
    ----------
    patient_dir : Path
    run_dir : Path
    bold_4d : nibabel data proxy (already loaded)
    ref_img : nibabel image (for patient name in report)
    phase_encode_axis : int
        Axis along which to compute signal profile (0=x, 1=y, 2=z).
    ghost_band_halfwidth : int
        Half-width in slices for local peak measurement.
    n_top_bands : int
        Maximum number of bands to report.

    Returns
    -------
    dict with keys: n_bands_detected, band_indices, band_peak_ratios, ...
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_dir": str(Path(patient_dir).resolve()),
        "phase_encode_axis": int(phase_encode_axis),
        "ghost_band_halfwidth": int(ghost_band_halfwidth),
        "n_bands_detected": 0,
        "band_indices": [],
        "band_peak_ratios": [],
        "warnings": [],
    }

    try:
        bold_data = np.asarray(bold_4d, dtype=np.float32)
        if bold_data.ndim != 4:
            report["warnings"].append(f"BOLD data is not 4-D (shape={bold_data.shape}). Skipping.")
            _write_epi_ghost_outputs(run_dir, report, mean_profile=None)
            return report

        mean_vol = np.mean(bold_data, axis=-1)  # (x, y, z)
        axes_to_avg = tuple(ax for ax in range(3) if ax != phase_encode_axis)
        profile = np.mean(mean_vol, axis=axes_to_avg)  # (n_slices,)

        profile_mean = float(np.mean(profile))
        profile_std  = float(np.std(profile))
        threshold    = profile_mean + 2.0 * profile_std

        report["mean_profile_mean"]    = profile_mean
        report["mean_profile_std"]     = profile_std
        report["detection_threshold"]  = threshold

        n_slices = len(profile)
        band_indices: List[int] = []
        band_peak_ratios: List[float] = []

        i = ghost_band_halfwidth
        while i < n_slices - ghost_band_halfwidth:
            local = profile[max(0, i - ghost_band_halfwidth): i + ghost_band_halfwidth + 1]
            local_peak = float(np.max(local))
            if local_peak > threshold:
                band_indices.append(int(i))
                band_peak_ratios.append(round(local_peak / (profile_mean + 1e-9), 4))
                i += ghost_band_halfwidth * 2
            else:
                i += 1

        if len(band_indices) > n_top_bands:
            order = np.argsort(band_peak_ratios)[::-1][:n_top_bands]
            band_indices    = [band_indices[j] for j in order]
            band_peak_ratios = [band_peak_ratios[j] for j in order]

        report["n_bands_detected"] = len(band_indices)
        report["band_indices"]     = band_indices
        report["band_peak_ratios"] = band_peak_ratios

        if band_indices:
            report["warnings"].append(
                f"Possible EPI ghost bands detected at axis-{phase_encode_axis} slices "
                f"{band_indices} (peak/mean ratios: {band_peak_ratios}). "
                "Visual inspection recommended."
            )

        _write_epi_ghost_outputs(run_dir, report, mean_profile=profile)

        print("\n[epi-ghost] artifact check:")
        print(f"  phase_encode_axis: {phase_encode_axis}")
        print(f"  n_bands_detected: {len(band_indices)}")
        if band_indices:
            print(f"  band_indices: {band_indices}")
            print(f"  band_peak_ratios: {band_peak_ratios}")
        for w in report["warnings"]:
            print(f"  WARNING: {w}")

    except Exception as e:
        report["warnings"].append(f"EPI ghost check failed: {e}")
        print(f"[epi-ghost] check failed (non-fatal): {e}")
        _write_epi_ghost_outputs(run_dir, report, mean_profile=None)

    return report


# ---------------------------------------------------------------------------
# Proximity / MDS
# ---------------------------------------------------------------------------

def _proximity_mds_analysis(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    tag: str,
    max_samples: int = 999999,
    seed: int = 42,
    classifier_name: str = "rf",
) -> Dict[str, Any]:
    """
    Compute Random Forest proximity on training samples and reduce to 2-D.

    - Balanced per-class subsampling (max_samples per class).
    - t-SNE fallback (metric="precomputed") when N > _MDS_MAX_SAMPLES.
    - Richer plot with legend + class counts.
    - Saves proximity_info_{tag}.json.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c = _normalize_classifier_name(classifier_name)
    if c != "rf":
        return {"status": "skipped", "reason": "not RF"}

    try:
        y = np.asarray(y, dtype=np.int64).ravel()
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        n_pos_available = len(pos_idx)
        n_neg_available = len(neg_idx)

        n_pos_sample = min(n_pos_available, max_samples)
        n_neg_sample = min(n_neg_available, max_samples)

        if n_pos_sample == 0 or n_neg_sample == 0:
            print(
                f"[proximity-{tag}] Not enough positive or negative samples: "
                f"pos={n_pos_available}, neg={n_neg_available}. Skipping."
            )
            return {"status": "skipped", "n_pos": n_pos_available, "n_neg": n_neg_available}

        rng = np.random.RandomState(seed)
        chosen_pos = rng.choice(pos_idx, size=n_pos_sample, replace=False)
        chosen_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)
        sub_idx = np.concatenate([chosen_pos, chosen_neg])
        X_sub = X[sub_idx]
        y_sub = y[sub_idx]
        n_total = len(sub_idx)

        print(
            f"[proximity-{tag}] Computing proximity matrix for {n_total} samples "
            f"(pos={n_pos_sample} of {n_pos_available}, neg={n_neg_sample} of {n_neg_available})..."
        )
        prox = get_sample_proximity(model, X_sub)
        dist = 1.0 - prox
        np.fill_diagonal(dist, 0.0)

        if n_total > _MDS_MAX_SAMPLES:
            print(
                f"[proximity-{tag}] {n_total} samples exceeds MDS threshold ({_MDS_MAX_SAMPLES}). "
                "Using t-SNE instead."
            )
            try:
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=2,
                    metric="precomputed",
                    init="random",
                    random_state=seed,
                    perplexity=min(30, n_total - 1),
                    n_iter=1000,
                    verbose=0,
                )
                method_label = "t-SNE"
            except ImportError:
                print(f"[proximity-{tag}] TSNE not available; falling back to MDS with subsampling.")
                n_sub = _MDS_MAX_SAMPLES // 2
                sub2_pos = rng.choice(np.where(y_sub == 1)[0], size=min(n_sub, n_pos_sample), replace=False)
                sub2_neg = rng.choice(np.where(y_sub == 0)[0], size=min(n_sub, n_neg_sample), replace=False)
                sub2_idx = np.concatenate([sub2_pos, sub2_neg])
                dist = dist[np.ix_(sub2_idx, sub2_idx)]
                y_sub = y_sub[sub2_idx]
                print(f"[proximity-{tag}] MDS fallback with {len(sub2_idx)} samples.")
                reducer = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, n_init=2, max_iter=200)
                method_label = "MDS"
        else:
            reducer = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, n_init=4, max_iter=300)
            method_label = "MDS"

        print(f"[proximity-{tag}] Running {method_label}...")
        coords = reducer.fit_transform(dist)

        np.save(out_dir / f"proximity_mds_coords_{tag}.npy", coords.astype(np.float32))
        np.save(out_dir / f"proximity_labels_{tag}.npy", y_sub.astype(np.int64))

        # Plot with legend and class counts
        try:
            plt.figure(figsize=(8, 6))
            colors = ["red" if lbl == 1 else "blue" for lbl in y_sub]
            plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.6, s=10)
            plt.title(
                f"RF Proximity {method_label} ({tag}) — {len(y_sub)} samples\n"
                f"tumor: {int((y_sub==1).sum())} of {n_pos_available}  "
                f"healthy: {int((y_sub==0).sum())} of {n_neg_available}"
            )
            plt.xlabel(f"{method_label} dim 1")
            plt.ylabel(f"{method_label} dim 2")
            plt.legend(handles=[
                Patch(facecolor="red",  label=f"Tumor (n={int((y_sub==1).sum())})"),
                Patch(facecolor="blue", label=f"Healthy (n={int((y_sub==0).sum())})"),
            ])
            plt.tight_layout()
            plot_path = out_dir / f"proximity_mds_{tag}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"[proximity-{tag}] plot skipped: {e}")
            plot_path = None

        info: Dict[str, Any] = {
            "tag": tag,
            "method": method_label,
            "n_total": int(len(y_sub)),
            "n_pos": int((y_sub == 1).sum()),
            "n_neg": int((y_sub == 0).sum()),
            "n_pos_available": int(n_pos_available),
            "n_neg_available": int(n_neg_available),
            "max_samples_per_class": int(max_samples),
            "plot": str(plot_path) if plot_path else None,
            "coords_file": str(out_dir / f"proximity_mds_coords_{tag}.npy"),
            "labels_file": str(out_dir / f"proximity_labels_{tag}.npy"),
        }
        with open(out_dir / f"proximity_info_{tag}.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"[proximity-{tag}] Done ({method_label}). Plot saved to {plot_path}")
        return info

    except Exception as e:
        print(f"[proximity-{tag}] skipped (non-fatal): {e}")
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Core experiment implementation
# ---------------------------------------------------------------------------

def _run_one_experiment_single(
    spec: ExperimentSpec,
    patient_dir: Path,
    run_dir: Path,
    *,
    bold_branch: str,
    config: Optional[Any] = None,
) -> None:
    """Run one experiment configuration for one patient."""
    patient_dir = Path(patient_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_obj(config)

    # Tee stdout to log file
    log_path = run_dir / "trial_output.txt"
    _orig_stdout = sys.stdout

    class _Tee:
        def __init__(self, *files: Any) -> None:
            self.files = files
        def write(self, obj: str) -> None:
            for f in self.files:
                f.write(obj); f.flush()
        def flush(self) -> None:
            for f in self.files:
                f.flush()

    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(_orig_stdout, _log_file)

    try:
        _run_experiment_body(spec, patient_dir, run_dir, bold_branch=bold_branch, cfg=cfg)
    finally:
        sys.stdout = _orig_stdout
        _log_file.close()


def _run_experiment_body(
    spec: ExperimentSpec,
    patient_dir: Path,
    run_dir: Path,
    *,
    bold_branch: str,
    cfg: Dict[str, Any],
) -> None:
    sampling_raw, sampling_mode = _normalize_sampling_mode(spec.sampling)
    classifier_name = _normalize_classifier_name(spec.classifier)

    print(f"\n{'='*80}")
    print("DEEPNEUROBOLD v2 — EXPERIMENT START")
    print(f"Timestamp:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Representation: {spec.representation}")
    print(f"Classifier:     {classifier_name}")
    print(f"Sampling:       {spec.sampling}")
    print(f"Seed:           {spec.seed}")
    print(f"Positive:       {spec.positive}")
    print(f"Negative:       {spec.negative}")
    print(f"Patient dir:    {patient_dir}")
    print(f"Run dir:        {run_dir}")
    print(f"{'='*80}")

    # --- Config values ---
    use_brats             = bool(_cfg_get(cfg, "use_brats", False))
    dt                    = float(_cfg_get(cfg, "dt", 1.8))
    margin_iters          = int(_cfg_get(cfg, "margin_iters", 2))
    controlled_outer_iters = int(_cfg_get(cfg, "controlled_outer_iters", 12))
    n_per_class           = int(_cfg_get(cfg, "sampling.n_per_class", _cfg_get(cfg, "n_per_class", 20000)))
    holdout_frac          = float(_cfg_get(cfg, "sampling.holdout_frac", _cfg_get(cfg, "holdout_frac", 0.10)))
    patch_half_width      = int(_cfg_get(cfg, "sampling.patch_half_width", _cfg_get(cfg, "patch_half_width", 2)))
    grid_block_mm_init    = float(_cfg_get(cfg, "sampling.grid_block_mm", getattr(spec, "grid_block_mm", None) or 20.0))
    grid_buffer_mm_init   = float(_cfg_get(cfg, "sampling.grid_buffer_mm", getattr(spec, "grid_buffer_mm", None) or 5.0))
    proximity_max_samples = int(_cfg_get(cfg, "proximity.max_samples", 999999))
    cv_folds_raw          = getattr(spec, "cv_folds", None)
    cv_folds              = max(1, int(cv_folds_raw) if cv_folds_raw is not None else int(_cfg_get(cfg, "cv_folds", 1)))
    bold_source           = str(_cfg_get(
        cfg, "bold_source",
        "preprocessed" if str(spec.representation).lower() == "ts_preprocessed" else "in_t1",
    ))

    # RF hyperparameters (config-overridable to match original deepneurobold)
    _rf_max_depth_raw   = _cfg_get(cfg, "rf.max_depth", None)
    rf_max_depth        = int(_rf_max_depth_raw) if _rf_max_depth_raw is not None else None
    rf_n_estimators     = int(_cfg_get(cfg, "rf.n_estimators", 500))
    rf_min_samples_leaf = int(_cfg_get(cfg, "rf.min_samples_leaf", 1))
    rf_max_features     = str(_cfg_get(cfg, "rf.max_features", "sqrt"))

    # --- BOLD branch ---
    _set_branch_env(bold_branch)

    # --- Load data ---
    bold_4d, brain_flat, ref_img = load_bold_and_brain_mask(patient_dir, bold_source=bold_source)
    seg_masks, _ = load_segmentation_masks(patient_dir, use_brats=use_brats)

    # --- EPI ghost detection (QC, non-fatal) ---
    try:
        _detect_epi_ghost_artifacts(
            patient_dir=patient_dir,
            run_dir=run_dir,
            bold_4d=bold_4d,
            ref_img=ref_img,
        )
    except Exception as e:
        print(f"[epi-ghost] skipped (non-fatal): {e}")

    vol_shape_3d: Tuple[int, int, int] = tuple(int(x) for x in ref_img.shape[:3])  # type: ignore[assignment]
    n_vox = int(brain_flat.shape[0])

    aff = ref_img.affine
    voxel_size_mm: Tuple[float, float, float] = (
        float(np.linalg.norm(aff[:3, 0])),
        float(np.linalg.norm(aff[:3, 1])),
        float(np.linalg.norm(aff[:3, 2])),
    )

    print(f"\n[DATA] BOLD shape: {ref_img.shape}  brain_voxels: {int(np.sum(brain_flat)):,}")
    print(f"       voxel_size_mm: {voxel_size_mm}")

    # --- Train masks ---
    masks_bundle = build_train_masks(
        brain_flat=brain_flat,
        masks=seg_masks,
        positive=spec.positive,
        negative=spec.negative,
        margin_iters=margin_iters,
        controlled_outer_iters=controlled_outer_iters,
        voxel_size_mm=voxel_size_mm,
    )

    tumor_train_flat   = masks_bundle["tumor_train_flat"]
    healthy_train_flat = masks_bundle["healthy_train_flat"]
    edema_flat         = masks_bundle["edema_flat"]
    test_region_flat   = masks_bundle["test_region_flat"]
    tumor_core_flat    = masks_bundle["tumor_core_flat"]
    tumor_margin_flat  = masks_bundle["tumor_margin_flat"]
    train_mask_flat    = masks_bundle["train_mask_flat"]

    tumor_voxels  = int(np.sum(tumor_train_flat))
    healthy_voxels = int(np.sum(healthy_train_flat))
    print(f"\n[MASKS] tumor={tumor_voxels:,}  healthy={healthy_voxels:,}  "
          f"edema={int(np.sum(edema_flat)):,}  test_region={int(np.sum(test_region_flat)):,}")

    if tumor_voxels == 0:
        print("[SKIP] tumor_train_flat is empty — skipping experiment.")
        return

    # --- Grid-block params from sampling string (override init values) ---
    if sampling_mode == "grid_block":
        grid_block_mm_init = _sampling_get_float(sampling_raw, "block_mm") or grid_block_mm_init
        grid_buffer_mm_init = _sampling_get_float(sampling_raw, "buffer_mm") or grid_buffer_mm_init

    # --- Save masks for QA ---
    qa_dir = run_dir / "masks_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for _name, _mask in [
        ("mask_brain", brain_flat), ("mask_edema", edema_flat),
        ("mask_tumor_core", tumor_core_flat), ("mask_tumor_margin", tumor_margin_flat),
        ("mask_healthy_train", healthy_train_flat), ("mask_train", train_mask_flat),
        ("mask_test_region", test_region_flat),
    ]:
        _save_mask_nifti(out_path=qa_dir / f"{_name}.nii.gz",
                         mask_flat_bool=_mask, vol_shape_3d=vol_shape_3d, ref_img=ref_img)

    # =====================================================================
    # Feature extraction — full brain (train + test)
    # =====================================================================
    search_mask = brain_flat  # extract for all brain voxels
    print(f"\n[FEATURES] extracting '{spec.representation}' for {int(np.sum(search_mask)):,} voxels...")

    feat_out = extract_features(
        bold_4d=bold_4d,
        mask_flat=search_mask,
        dt=dt,
        representation=spec.representation,
        patient_dir=patient_dir,
    )
    X_all    = feat_out["features"]   # (n_brain_voxels, n_features)
    feat_names = feat_out.get("names", np.array([]))

    brain_idx = np.flatnonzero(search_mask)  # maps row -> flat voxel index

    # Map flat voxel indices -> row in X_all
    flat_to_row = np.full(n_vox, -1, dtype=np.int64)
    flat_to_row[brain_idx] = np.arange(len(brain_idx), dtype=np.int64)

    def _get_X_for_mask(mask_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.flatnonzero(mask_flat)
        rows = flat_to_row[idx]
        ok = rows >= 0
        return X_all[rows[ok]], idx[ok]

    # =====================================================================
    # CV fold loop
    # =====================================================================
    prob_accum    = np.zeros(n_vox, dtype=np.float64)
    prob_count    = np.zeros(n_vox, dtype=np.int32)
    fold_metrics: List[Dict[str, Any]] = []
    prob_flats:   List[np.ndarray]     = []   # per-fold maps for ensemble std

    split_dir = run_dir / "split_masks"
    split_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(cv_folds):
        fold_id   = fold + 1   # 1-indexed to match original deepneurobold (fold_01, fold_02 ...)
        fold_seed = int(spec.seed) + fold
        fold_dir  = run_dir / (f"fold_{fold_id:02d}" if cv_folds > 1 else "fold_01")
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"FOLD {fold+1}/{cv_folds}  seed={fold_seed}")
        print(f"{'─'*60}")

        # --- Split ---
        split_def: Dict[str, Any] = {
            "sampling_mode": sampling_mode,
            "seed": fold_seed,
            "n_per_class": n_per_class,
            "holdout_frac": holdout_frac,
            "patch_half_width": patch_half_width,
            "grid_block_mm": grid_block_mm_init,
            "grid_buffer_mm": grid_buffer_mm_init,
            "positive": spec.positive,
            "negative": spec.negative,
            "patient_dir": str(patient_dir.resolve()),
        }

        # Note: holdout_frac and patch_half_width are NOT included here for grid_block mode.
        # The original deepneurobold runner did not pass these to make_split for grid_block
        # (they were not in make_split_kwargs_base), so make_split uses its function defaults
        # (holdout_frac=0.10, patch_half_width=2).  For patch_holdout fallback, the adaptive
        # split helper adds them explicitly from frac_cands / phw_cands.
        split_kwargs: Dict[str, Any] = dict(
            tumor_train_flat=tumor_train_flat,
            healthy_train_flat=healthy_train_flat,
            vol_shape_3d=vol_shape_3d,
            mode=sampling_mode,
            seed=fold_seed,
            n_per_class=n_per_class,
            voxel_size_mm=voxel_size_mm,
        )
        if sampling_mode == "grid_block":
            split_kwargs["grid_block_mm"]  = grid_block_mm_init
            split_kwargs["grid_buffer_mm"] = grid_buffer_mm_init

        split, split_src, split_used = _adaptive_make_split_cached(
            cache_dir=fold_dir / ".split_cache",
            split_def_base=split_def,
            make_split_fn=make_split,
            make_split_kwargs_base=split_kwargs,
            sampling_mode=sampling_mode,
            voxel_size_mm=voxel_size_mm,
            grid_block_mm_init=grid_block_mm_init,
            grid_buffer_mm_init=grid_buffer_mm_init,
            holdout_frac_init=holdout_frac,
            patch_half_width_init=patch_half_width,
        )
        split_src = f"{split_src}:{split_used.get('final_mode', sampling_mode)}"

        train_idx = np.asarray(split.train_idx, dtype=np.int64).ravel()
        val_idx   = np.asarray(split.val_idx,   dtype=np.int64).ravel()
        y_train   = np.asarray(split.train_y,   dtype=np.int64).ravel()
        y_val     = np.asarray(split.val_y,     dtype=np.int64).ravel()

        # --- Sort by flat voxel index (CRITICAL: matches original deepneurobold) ---
        # Original runner calls _sorted_idx_and_labels() which sorts train/val idx by
        # ascending voxel index before extracting features.  This makes X_train rows
        # correspond to voxels in the same order as np.flatnonzero(train_mask_flat).
        # RF bootstrap sampling is NOT order-invariant for a fixed seed: without this
        # sort, the same seed picks DIFFERENT actual voxels per tree → different forest
        # → different (noisier) probability maps.
        order_tr  = np.argsort(train_idx)
        train_idx = train_idx[order_tr]
        y_train   = y_train[order_tr]
        if val_idx.size > 0:
            order_val = np.argsort(val_idx)
            val_idx   = val_idx[order_val]
            y_val     = y_val[order_val]
        print(f"[split] train={len(train_idx)}  val={len(val_idx)}  (source={split_src})")
        print(f"[sort]  train_idx[0:5]={train_idx[:5].tolist()}  y_train[0:5]={y_train[:5].tolist()}")

        # --- Build train/val masks ---
        train_mask = _mask_from_idx(n_vox, train_idx)
        val_mask   = _mask_from_idx(n_vox, val_idx)

        # Save split masks
        _save_mask_nifti(out_path=split_dir / f"fold{fold:02d}_train_pos.nii.gz",
                         mask_flat_bool=train_mask & tumor_train_flat, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        _save_mask_nifti(out_path=split_dir / f"fold{fold:02d}_train_neg.nii.gz",
                         mask_flat_bool=train_mask & healthy_train_flat, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        _save_mask_nifti(out_path=split_dir / f"fold{fold:02d}_val.nii.gz",
                         mask_flat_bool=val_mask, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        _save_mask_nifti(out_path=split_dir / "test_region.nii.gz",
                         mask_flat_bool=test_region_flat, vol_shape_3d=vol_shape_3d, ref_img=ref_img)

        # --- Split audit files ---
        _save_split_audit_files(
            fold_dir=fold_dir,
            n_vox=n_vox,
            split=split,
            train_mask_flat=train_mask,
            val_mask_flat=val_mask,
            brain_flat=brain_flat,
        )

        # --- Features for train and val ---
        train_rows = flat_to_row[train_idx]
        ok_tr = train_rows >= 0
        X_train = X_all[train_rows[ok_tr]]
        y_train_ok = y_train[ok_tr]

        if val_idx.size > 0:
            val_rows = flat_to_row[val_idx]
            ok_val = val_rows >= 0
            X_val   = X_all[val_rows[ok_val]]
            y_val_ok = y_val[ok_val]
        else:
            X_val, y_val_ok = np.zeros((0, X_train.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        print(f"[features] X_train={X_train.shape}  X_val={X_val.shape}")

        # --- NaN / Inf cleanup ---
        if np.any(np.isnan(X_train)):
            X_train = np.nan_to_num(X_train, nan=0.0)
        if np.any(np.isinf(X_train)):
            X_train = np.clip(X_train, -1e6, 1e6)
        if X_val.shape[0] > 0:
            if np.any(np.isnan(X_val)):
                X_val = np.nan_to_num(X_val, nan=0.0)
            if np.any(np.isinf(X_val)):
                X_val = np.clip(X_val, -1e6, 1e6)

        # --- Train ---
        # use_scaler: matches original deepneurobold default (True for all classifiers).
        # RF is scale-invariant, so this does not change the predictions; it only
        # ensures StandardScaler is applied consistently when the config requests it.
        use_scaler = bool(_cfg_get(cfg, "classifier.use_scaler", True))
        if str(spec.representation).lower().strip() == "full_17":
            use_scaler = True  # Always scale for full_17 (mixed feature types)

        model = _train_model_any(
            classifier_name=classifier_name,
            X_train=X_train,
            y_train=y_train_ok,
            use_scaler=use_scaler,
            seed=fold_seed,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_min_samples_leaf=rf_min_samples_leaf,
            rf_max_features=rf_max_features,
        )

        # --- Validation metrics ---
        if X_val.shape[0] > 0:
            proba_val = _predict_proba_any(classifier_name, model, X_val)
            p_val = proba_val[:, 1].astype(np.float32)
            metrics = _compute_val_metrics(y_val_ok, p_val)
            metrics["n_train_samples"] = float(len(train_idx))
            metrics["n_val_samples"]   = float(len(val_idx))
            print("[val]", "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                     if isinstance(v, float) and not np.isnan(v)))
        else:
            metrics = {
                "val_auc": float("nan"), "val_ap": float("nan"),
                "val_bal_acc": float("nan"), "val_brier": float("nan"),
                "val_mean_prob_pos": float("nan"), "val_mean_prob_neg": float("nan"),
                "n_train_samples": float(len(train_idx)), "n_val_samples": 0.0,
            }

        fold_metrics.append({"fold": fold, "seed": fold_seed, "split_src": split_src, **metrics})
        with open(fold_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_metrics[-1], f, indent=2)

        # --- Predict on full brain ---
        print(f"[inference] predicting on {X_all.shape[0]:,} brain voxels...")
        proba_all = _predict_proba_any(classifier_name, model, X_all)
        prob_brain = proba_all[:, 1].astype(np.float32)

        # Accumulate into brain-flat array
        prob_accum[brain_idx] += prob_brain.astype(np.float64)
        prob_count[brain_idx] += 1

        # --- Save fold probability map ---
        prob_flat_fold = np.zeros(n_vox, dtype=np.float32)
        prob_flat_fold[brain_idx] = prob_brain
        _save_prob_map(fold_dir, prob_flat_fold, vol_shape_3d, ref_img, "prob_tumor_like.nii.gz")
        prob_flats.append(prob_flat_fold)   # keep for ensemble std

        # per-fold edema map — mirrors original deepneurobold outputs.py save_prob_maps()
        _prob_edema_fold = np.zeros(n_vox, dtype=np.float32)
        _edema_in_brain  = edema_flat & brain_flat
        _prob_edema_fold[_edema_in_brain] = prob_flat_fold[_edema_in_brain]
        _save_float_nifti(out_path=fold_dir / "prob_tumor_like_in_edema.nii.gz",
                          data_flat=_prob_edema_fold, vol_shape_3d=vol_shape_3d, ref_img=ref_img)

        # --- Feature importance (RF) ---
        if classifier_name == "rf":
            try:
                importances = get_feature_importance(model)
                np.save(fold_dir / "rf_feature_importances.npy", importances)
                # Save X_train for out-of-sample proximity projection
                np.save(fold_dir / "X_train.npy", X_train)
            except Exception as e:
                print(f"[warning] feature importance: {e}")

        # --- Edema stats for this fold ---
        edema_mask_active = edema_flat & brain_flat
        if np.any(edema_mask_active):
            edema_probs = prob_flat_fold[edema_mask_active]
            fold_metrics[-1]["edema_n_voxels"]               = float(edema_probs.size)
            fold_metrics[-1]["edema_mean_prob"]               = float(np.mean(edema_probs))
            fold_metrics[-1]["edema_std_prob"]                = float(np.std(edema_probs))
            fold_metrics[-1]["edema_pct_tumor_like_0.5"]      = float(np.mean(edema_probs >= 0.5) * 100)
            fold_metrics[-1]["edema_pct_healthy_like_0.5"]    = float(np.mean(edema_probs < 0.5) * 100)
        else:
            for _k in ["edema_n_voxels", "edema_mean_prob", "edema_std_prob",
                       "edema_pct_tumor_like_0.5", "edema_pct_healthy_like_0.5"]:
                fold_metrics[-1][_k] = float("nan")

        # Update fold_metrics.json with edema stats
        with open(fold_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_metrics[-1], f, indent=2)

        # --- RunSummary.csv (per-fold row) ---
        _write_run_summary(
            run_dir=run_dir,
            patient_dir=patient_dir,
            spec=spec,
            method_name=classifier_name,
            bold_branch=bold_branch,
            bold_source=bold_source,
            sampling_raw=sampling_raw,
            sampling_mode_requested=sampling_mode,
            sampling_mode_effective=sampling_mode,
            split_source=split_src,
            n_per_class=n_per_class,
            holdout_frac=holdout_frac,
            patch_half_width=patch_half_width,
            grid_block_mm=grid_block_mm_init,
            grid_buffer_mm=grid_buffer_mm_init,
            fold_id=fold_id,
            fold_seed=fold_seed,
            metrics=fold_metrics[-1],
        )

        # --- Proximity MDS ---
        _proximity_mds_analysis(
            model=model,
            X=X_train,
            y=y_train_ok,
            out_dir=fold_dir,
            tag="train",
            max_samples=proximity_max_samples,
            seed=fold_seed,
            classifier_name=classifier_name,
        )
        if X_val.shape[0] > 0:
            _proximity_mds_analysis(
                model=model,
                X=X_val,
                y=y_val_ok,
                out_dir=fold_dir,
                tag="val",
                max_samples=proximity_max_samples,
                seed=fold_seed,
                classifier_name=classifier_name,
            )

    # =====================================================================
    # Ensemble average across folds
    # =====================================================================
    mask_has_pred = prob_count > 0
    prob_flat_final = np.zeros(n_vox, dtype=np.float32)
    prob_flat_final[mask_has_pred] = (
        prob_accum[mask_has_pred] / prob_count[mask_has_pred]
    ).astype(np.float32)

    prob_map_path = _save_prob_map(run_dir, prob_flat_final, vol_shape_3d, ref_img, "prob_tumor_like.nii.gz")

    # ── ensemble/prob_mean.nii.gz + prob_std.nii.gz ───────────────────────
    # Mirrors the original deepneurobold which saves these under ensemble/
    # when cv_folds > 1.  The trial_visualizer reads ensemble/prob_mean.nii.gz.
    if cv_folds > 1 and len(prob_flats) > 1:
        ens_dir = run_dir / "ensemble"
        ens_dir.mkdir(parents=True, exist_ok=True)
        prob_stack = np.stack(prob_flats, axis=0).astype(np.float32, copy=False)
        prob_mean  = np.mean(prob_stack, axis=0).astype(np.float32, copy=False)
        prob_std   = np.std(prob_stack, axis=0).astype(np.float32, copy=False)
        _save_float_nifti(out_path=ens_dir / "prob_mean.nii.gz",
                          data_flat=prob_mean, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        _save_float_nifti(out_path=ens_dir / "prob_std.nii.gz",
                          data_flat=prob_std, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        # ensemble/prob_tumor_like_in_edema.nii.gz — mirrors original deepneurobold
        _ens_edema_flat = np.zeros(n_vox, dtype=np.float32)
        _ens_edema_flat[edema_flat & brain_flat] = prob_mean[edema_flat & brain_flat]
        _save_float_nifti(out_path=ens_dir / "prob_tumor_like_in_edema.nii.gz",
                          data_flat=_ens_edema_flat, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        # ensemble_metrics.json — mirrors original deepneurobold
        _ens_edema_probs = prob_mean[edema_flat & brain_flat]
        _ens_edema_stats: Dict[str, Any] = {
            "n_edema_voxels": int(_ens_edema_probs.size),
            "mean_prob_edema": float(np.mean(_ens_edema_probs)) if _ens_edema_probs.size > 0 else float("nan"),
            "std_prob_edema":  float(np.std(_ens_edema_probs))  if _ens_edema_probs.size > 0 else float("nan"),
        }
        for _thr in [0.3, 0.5, 0.7]:
            _pct = float(np.mean(_ens_edema_probs >= _thr) * 100) if _ens_edema_probs.size > 0 else float("nan")
            _ens_edema_stats[f"pct_tumor_like_thr_{_thr}"] = _pct
            _ens_edema_stats[f"pct_healthy_like_thr_{_thr}"] = (100.0 - _pct) if not np.isnan(_pct) else float("nan")
        with open(ens_dir / "ensemble_metrics.json", "w", encoding="utf-8") as _f:
            json.dump({"folds": fold_metrics, "ensemble_edema_analysis": _ens_edema_stats},
                      _f, indent=2, sort_keys=True)
        print(f"[ensemble] prob_mean + prob_std + edema + metrics → {ens_dir}")

    # Save per-region maps
    prob_edema_flat = np.zeros(n_vox, dtype=np.float32)
    prob_edema_flat[edema_flat & brain_flat] = prob_flat_final[edema_flat & brain_flat]
    _save_float_nifti(out_path=run_dir / "prob_tumor_like_in_edema.nii.gz",
                      data_flat=prob_edema_flat, vol_shape_3d=vol_shape_3d, ref_img=ref_img)

    prob_test_flat = np.zeros(n_vox, dtype=np.float32)
    prob_test_flat[test_region_flat & brain_flat] = prob_flat_final[test_region_flat & brain_flat]
    _save_float_nifti(out_path=run_dir / "prob_tumor_like_in_test_region.nii.gz",
                      data_flat=prob_test_flat, vol_shape_3d=vol_shape_3d, ref_img=ref_img)

    # --- Aggregate metrics ---
    agg_metrics: Dict[str, Any] = {
        "n_folds": cv_folds,
        "positive": spec.positive,
        "negative": spec.negative,
        "representation": spec.representation,
        "classifier": classifier_name,
        "sampling": spec.sampling,
        "seed": int(spec.seed),
        "n_tumor_core": int(np.sum(tumor_core_flat)),
        "n_edema": int(np.sum(edema_flat)),
        "n_test_region": int(np.sum(test_region_flat)),
        "mean_prob_tumor_core": float(np.mean(prob_flat_final[tumor_core_flat])) if np.any(tumor_core_flat) else float("nan"),
        "mean_prob_edema": float(np.mean(prob_flat_final[edema_flat & brain_flat])) if np.any(edema_flat & brain_flat) else float("nan"),
        "mean_prob_test_region": float(np.mean(prob_flat_final[test_region_flat & brain_flat])) if np.any(test_region_flat & brain_flat) else float("nan"),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for k_metric in ["val_auc", "val_ap", "val_brier", "val_bal_acc"]:
        vals = [m[k_metric] for m in fold_metrics if k_metric in m and not np.isnan(m[k_metric])]
        if vals:
            agg_metrics[f"{k_metric}_mean"] = float(np.mean(vals))
            agg_metrics[f"{k_metric}_std"]  = float(np.std(vals))

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg_metrics, f, indent=2, sort_keys=True)

    # --- RunAggregate.csv ---
    _write_run_aggregate(
        run_dir=run_dir,
        patient_dir=patient_dir,
        spec=spec,
        method_name=classifier_name,
        cv_folds=cv_folds,
        sampling_mode_requested=sampling_mode,
        fold_metrics=fold_metrics,
    )

    print(f"\n[DONE] prob map: {prob_map_path}")
    print(f"       metrics:  {run_dir / 'metrics.json'}")
    print(f"       RunSummary.csv:   {run_dir / 'RunSummary.csv'}")
    print(f"       RunAggregate.csv: {run_dir / 'RunAggregate.csv'}")

    # --- Hotspot detection (skipped if hotspot.skip=True in config) ---
    if bool(_cfg_get(cfg, "hotspot.skip", False)):
        print("\n[hotspot] skipped (hotspot.skip=True — joint detection runs externally).")
        return

    try:
        from deepneurobold.analysis.hotspot.detector import HotspotDetector
        print("\n[hotspot] running hotspot detection...")
        detector = HotspotDetector(
            prob_map_path=prob_map_path,
            brain_mask_flat=brain_flat,
            test_region_flat=test_region_flat,
            edema_flat=edema_flat,
            vol_shape_3d=vol_shape_3d,
            ref_img=ref_img,
            run_dir=run_dir,
        )
        detector.detect()
    except ImportError:
        print("[hotspot] HotspotDetector not available; skipping.")
    except Exception as e:
        print(f"[hotspot] detection failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Grid expansion helpers
# ---------------------------------------------------------------------------

def _is_listlike(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _as_list(x: Any) -> List[Any]:
    return list(x) if _is_listlike(x) else [x]


def _iter_spec_combinations(spec: ExperimentSpec) -> List[Tuple[ExperimentSpec, Dict[str, Any]]]:
    """Expand any list-valued fields into individual specs."""
    from itertools import product as _product

    fields = {
        "positive": _as_list(spec.positive),
        "representation": _as_list(spec.representation),
        "sampling": _as_list(spec.sampling),
        "negative": _as_list(spec.negative),
        "classifier": _as_list(spec.classifier),
        "seed": _as_list(spec.seed),
        "cv_folds": _as_list(spec.cv_folds),
    }

    keys = list(fields.keys())
    out: List[Tuple[ExperimentSpec, Dict[str, Any]]] = []
    for combo_vals in _product(*[fields[k] for k in keys]):
        combo = dict(zip(keys, combo_vals))
        new_spec = ExperimentSpec(
            positive=combo["positive"],
            representation=combo["representation"],
            sampling=combo["sampling"],
            negative=combo["negative"],
            classifier=combo["classifier"],
            seed=combo["seed"],
            cv_folds=combo["cv_folds"],
            grid_block_mm=spec.grid_block_mm,
            grid_buffer_mm=spec.grid_buffer_mm,
            notes=spec.notes,
        )
        out.append((new_spec, combo))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_one_experiment(
    spec: ExperimentSpec,
    patient_dir: Path,
    run_dir: Path,
    *,
    bold_branch: str,
    config: Optional[Any] = None,
) -> None:
    """
    Main entry point — handles both single experiments and parameter grids.

    Parameters
    ----------
    spec : ExperimentSpec
        Experiment configuration. Any field can be a list for grid search.
    patient_dir : Path
        Patient data directory.
    run_dir : Path
        Root output directory for this run.
    bold_branch : str
        BOLD processing branch: ``"nomcst"`` or ``"mcst"``.
    config : path, dict, or None
        JSON config file path or dict with runtime overrides.
    """
    patient_dir = Path(patient_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    combos = _iter_spec_combinations(spec)

    if len(combos) == 1:
        _run_one_experiment_single(spec, patient_dir, run_dir, bold_branch=bold_branch, config=config)
        return

    print(f"\n[grid] Expanding ExperimentSpec into {len(combos)} configuration(s)")

    for i, (spec_i, combo_dict) in enumerate(combos, start=1):
        combo_id = "_".join(f"{k}={v}" for k, v in sorted(combo_dict.items()))[:64]
        cfg_dir  = run_dir / f"cfg_{i:03d}_{combo_id}"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        with open(cfg_dir / "grid_combo.json", "w", encoding="utf-8") as f:
            json.dump({
                "cfg_index": int(i), "combo": combo_dict,
                "patient_dir": str(patient_dir.resolve()),
                "cfg_dir": str(cfg_dir.resolve()),
            }, f, indent=2, sort_keys=True)

        print(f"\n[grid] ({i}/{len(combos)}) dir={cfg_dir}")
        _run_one_experiment_single(spec_i, patient_dir, cfg_dir, bold_branch=bold_branch, config=config)


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class ExperimentRunner(BaseRunner):
    """
    Full supervised experiment for one patient.

    Thin class wrapper around :func:`run_one_experiment`.

    Parameters
    ----------
    patient_dir : Path
    output_dir : Path
    spec : ExperimentSpec
    bold_branch : str
    config : dict or path, optional
    """

    def __init__(
        self,
        patient_dir: Path,
        output_dir: Path,
        spec: ExperimentSpec,
        bold_branch: str = "nomcst",
        config: Optional[Any] = None,
    ) -> None:
        super().__init__(name="ExperimentRunner", config=config if isinstance(config, dict) else None)
        self.patient_dir = Path(patient_dir)
        self.output_dir  = Path(output_dir)
        self.spec        = spec
        self.bold_branch = bold_branch
        self._config     = config

    def setup(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Run the experiment and return the output directory path."""
        self.setup()
        run_one_experiment(
            self.spec,
            self.patient_dir,
            self.output_dir,
            bold_branch=self.bold_branch,
            config=self._config,
        )
        return {"run_dir": str(self.output_dir)}
