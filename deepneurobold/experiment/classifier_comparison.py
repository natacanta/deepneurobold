"""
experiment.classifier_comparison
================================
Per-patient orchestrator for ``trial_144_classifier_comparison``.

Trains **three classifiers** (RandomForest, calibrated LinearSVM, Logistic
Regression) on **identical** train/val voxel splits within the same spatial
CV folds used by trial_143, then runs paired statistical comparisons and
agreement analyses.

This module is intentionally a parallel implementation of the trial_143
runner (``deepneurobold.experiment.runner.run_one_experiment``):
- It reuses the same data-loading, mask-building, feature-extraction,
  split-caching, and voxel-sorting utilities.
- It does **not** modify the trial_143 code path.
- The split cache key is the same, so the exact same train/val voxels are
  selected as in trial_143 for the same patient + channel + fold + seed.

Outputs per channel (under ``run_dir``):

    fold_NN/
        prob_rf.nii.gz, prob_svm.nii.gz, prob_lr.nii.gz
        val_scores_rf.npy, val_scores_svm.npy, val_scores_lr.npy, val_y.npy
        fold_metrics.json      (per-classifier metrics + paired comparisons)
        feature_importance_rf.npy
        split_masks/fold{NN}_{train_pos,train_neg,val}.nii.gz
    ensemble/
        prob_mean_{rf,svm,lr}.nii.gz
        prob_std_{rf,svm,lr}.nii.gz
        classifier_comparison.json   (per-fold + aggregate)
        statistical_tests.json       (DeLong, paired-t, Wilcoxon, effect sizes)
        voxel_agreement.json         (Spearman/Pearson + Dice@p90/p95 per pair)
        calibration_curves.json
        decision_curves.json
        timings.json
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from deepneurobold.experiment.spec import ExperimentSpec
from deepneurobold.experiment.runner import (
    _adaptive_make_split_cached,
    _cfg_get,
    _compute_val_metrics,
    _load_config_obj,
    _mask_from_idx,
    _normalize_sampling_mode,
    _save_float_nifti,
    _save_mask_nifti,
    _save_prob_map,
    _set_branch_env,
)
from deepneurobold.data.loader import load_bold_and_brain_mask
from deepneurobold.data.masks import (
    load_segmentation_masks, build_train_masks, load_synthseg_seg,
)
from deepneurobold.data.sampling import make_split
from deepneurobold.features.bold_features import extract_features

from deepneurobold.classifiers.random_forest import (
    train_random_forest, predict_proba as predict_proba_rf,
    get_feature_importance,
)
from deepneurobold.classifiers.svm import (
    train_svm_linear, predict_proba as predict_proba_svm,
)
from deepneurobold.classifiers.logistic_regression import (
    train_logreg, predict_proba as predict_proba_lr,
)

from deepneurobold.experiment.classifier_stats import (
    delong_test, paired_t_metric, wilcoxon_signed_rank,
    cliffs_delta, cohen_dz_paired,
    bonferroni_adjust, benjamini_hochberg,
)
from deepneurobold.analysis.classifier_agreement import (
    voxel_agreement, hotspot_dice,
    calibration_curve_data, decision_curve_data,
)


# ---------------------------------------------------------------------------
# Per-classifier training wrappers (with timing)
# ---------------------------------------------------------------------------

CLASSIFIER_KEYS = ("rf", "svm", "lr")
CLASSIFIER_LABELS = {"rf": "RandomForest", "svm": "LinearSVM_Platt", "lr": "LogisticRegression"}


def _train_all_three(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_scaler: bool,
    fold_seed: int,
    *,
    rf_n_estimators: int = 500,
    rf_max_depth: Optional[int] = None,
    rf_min_samples_leaf: int = 1,
    rf_max_features: str = "sqrt",
    svm_calibration_cv: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Train all three classifiers on the same X/y. Returns models + timings."""
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    rf_model = train_random_forest(
        X_train=X_train, y_train=y_train,
        use_scaler=use_scaler, seed=fold_seed,
        n_estimators=rf_n_estimators, max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf, max_features=rf_max_features,
    )
    timings["train_rf_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    svm_model = train_svm_linear(
        X_train=X_train, y_train=y_train,
        use_scaler=use_scaler, seed=fold_seed,
        calibration_cv=svm_calibration_cv,
    )
    timings["train_svm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lr_model = train_logreg(
        X_train=X_train, y_train=y_train,
        use_scaler=use_scaler, seed=fold_seed,
    )
    timings["train_lr_s"] = time.perf_counter() - t0

    return {"rf": rf_model, "svm": svm_model, "lr": lr_model}, timings


def _predict_pos_prob_three(models: Dict[str, Any], X: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Predict positive-class probability for all three classifiers."""
    timings: Dict[str, float] = {}
    probs: Dict[str, np.ndarray] = {}

    t0 = time.perf_counter()
    probs["rf"] = predict_proba_rf(models["rf"], X)[:, 1].astype(np.float32)
    timings["pred_rf_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs["svm"] = predict_proba_svm(models["svm"], X)[:, 1].astype(np.float32)
    timings["pred_svm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs["lr"] = predict_proba_lr(models["lr"], X)[:, 1].astype(np.float32)
    timings["pred_lr_s"] = time.perf_counter() - t0

    return probs, timings


# ---------------------------------------------------------------------------
# Statistical tests on per-fold metric vectors
# ---------------------------------------------------------------------------

METRICS_FOR_TESTS = ("val_auc", "val_ap", "val_brier", "val_bal_acc")
CLASSIFIER_PAIRS = (("rf", "svm"), ("rf", "lr"), ("svm", "lr"))


def _paired_metric_tests(per_fold_metrics: List[Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """
    Per-fold metric vectors → paired-t, Wilcoxon, Cliff's δ, Cohen's d_z.
    Bonferroni + BH multiple-testing correction across all (pair × metric) tests.

    Parameters
    ----------
    per_fold_metrics : list of dicts. Each fold dict has keys ``rf, svm, lr`` →
        each maps to a metric dict containing keys in METRICS_FOR_TESTS.

    Returns
    -------
    dict with: ``per_pair`` (nested), ``raw_pvalues``, ``bonferroni``, ``bh_fdr``.
    """
    out: Dict[str, Any] = {"per_pair": {}, "raw_pvalues": [], "test_names": []}

    for (a, b) in CLASSIFIER_PAIRS:
        pair_key = f"{a}_vs_{b}"
        out["per_pair"][pair_key] = {}
        for metric in METRICS_FOR_TESTS:
            va = [fm[a].get(metric, np.nan) for fm in per_fold_metrics]
            vb = [fm[b].get(metric, np.nan) for fm in per_fold_metrics]
            t_res = paired_t_metric(va, vb)
            w_res = wilcoxon_signed_rank(va, vb)
            d_res = cliffs_delta(va, vb)
            dz_res = cohen_dz_paired(va, vb)
            out["per_pair"][pair_key][metric] = {
                "values_a": [float(x) for x in va],
                "values_b": [float(x) for x in vb],
                "paired_t": t_res,
                "wilcoxon": w_res,
                "cliffs_delta": d_res,
                "cohen_dz": dz_res,
            }
            # Use Wilcoxon p as the "primary" non-parametric p-value
            out["raw_pvalues"].append(w_res.get("p_two_sided", np.nan))
            out["test_names"].append(f"{pair_key}:{metric}:wilcoxon")

    # Multiple-testing corrections across (3 pairs × 4 metrics) = 12 tests
    raw_p = [float(p) if np.isfinite(p) else 1.0 for p in out["raw_pvalues"]]
    bonf = bonferroni_adjust(raw_p, n_tests=len(raw_p))
    bh = benjamini_hochberg(raw_p)
    out["bonferroni"] = dict(zip(out["test_names"], bonf))
    out["bh_fdr"] = dict(zip(out["test_names"], bh))
    return out


def _delong_tests_pooled(
    pooled_y: np.ndarray,
    pooled_scores: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """DeLong on pooled validation scores across folds, per classifier pair."""
    out: Dict[str, Any] = {}
    raw_p: List[float] = []
    names: List[str] = []
    for (a, b) in CLASSIFIER_PAIRS:
        key = f"{a}_vs_{b}"
        res = delong_test(pooled_y, pooled_scores[a], pooled_scores[b])
        out[key] = res
        raw_p.append(float(res.get("p_two_sided", np.nan)) if np.isfinite(res.get("p_two_sided", np.nan)) else 1.0)
        names.append(key)
    out["bonferroni"] = dict(zip(names, bonferroni_adjust(raw_p, n_tests=len(raw_p))))
    out["bh_fdr"] = dict(zip(names, benjamini_hochberg(raw_p)))
    return out


# ---------------------------------------------------------------------------
# Voxel agreement on probability maps (within test_region)
# ---------------------------------------------------------------------------

def _voxel_agreement_all_pairs(
    prob_maps: Dict[str, np.ndarray],
    region_mask: np.ndarray,
) -> Dict[str, Any]:
    """Spearman/Pearson + Dice@p90/p95 for each classifier pair."""
    out: Dict[str, Any] = {}
    for (a, b) in CLASSIFIER_PAIRS:
        key = f"{a}_vs_{b}"
        agreement = voxel_agreement(prob_maps[a], prob_maps[b], region_mask)
        dice = hotspot_dice(prob_maps[a], prob_maps[b], region_mask, percentiles=(90.0, 95.0))
        out[key] = {"voxel_agreement": agreement, "hotspot_dice": dice}
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_classifier_comparison(
    spec: ExperimentSpec,
    patient_dir: Path,
    run_dir: Path,
    *,
    bold_branch: str = "nomcst",
    config: Any = None,
) -> Dict[str, Any]:
    """
    Run the 3-classifier comparison for ONE channel of ONE patient.

    Reads the same config keys as the trial_143 runner. The ``spec.classifier``
    field is ignored — all three classifiers are trained.
    """
    cfg = _load_config_obj(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("DEEPNEUROBOLD v2 — CLASSIFIER COMPARISON")
    print(f"Timestamp:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Representation: {spec.representation}")
    print(f"Classifiers:    RF + SVM(cv=3 Platt) + LogReg")
    print(f"Sampling:       {spec.sampling}")
    print(f"Seed:           {spec.seed}")
    print(f"Positive:       {spec.positive}")
    print(f"Negative:       {spec.negative}")
    print(f"Patient dir:    {patient_dir}")
    print(f"Run dir:        {run_dir}")
    print(f"{'='*80}")

    # ---- Config values (mirror trial_143) ----
    use_brats = bool(_cfg_get(cfg, "use_brats", False))
    dt = float(_cfg_get(cfg, "dt", 1.8))
    margin_iters = int(_cfg_get(cfg, "margin_iters", 2))
    controlled_outer_iters = int(_cfg_get(cfg, "controlled_outer_iters", 12))
    n_per_class = int(_cfg_get(cfg, "sampling.n_per_class", _cfg_get(cfg, "n_per_class", 20000)))
    holdout_frac = float(_cfg_get(cfg, "sampling.holdout_frac", _cfg_get(cfg, "holdout_frac", 0.10)))
    patch_half_width = int(_cfg_get(cfg, "sampling.patch_half_width", _cfg_get(cfg, "patch_half_width", 2)))
    grid_block_mm_init = float(_cfg_get(cfg, "sampling.grid_block_mm",
                                         getattr(spec, "grid_block_mm", None) or 20.0))
    grid_buffer_mm_init = float(_cfg_get(cfg, "sampling.grid_buffer_mm",
                                          getattr(spec, "grid_buffer_mm", None) or 5.0))
    cv_folds_raw = getattr(spec, "cv_folds", None)
    cv_folds = max(1, int(cv_folds_raw) if cv_folds_raw is not None else int(_cfg_get(cfg, "cv_folds", 1)))
    bold_source = str(_cfg_get(
        cfg, "bold_source",
        "preprocessed" if str(spec.representation).lower() == "ts_preprocessed" else "in_t1",
    ))

    # RF hyperparameters
    rf_n_estimators = int(_cfg_get(cfg, "rf.n_estimators", 500))
    rf_max_depth_raw = _cfg_get(cfg, "rf.max_depth", None)
    rf_max_depth = int(rf_max_depth_raw) if rf_max_depth_raw is not None else None
    rf_min_samples_leaf = int(_cfg_get(cfg, "rf.min_samples_leaf", 1))
    rf_max_features = str(_cfg_get(cfg, "rf.max_features", "sqrt"))

    # SVM calibration CV folds (publication spec → 3)
    svm_calibration_cv = int(_cfg_get(cfg, "svm.calibration_cv", 3))

    sampling_raw, sampling_mode = _normalize_sampling_mode(spec.sampling)

    # ---- BOLD branch ----
    _set_branch_env(bold_branch)

    # ---- Load data ----
    bold_4d, brain_flat, ref_img = load_bold_and_brain_mask(patient_dir, bold_source=bold_source)
    seg_masks, _ = load_segmentation_masks(patient_dir, use_brats=use_brats)

    vol_shape_3d: Tuple[int, int, int] = tuple(int(x) for x in ref_img.shape[:3])
    aff = ref_img.affine
    voxel_size_mm = (
        float(np.linalg.norm(aff[:3, 0])),
        float(np.linalg.norm(aff[:3, 1])),
        float(np.linalg.norm(aff[:3, 2])),
    )
    n_vox = int(np.prod(vol_shape_3d))
    brain_flat = np.asarray(brain_flat, dtype=bool).ravel()

    # ---- Training masks (optional SynthSeg-based stratification) ----
    synthseg_seg = load_synthseg_seg(patient_dir, ref_img=ref_img)
    if synthseg_seg is not None:
        print(f"[synthseg] tissue stratification enabled (1/3 GM + 2/3 WM, CSF excluded)")
    else:
        print(f"[synthseg] no parcellation -> uniform background pool")

    masks_bundle = build_train_masks(
        brain_flat=brain_flat,
        masks=seg_masks,
        positive=spec.positive,
        negative=spec.negative,
        margin_iters=margin_iters,
        controlled_outer_iters=controlled_outer_iters,
        voxel_size_mm=voxel_size_mm,
        synthseg_seg=synthseg_seg,
    )
    tumor_train_flat = masks_bundle["tumor_train_flat"]
    healthy_train_flat = masks_bundle["healthy_train_flat"]
    train_mask_flat = masks_bundle["train_mask_flat"]
    edema_flat = masks_bundle["edema_flat"]
    test_region_flat = masks_bundle["test_region_flat"]
    tumor_core_flat = masks_bundle["tumor_core_flat"]
    bg_gm_flat = masks_bundle.get("gm_flat")
    bg_wm_flat = masks_bundle.get("wm_flat")

    print(f"[mask] tumor_train={int(np.sum(tumor_train_flat))}  "
          f"healthy_train={int(np.sum(healthy_train_flat))}  "
          f"test_region={int(np.sum(test_region_flat))}")

    # ---- Feature extraction (full brain, once) ----
    print(f"\n[FEATURES] extracting '{spec.representation}' for "
          f"{int(np.sum(brain_flat)):,} voxels...")
    feat_out = extract_features(
        bold_4d=bold_4d, mask_flat=brain_flat,
        dt=dt, representation=spec.representation,
        patient_dir=patient_dir,
    )
    X_all = feat_out["features"]
    brain_idx = np.flatnonzero(brain_flat)
    flat_to_row = np.full(n_vox, -1, dtype=np.int64)
    flat_to_row[brain_idx] = np.arange(len(brain_idx), dtype=np.int64)
    print(f"[FEATURES] X_all={X_all.shape}")

    # ---- CV fold loop ----
    fold_metrics_all: List[Dict[str, Dict[str, float]]] = []
    timings_all: List[Dict[str, float]] = []
    prob_flats: Dict[str, List[np.ndarray]] = {k: [] for k in CLASSIFIER_KEYS}

    pooled_val_y: List[np.ndarray] = []
    pooled_val_scores: Dict[str, List[np.ndarray]] = {k: [] for k in CLASSIFIER_KEYS}

    split_dir = run_dir / "split_masks"
    split_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(cv_folds):
        fold_id = fold + 1
        fold_seed = int(spec.seed) + fold
        fold_dir = run_dir / (f"fold_{fold_id:02d}" if cv_folds > 1 else "fold_01")
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"FOLD {fold_id}/{cv_folds}  seed={fold_seed}")
        print(f"{'─'*60}")

        # ---- Spatial split (cached, identical key to trial_143) ----
        # The split_def is the SHA1-keyed cache identifier. We use the SAME
        # construction as the trial_143 runner (runner.py:1218-1229) so that
        # the cache hits cross-trial: trial_144/145 reuses the exact splits
        # that trial_143 already computed for this patient/seed/fold.
        split_def: Dict[str, Any] = {
            "sampling_mode": sampling_mode,
            "seed": fold_seed,
            "n_per_class": n_per_class,
            "holdout_frac": holdout_frac,
            "patch_half_width": patch_half_width,
            "grid_block_mm": grid_block_mm_init,
            "grid_buffer_mm": grid_buffer_mm_init,
            "positive": str(spec.positive),
            "negative": str(spec.negative),
            "patient_dir": str(patient_dir.resolve()),
        }
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
            split_kwargs["grid_block_mm"] = grid_block_mm_init
            split_kwargs["grid_buffer_mm"] = grid_buffer_mm_init
        if bg_gm_flat is not None and bg_wm_flat is not None:
            split_kwargs["bg_gm_flat"] = bg_gm_flat
            split_kwargs["bg_wm_flat"] = bg_wm_flat
            split_def["bg_stratified"] = "synthseg_gm_wm_1_3"

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

        train_idx = np.asarray(split.train_idx, dtype=np.int64).ravel()
        val_idx = np.asarray(split.val_idx, dtype=np.int64).ravel()
        y_train = np.asarray(split.train_y, dtype=np.int64).ravel()
        y_val = np.asarray(split.val_y, dtype=np.int64).ravel()

        # ---- Sort by flat voxel index (CRITICAL — identical to trial_143) ----
        order_tr = np.argsort(train_idx)
        train_idx = train_idx[order_tr]
        y_train = y_train[order_tr]
        if val_idx.size > 0:
            order_val = np.argsort(val_idx)
            val_idx = val_idx[order_val]
            y_val = y_val[order_val]
        print(f"[split] train={len(train_idx)}  val={len(val_idx)}  (source={split_src})")

        # ---- Save split masks for QA ----
        train_mask = _mask_from_idx(n_vox, train_idx)
        val_mask = _mask_from_idx(n_vox, val_idx)
        _save_mask_nifti(out_path=split_dir / f"fold{fold:02d}_train_pos.nii.gz",
                         mask_flat_bool=train_mask & tumor_train_flat,
                         vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        _save_mask_nifti(out_path=split_dir / f"fold{fold:02d}_train_neg.nii.gz",
                         mask_flat_bool=train_mask & healthy_train_flat,
                         vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        _save_mask_nifti(out_path=split_dir / f"fold{fold:02d}_val.nii.gz",
                         mask_flat_bool=val_mask,
                         vol_shape_3d=vol_shape_3d, ref_img=ref_img)
        _save_mask_nifti(out_path=split_dir / "test_region.nii.gz",
                         mask_flat_bool=test_region_flat,
                         vol_shape_3d=vol_shape_3d, ref_img=ref_img)

        # ---- Extract X_train, X_val from cached X_all ----
        train_rows = flat_to_row[train_idx]
        ok_tr = train_rows >= 0
        X_train = X_all[train_rows[ok_tr]]
        y_train_ok = y_train[ok_tr]
        if val_idx.size > 0:
            val_rows = flat_to_row[val_idx]
            ok_val = val_rows >= 0
            X_val = X_all[val_rows[ok_val]]
            y_val_ok = y_val[ok_val]
        else:
            X_val = np.zeros((0, X_train.shape[1]), dtype=np.float32)
            y_val_ok = np.zeros((0,), dtype=np.int64)

        # ---- NaN/Inf cleanup (same as trial_143) ----
        if np.any(np.isnan(X_train)):
            X_train = np.nan_to_num(X_train, nan=0.0)
        if np.any(np.isinf(X_train)):
            X_train = np.clip(X_train, -1e6, 1e6)
        if X_val.shape[0] > 0:
            if np.any(np.isnan(X_val)):
                X_val = np.nan_to_num(X_val, nan=0.0)
            if np.any(np.isinf(X_val)):
                X_val = np.clip(X_val, -1e6, 1e6)

        print(f"[features] X_train={X_train.shape}  X_val={X_val.shape}")

        # ---- Train all 3 classifiers ----
        use_scaler = bool(_cfg_get(cfg, "classifier.use_scaler", True))
        models, t_train = _train_all_three(
            X_train=X_train, y_train=y_train_ok,
            use_scaler=use_scaler, fold_seed=fold_seed,
            rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth,
            rf_min_samples_leaf=rf_min_samples_leaf, rf_max_features=rf_max_features,
            svm_calibration_cv=svm_calibration_cv,
        )

        # ---- Validation metrics ----
        per_clf_metrics: Dict[str, Dict[str, float]] = {}
        val_scores: Dict[str, np.ndarray] = {}
        if X_val.shape[0] > 0:
            probs_val, _ = _predict_pos_prob_three(models, X_val)
            for k in CLASSIFIER_KEYS:
                p = probs_val[k]
                m = _compute_val_metrics(y_val_ok, p)
                m["n_train_samples"] = float(len(train_idx))
                m["n_val_samples"] = float(len(val_idx))
                # Add sensitivity / specificity at threshold 0.5
                pred_pos = (p >= 0.5)
                tp = float(np.sum(pred_pos & (y_val_ok == 1)))
                fn = float(np.sum(~pred_pos & (y_val_ok == 1)))
                tn = float(np.sum(~pred_pos & (y_val_ok == 0)))
                fp = float(np.sum(pred_pos & (y_val_ok == 0)))
                m["val_sensitivity@0.5"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
                m["val_specificity@0.5"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
                per_clf_metrics[k] = m
                val_scores[k] = p
                print(f"[val:{k}] AUC={m['val_auc']:.4f}  AP={m['val_ap']:.4f}  "
                      f"Brier={m['val_brier']:.4f}  bal_acc={m['val_bal_acc']:.4f}")
            # Save raw paired scores for DeLong
            np.save(fold_dir / "val_y.npy", y_val_ok.astype(np.int32))
            for k in CLASSIFIER_KEYS:
                np.save(fold_dir / f"val_scores_{k}.npy", val_scores[k].astype(np.float32))
            pooled_val_y.append(y_val_ok.astype(np.int32))
            for k in CLASSIFIER_KEYS:
                pooled_val_scores[k].append(val_scores[k].astype(np.float64))
        else:
            nan = float("nan")
            for k in CLASSIFIER_KEYS:
                per_clf_metrics[k] = {
                    "val_auc": nan, "val_ap": nan, "val_bal_acc": nan,
                    "val_brier": nan, "val_mean_prob_pos": nan, "val_mean_prob_neg": nan,
                    "val_sensitivity@0.5": nan, "val_specificity@0.5": nan,
                    "n_train_samples": float(len(train_idx)), "n_val_samples": 0.0,
                }

        # ---- Whole-brain inference (all 3) ----
        print(f"[inference] predicting on {X_all.shape[0]:,} brain voxels (×3)...")
        probs_all, t_pred = _predict_pos_prob_three(models, X_all)

        # Save NIfTI maps
        for k in CLASSIFIER_KEYS:
            prob_flat = np.zeros(n_vox, dtype=np.float32)
            prob_flat[brain_idx] = probs_all[k]
            _save_prob_map(fold_dir, prob_flat, vol_shape_3d, ref_img, f"prob_{k}.nii.gz")
            prob_flats[k].append(prob_flat)

        # ---- RF feature importance ----
        try:
            np.save(fold_dir / "feature_importance_rf.npy", get_feature_importance(models["rf"]))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] feature importance failed: {e}")

        # ---- fold_metrics.json (all classifiers + paired diff for this fold) ----
        fold_pair_diffs: Dict[str, Dict[str, float]] = {}
        for (a, b) in CLASSIFIER_PAIRS:
            fold_pair_diffs[f"{a}_vs_{b}"] = {
                m: float(per_clf_metrics[a].get(m, np.nan) - per_clf_metrics[b].get(m, np.nan))
                for m in METRICS_FOR_TESTS
            }

        timings_fold = {**t_train, **t_pred,
                        "fold_id": int(fold_id), "fold_seed": int(fold_seed)}
        with open(fold_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "fold_id": int(fold_id),
                "fold_seed": int(fold_seed),
                "per_classifier": per_clf_metrics,
                "pair_differences": fold_pair_diffs,
                "split_source": str(split_src),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "timings_s": timings_fold,
            }, f, indent=2, sort_keys=True)

        fold_metrics_all.append(per_clf_metrics)
        timings_all.append(timings_fold)

    # ==========================================================
    # Ensemble + statistical tests + agreement
    # ==========================================================
    ens_dir = run_dir / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    prob_mean_maps: Dict[str, np.ndarray] = {}
    for k in CLASSIFIER_KEYS:
        if len(prob_flats[k]) >= 1:
            stack = np.stack(prob_flats[k], axis=0).astype(np.float32, copy=False)
            mean = np.mean(stack, axis=0).astype(np.float32)
            std = np.std(stack, axis=0).astype(np.float32)
            _save_float_nifti(out_path=ens_dir / f"prob_mean_{k}.nii.gz",
                              data_flat=mean, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
            _save_float_nifti(out_path=ens_dir / f"prob_std_{k}.nii.gz",
                              data_flat=std, vol_shape_3d=vol_shape_3d, ref_img=ref_img)
            prob_mean_maps[k] = mean

    # ---- Per-fold paired tests (paired-t, Wilcoxon, Cliff's δ, Cohen's d_z) ----
    per_fold_tests = _paired_metric_tests(fold_metrics_all)

    # ---- DeLong on pooled validation scores ----
    if pooled_val_y:
        py = np.concatenate(pooled_val_y, axis=0)
        ps = {k: np.concatenate(pooled_val_scores[k], axis=0) for k in CLASSIFIER_KEYS}
        delong_pooled = _delong_tests_pooled(py, ps)
    else:
        delong_pooled = {"note": "no validation samples"}

    statistical_tests = {
        "primary_per_fold": per_fold_tests,
        "secondary_delong_pooled": delong_pooled,
        "classifier_pairs": [f"{a}_vs_{b}" for a, b in CLASSIFIER_PAIRS],
        "metrics_tested": list(METRICS_FOR_TESTS),
        "n_tests_corrected": len(CLASSIFIER_PAIRS) * len(METRICS_FOR_TESTS),
        "alpha_bonferroni": 0.05 / (len(CLASSIFIER_PAIRS) * len(METRICS_FOR_TESTS)),
    }
    with open(ens_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(statistical_tests, f, indent=2, sort_keys=True)

    # ---- Voxel agreement (Spearman, Pearson, Dice@p90/p95) on ensemble mean maps ----
    region_for_agreement = test_region_flat & brain_flat
    if prob_mean_maps:
        agreement_pairs = _voxel_agreement_all_pairs(prob_mean_maps, region_for_agreement)
        with open(ens_dir / "voxel_agreement.json", "w", encoding="utf-8") as f:
            json.dump(agreement_pairs, f, indent=2, sort_keys=True)

    # ---- Calibration curves + decision curves (pooled validation) ----
    calibration: Dict[str, Any] = {}
    dca: Dict[str, Any] = {}
    if pooled_val_y:
        for k in CLASSIFIER_KEYS:
            calibration[k] = calibration_curve_data(py, ps[k], n_bins=10, strategy="quantile")
            dca[k] = decision_curve_data(py, ps[k])
    with open(ens_dir / "calibration_curves.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, sort_keys=True)
    with open(ens_dir / "decision_curves.json", "w", encoding="utf-8") as f:
        json.dump(dca, f, indent=2, sort_keys=True)

    # ---- Aggregate classifier_comparison.json (per-fold + summary) ----
    per_clf_summary: Dict[str, Dict[str, Any]] = {}
    for k in CLASSIFIER_KEYS:
        per_clf_summary[k] = {"label": CLASSIFIER_LABELS[k], "per_fold": {}}
        for metric in METRICS_FOR_TESTS:
            vals = [fm[k].get(metric, np.nan) for fm in fold_metrics_all]
            valid = [v for v in vals if np.isfinite(v)]
            per_clf_summary[k]["per_fold"][metric] = [float(v) for v in vals]
            per_clf_summary[k][f"{metric}_mean"] = float(np.mean(valid)) if valid else float("nan")
            per_clf_summary[k][f"{metric}_std"] = float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")
            per_clf_summary[k][f"{metric}_median"] = float(np.median(valid)) if valid else float("nan")

    classifier_comparison = {
        "trial": "trial_144_classifier_comparison",
        "patient_dir": str(patient_dir),
        "run_dir": str(run_dir),
        "spec": spec.to_dict(),
        "bold_branch": bold_branch,
        "bold_source": bold_source,
        "cv_folds": cv_folds,
        "n_train_pos_pool": int(np.sum(tumor_train_flat)),
        "n_train_neg_pool": int(np.sum(healthy_train_flat)),
        "n_test_region": int(np.sum(test_region_flat)),
        "n_brain_voxels": int(np.sum(brain_flat)),
        "per_classifier_summary": per_clf_summary,
        "rf_hyperparameters": {
            "n_estimators": rf_n_estimators, "max_depth": rf_max_depth,
            "min_samples_leaf": rf_min_samples_leaf, "max_features": rf_max_features,
        },
        "svm_calibration_cv": svm_calibration_cv,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ens_dir / "classifier_comparison.json", "w", encoding="utf-8") as f:
        json.dump(classifier_comparison, f, indent=2, sort_keys=True)

    # ---- Timings ----
    with open(ens_dir / "timings.json", "w", encoding="utf-8") as f:
        json.dump({"per_fold": timings_all}, f, indent=2, sort_keys=True)

    print(f"\n[done] classifier comparison → {ens_dir}")
    return classifier_comparison
