"""
analysis.hotspot.visualizer
===========================
Post-trial visualization — reads completed experiment output directories and
generates diagnostic plots + runs hotspot detection via :class:`HotspotDetector`.

This class is the refactored version of ``trial_visualizer.py`` (Desktop).
It wraps :class:`HotspotDetector` for cluster detection and adds:
  - Classification metrics bar chart (AUC, AP, Brier score per run)
  - Hotspot fraction comparison across runs and percentiles
  - RF proximity scatter (MDS or t-SNE) with test-set voxel overlay
  - Feature importance time-series plot
  - BOLD time-series per ranked hotspot cluster

Usage
-----
>>> from deepneurobold.analysis.hotspot.visualizer import TrialVisualizer
>>> viz = TrialVisualizer(trial_dir)
>>> viz.print_metrics()
>>> viz.plot_classification_metrics()
>>> df = viz.compute_hotspots()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from deepneurobold.analysis.base import BaseAnalysis
from .detector import HotspotDetector, _bh_fdr, _MIN_CLUSTER_VOXELS, _MIN_DIST_FROM_CORE_MM, _MAX_CLUSTERS, _RUN_TO_LABEL


# ---------------------------------------------------------------------------
# TrialVisualizer
# ---------------------------------------------------------------------------

class TrialVisualizer(BaseAnalysis):
    """
    Generate diagnostic plots for a completed experiment trial.

    Parameters
    ----------
    trial_dir : Path
        Trial output directory (contains ``branch_*/run_*/``).
    config : dict, optional
        Runtime configuration.
    """

    def __init__(
        self,
        trial_dir: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(trial_dir=trial_dir, config=config)

        branch_dirs = sorted(self.trial_dir.glob("branch_*"))
        if not branch_dirs:
            raise FileNotFoundError(f"No branch_* folder found in {self.trial_dir}")
        self.branch_dir = branch_dirs[0]
        print(f"Using branch: {self.branch_dir.name}")

        self.runs = sorted(self.branch_dir.glob("run_*"))
        if not self.runs:
            raise FileNotFoundError(f"No run_* folders found in {self.branch_dir}")
        print(f"Found {len(self.runs)} run(s).")

        self._load_reference_data()
        self.metrics_list = self._load_all_metrics()
        if not self.metrics_list:
            raise RuntimeError("No valid metrics loaded from any run.")

        self._identify_rf_runs()
        self._sort_metrics()
        self._load_patient_info()

        # Per-channel probability maps
        self._prob_maps: Dict[str, Optional[np.ndarray]] = {}
        self._load_all_prob_maps()

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------

    def _load_reference_data(self) -> None:
        first_run = self.runs[0]
        brain_path = first_run / "masks_qa" / "mask_brain.nii.gz"
        if brain_path.exists():
            img = nib.load(str(brain_path))
            self.brain_mask_flat: np.ndarray = img.get_fdata().ravel().astype(bool)
            self.affine = img.affine
            self.header = img.header
            self.shape_3d = img.shape
        else:
            raise FileNotFoundError(f"Brain mask not found: {brain_path}")

        edema_path = first_run / "fold_01" / "split_masks" / "edema_context.nii.gz"
        if not edema_path.exists():
            # Fallback: masks_qa
            edema_path = first_run / "masks_qa" / "mask_edema.nii.gz"
        if edema_path.exists():
            self.edema_flat: np.ndarray = nib.load(str(edema_path)).get_fdata().ravel().astype(bool)
        else:
            self.edema_flat = np.zeros_like(self.brain_mask_flat)
            print("Warning: edema mask not found; using empty mask.")

        aff = self.affine
        self.voxel_size_mm: Tuple[float, float, float] = (
            float(np.linalg.norm(aff[:3, 0])),
            float(np.linalg.norm(aff[:3, 1])),
            float(np.linalg.norm(aff[:3, 2])),
        )
        self._voxel_vol_ml: float = (
            self.voxel_size_mm[0] * self.voxel_size_mm[1] * self.voxel_size_mm[2]
        ) / 1000.0

    def _load_patient_info(self) -> None:
        cfg_file = self.runs[0] / "run_config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg = json.load(f)
            self.patient_dir = Path(cfg.get("patient_dir", ""))
            self.bold_source = cfg.get("bold_source", "in_t1")
            self.bold_branch = cfg.get("branch", "nomcst")
            self.dt          = float(cfg.get("tr", 1.8))
        else:
            self.patient_dir = None
            self.bold_source = "in_t1"
            self.bold_branch = "nomcst"
            self.dt          = 1.8

    def _load_all_prob_maps(self) -> None:
        for run_path in self.runs:
            prob_path = run_path / "ensemble" / "prob_mean.nii.gz"
            if prob_path.exists():
                try:
                    flat = nib.load(str(prob_path)).get_fdata().ravel().astype(np.float32)
                    self._prob_maps[run_path.name] = flat if len(flat) == len(self.brain_mask_flat) else None
                except Exception as e:
                    self._prob_maps[run_path.name] = None
                    print(f"  [prob-maps] could not load {run_path.name}: {e}")
            else:
                self._prob_maps[run_path.name] = None

    # ------------------------------------------------------------------
    # Metrics loading
    # ------------------------------------------------------------------

    def _load_all_metrics(self) -> List[Dict[str, Any]]:
        return [m for m in (_self_load_run_metrics(r) for r in self.runs) if m]

    def _identify_rf_runs(self) -> None:
        self.rf_runs = {m["run_name"] for m in self.metrics_list if m.get("classifier") == "rf"}

    def _sort_metrics(self) -> None:
        priority = {"necrosis": 0, "enhancing": 1, "enhancing_plus_necrosis": 2}
        self.metrics_list.sort(key=lambda x: priority.get(x["positive"], 999))

    # ------------------------------------------------------------------
    # Helper masks
    # ------------------------------------------------------------------

    def _load_test_region(self, run_path: Path) -> np.ndarray:
        for cand in [
            run_path / "split_masks" / "test_region.nii.gz",
            run_path / "fold_00" / "split_masks" / "test_region.nii.gz",
            run_path / "fold_01" / "split_masks" / "test_region_30mm_plus_edema.nii.gz",
            run_path / "masks_qa" / "mask_test_region.nii.gz",
        ]:
            if cand.exists():
                flat = nib.load(str(cand)).get_fdata().ravel().astype(bool)
                if len(flat) == len(self.brain_mask_flat):
                    return flat
        print(f"  Warning: test region not found in {run_path.name}; using edema only.")
        return self.edema_flat.copy()

    def _load_tumor_core(self, run_path: Path) -> Optional[np.ndarray]:
        for cand in [
            run_path / "masks_qa" / "mask_tumor_core.nii.gz",
            run_path / "fold_00" / "masks_qa" / "mask_tumor_core.nii.gz",
        ]:
            if cand.exists():
                flat = nib.load(str(cand)).get_fdata().ravel().astype(bool)
                if len(flat) == len(self.brain_mask_flat):
                    return flat
        return None

    def _load_training_masks(
        self, run_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        for fold_name in ["fold_00", "fold_01"]:
            pos_p = run_path / fold_name / "split_masks" / "train_pos.nii.gz"
            neg_p = run_path / fold_name / "split_masks" / "train_neg.nii.gz"
            if pos_p.exists() and neg_p.exists():
                pf = nib.load(str(pos_p)).get_fdata().ravel().astype(bool)
                nf = nib.load(str(neg_p)).get_fdata().ravel().astype(bool)
                if len(pf) == len(self.brain_mask_flat):
                    return pf, nf
        return None, None

    # ------------------------------------------------------------------
    # Print / plots
    # ------------------------------------------------------------------

    def print_metrics(self) -> None:
        print("\n" + "=" * 60)
        print("CLASSIFICATION METRICS")
        print("=" * 60)
        for m in self.metrics_list:
            rf_tag = " (RF)" if m["run_name"] in self.rf_runs else ""
            print(f"\n--- {m['positive']} ({m['run_name']}){rf_tag} ---")
            print(f"AUC   = {m.get('val_auc_mean', float('nan')):.3f} ± {m.get('val_auc_std', float('nan')):.3f}")
            print(f"AP    = {m.get('val_ap_mean', float('nan')):.3f} ± {m.get('val_ap_std', float('nan')):.3f}")
            b = m.get("val_brier_mean", float("nan"))
            if not np.isnan(b):
                print(f"Brier = {b:.4f} ± {m.get('val_brier_std', float('nan')):.4f}")

    def plot_classification_metrics(
        self, out_file: str = "classification_comparison.png"
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping classification plot.")
            return

        labels = [f"{m['run_name']}\n({m['positive']})" for m in self.metrics_list]
        x = np.arange(len(labels))
        w = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.bar(x - w/2, [m.get("val_auc_mean", 0) for m in self.metrics_list], w,
                yerr=[m.get("val_auc_std", 0) for m in self.metrics_list],
                capsize=3, label="AUC", color="steelblue", alpha=0.8)
        ax1.bar(x + w/2, [m.get("val_ap_mean", 0) for m in self.metrics_list], w,
                yerr=[m.get("val_ap_std", 0) for m in self.metrics_list],
                capsize=3, label="AP", color="darkorange", alpha=0.8)
        ax1.set_ylabel("Score"); ax1.set_title("Classification performance by run")
        ax1.legend(loc="lower right"); ax1.set_ylim(0.5, 1.05)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        ax2.bar(x, [m.get("val_brier_mean", 0) for m in self.metrics_list], w * 1.5,
                yerr=[m.get("val_brier_std", 0) for m in self.metrics_list],
                capsize=3, label="Brier", color="mediumpurple", alpha=0.8)
        ax2.set_ylabel("Brier score (lower=better)")
        ax2.set_xlabel("Run"); ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.legend(loc="upper right"); ax2.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        out_path = self.trial_dir / out_file
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Classification plot saved: {out_path}")

    # ------------------------------------------------------------------
    # Hotspot detection (delegates to HotspotDetector)
    # ------------------------------------------------------------------

    def compute_hotspots(
        self,
        percentiles: List[int] = (90, 95),
        min_cluster_size: int = _MIN_CLUSTER_VOXELS,
        min_dist_from_core_mm: float = _MIN_DIST_FROM_CORE_MM,
        max_clusters: int = _MAX_CLUSTERS,
        fdr_alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Run hotspot detection for all runs using :class:`HotspotDetector`.

        Returns
        -------
        pd.DataFrame with one row per (run, percentile) combination.
        """
        all_rows: List[Dict[str, Any]] = []

        for run_path in self.runs:
            run_name = run_path.name
            print(f"\n[TrialVisualizer] Processing {run_name}...")

            prob_path = run_path / "ensemble" / "prob_mean.nii.gz"
            if not prob_path.exists():
                print(f"  Warning: {prob_path} not found; skipping.")
                continue

            test_region_flat = self._load_test_region(run_path)
            tumor_core_flat  = self._load_tumor_core(run_path)
            train_pos, train_neg = self._load_training_masks(run_path)

            # Build joint prob maps for this run
            chan_maps: Dict[str, Optional[np.ndarray]] = {}
            for rn in ["run_001", "run_002", "run_003"]:
                pm = self._prob_maps.get(rn)
                chan_maps[rn] = pm

            detector = HotspotDetector(
                prob_map_path=prob_path,
                brain_mask_flat=self.brain_mask_flat,
                test_region_flat=test_region_flat,
                edema_flat=self.edema_flat,
                vol_shape_3d=self.shape_3d,
                ref_img=nib.Nifti1Image(
                    np.zeros(self.shape_3d, dtype=np.uint8), self.affine, self.header
                ),
                run_dir=run_path,
                percentiles=list(percentiles),
                min_cluster_size=min_cluster_size,
                min_dist_from_core_mm=min_dist_from_core_mm,
                max_clusters=max_clusters,
                fdr_alpha=fdr_alpha,
            )
            detector.set_channel_prob_maps(chan_maps)
            if tumor_core_flat is not None:
                detector.set_tumor_core_flat(tumor_core_flat)
            if train_pos is not None and train_neg is not None:
                detector.set_training_masks(train_pos, train_neg)

            summary = detector.detect()
            row: Dict[str, Any] = {"run": run_name}
            row.update(summary)
            all_rows.append(row)

        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    def analyze(self) -> Dict[str, Any]:
        """Run all visualizations and hotspot detection."""
        self.print_metrics()
        self.plot_classification_metrics()
        df_hotspots = self.compute_hotspots()
        return {"hotspots": df_hotspots}


# ---------------------------------------------------------------------------
# Helper (module-level — not a method)
# ---------------------------------------------------------------------------

def _self_load_run_metrics(run_path: Path) -> Optional[Dict[str, Any]]:
    agg_file = run_path / "RunAggregate.csv"
    metrics_file = run_path / "metrics.json"

    # Try metrics.json first (deepneurobold format)
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        return {
            "run_name":       run_path.name,
            "positive":       m.get("positive", run_path.name),
            "classifier":     m.get("classifier", "unknown"),
            "val_auc_mean":   m.get("val_auc_mean", float("nan")),
            "val_auc_std":    m.get("val_auc_std",  float("nan")),
            "val_ap_mean":    m.get("val_ap_mean",  float("nan")),
            "val_ap_std":     m.get("val_ap_std",   float("nan")),
            "val_brier_mean": m.get("val_brier_mean", float("nan")),
            "val_brier_std":  m.get("val_brier_std",  float("nan")),
        }

    # Fallback: RunAggregate.csv (legacy deepneurobold format)
    if not agg_file.exists():
        return None
    try:
        df = pd.read_csv(agg_file)
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "run_name":       run_path.name,
            "positive":       run_path.name,
            "classifier":     "unknown",
            "val_auc_mean":   row.get("val_auc_mean",   float("nan")),
            "val_auc_std":    row.get("val_auc_std",    float("nan")),
            "val_ap_mean":    row.get("val_ap_mean",    float("nan")),
            "val_ap_std":     row.get("val_ap_std",     float("nan")),
            "val_brier_mean": row.get("val_brier_mean", float("nan")),
            "val_brier_std":  row.get("val_brier_std",  float("nan")),
        }
    except Exception:
        return None
