#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Dict, Any, Optional

import numpy as np
import nibabel as nib
import pandas as pd

MetricName = Literal["tstd", "tcv", "tsnr"]


class BoldVesselProcessor:
    """
    Large vessel removal for BOLD (4D), producing:
      - metric map (3D)
      - keep/exclude masks (3D)
      - cleaned BOLD (4D)
    """

    def __init__(self, bold_path: Path, mask_path: Path, out_dir: Path):
        self.bold_path = Path(bold_path)
        self.mask_path = Path(mask_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.img_4d = nib.load(str(self.bold_path))
        self.bold_data = self.img_4d.get_fdata(dtype=np.float32)

        mask_img = nib.load(str(self.mask_path))
        self.brain_mask = (mask_img.get_fdata() > 0)

        if self.bold_data.ndim != 4:
            raise ValueError(f"Expected 4D BOLD, got shape={self.bold_data.shape} at {self.bold_path}")
        if self.brain_mask.ndim != 3:
            raise ValueError(f"Expected 3D mask, got shape={self.brain_mask.shape} at {self.mask_path}")

    def compute_metric(self, metric: MetricName) -> np.ndarray:
        eps = 1e-8

        if metric == "tstd":
            res = np.std(self.bold_data, axis=3)
        elif metric == "tcv":
            mean = np.mean(self.bold_data, axis=3)
            std = np.std(self.bold_data, axis=3)
            res = std / (mean + eps)
        elif metric == "tsnr":
            mean = np.mean(self.bold_data, axis=3)
            std = np.std(self.bold_data, axis=3)
            res = mean / (std + eps)
        else:
            res = np.std(self.bold_data, axis=3)

        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        res[~self.brain_mask] = 0.0
        return res.astype(np.float32)

    def run_vessel_removal(self, metric_type: MetricName = "tstd", percentile: float = 99.0):
        """
        Returns:
          bold_clean (4D float32), keep_mask (3D uint8), threshold (float), stats (dict)
        """
        print(f"[VesselProcessor] Computing metric: {metric_type}...")
        metric_map = self.compute_metric(metric_type)

        brain_vals = metric_map[self.brain_mask]
        brain_vals = brain_vals[brain_vals > 0]

        if brain_vals.size == 0:
            print("[VesselProcessor] WARNING: No valid metric values in brain mask.")
            return None, None, 0.0, {}

        threshold = float(np.percentile(brain_vals, percentile))
        print(f"[VesselProcessor] Threshold (p{percentile}): {threshold:.6f}")

        exclude_mask = ((metric_map >= threshold) & self.brain_mask)
        keep_mask = (self.brain_mask & (~exclude_mask)).astype(np.uint8)

        bold_clean = self.bold_data * keep_mask[..., np.newaxis]

        # --- Save outputs ---
        # 1) Save metric with a truthful name
        metric_name = f"metric_map__{metric_type}.nii.gz"
        self._save_nii(metric_map, metric_name)

        # 2) Legacy compatibility only: also write tSNR_map.nii.gz if some old code expects it
        #    (it will contain whichever metric_type you used)
        self._save_nii(metric_map, "tSNR_map.nii.gz")

        # 3) Masks
        self._save_nii(keep_mask, "large_vessel_keep_mask.nii.gz")
        self._save_nii(exclude_mask.astype(np.uint8), "large_vessel_exclude_mask.nii.gz")

        stats = self._generate_stats(metric_type, percentile, threshold, brain_vals, keep_mask, exclude_mask)
        return bold_clean, keep_mask, threshold, stats

    def _save_nii(self, data: np.ndarray, name: str) -> None:
        """
        Safe NIfTI writer:
          - always sets header shape to match data (3D or 4D)
          - preserves affine from original 4D BOLD
        """
        out_path = self.out_dir / name
        header = self.img_4d.header.copy()
        header.set_data_shape(data.shape)
        nib.save(nib.Nifti1Image(data, self.img_4d.affine, header), str(out_path))

    def _generate_stats(self, metric: str, p: float, thr: float, vals: np.ndarray,
                        keep: np.ndarray, exclude: np.ndarray) -> Dict[str, Any]:
        n_brain = int(np.sum(self.brain_mask))
        n_keep = int(np.sum(keep))
        n_exc = int(np.sum(exclude))

        stats: Dict[str, Any] = {
            "metric": metric,
            "percentile": float(p),
            "threshold": float(thr),
            "voxels_total": n_brain,
            "voxels_keep": n_keep,
            "voxels_removed": n_exc,
            "fraction_removed": float(n_exc / n_brain) if n_brain > 0 else 0.0,
            "p50": float(np.percentile(vals, 50)) if vals.size > 0 else 0.0,
            "p95": float(np.percentile(vals, 95)) if vals.size > 0 else 0.0,
            "p99": float(np.percentile(vals, 99)) if vals.size > 0 else 0.0,
            "min": float(np.min(vals)) if vals.size > 0 else 0.0,
            "max": float(np.max(vals)) if vals.size > 0 else 0.0,
        }

        json_path = self.out_dir / "vessel_analysis.json"
        csv_path = self.out_dir / "vessel_analysis_summary.csv"

        with open(json_path, "w") as f:
            json.dump(stats, f, indent=4)

        pd.DataFrame([stats]).to_csv(csv_path, index=False)
        return stats


def run_large_vessel_mask(bold_path: Path, mask_path: Path, out_dir: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper used by BasePreprocessing.

    Writes:
      - BOLD_cleaned_large_vessels_excluded.nii.gz (4D)
      - metric_map__{metric}.nii.gz + legacy tSNR_map.nii.gz
      - keep/exclude masks
      - vessel_analysis.json/csv
    """
    processor = BoldVesselProcessor(bold_path, mask_path, out_dir)

    metric = cfg.get("vessel_metric", "tstd")
    p = float(cfg.get("vessel_percentile", 99.0))

    bold_clean, keep_mask, threshold, stats = processor.run_vessel_removal(
        metric_type=metric,
        percentile=p
    )

    if bold_clean is None:
        return {"enabled": False, "error": "No valid data in brain mask"}

    cleaned_name = "BOLD_cleaned_large_vessels_excluded.nii.gz"
    processor._save_nii(bold_clean, cleaned_name)

    return {
        "enabled": True,
        "metric": metric,
        "percentile": p,
        "threshold": float(threshold),
        "stats": stats,
        "bold_cleaned": str(Path(out_dir) / cleaned_name),
        "keep_mask": str(Path(out_dir) / "large_vessel_keep_mask.nii.gz"),
        "exclude_mask": str(Path(out_dir) / "large_vessel_exclude_mask.nii.gz"),
        "metric_map": str(Path(out_dir) / f"metric_map__{metric}.nii.gz"),
        "analysis_json": str(Path(out_dir) / "vessel_analysis.json"),
        "analysis_csv": str(Path(out_dir) / "vessel_analysis_summary.csv"),
    }