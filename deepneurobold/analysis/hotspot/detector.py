"""
analysis.hotspot.detector
=========================
Tumor infiltration hotspot detector.

Pipeline:

1. Joint probability map — mean of three channels (necrosis / enhancing /
   enh+necrosis).  Used for cluster definition, z-test, and ranking.
2. Adaptive percentile threshold — computed on the joint map *within the
   test region* (edema + 30 mm perilesional band, excluding tumor core).
3. Connected components → size / distance / FDR filters.
4. Three-channel qualifier — individual channel probabilities used to label
   each cluster as necrosis-like, enhancing-like, or enh+necrosis-like.
5. Benjamini-Hochberg FDR correction on cluster z-test p-values.
6. Ranked NIfTI outputs + CSV rankings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import stats as scipy_stats

from .base import BaseHotspotDetector


# ---------------------------------------------------------------------------
# Clinical filtering parameters
# ---------------------------------------------------------------------------
_MIN_CLUSTER_VOXELS: int      = 150   # ~0.4 ml
_MIN_DIST_FROM_CORE_MM: float = 5.0   # spill-over exclusion
_MAX_CLUSTERS: int            = 3     # per patient per percentile

# Run name → tissue label mapping
_RUN_TO_LABEL: Dict[str, str] = {
    "run_001": "necrosis",
    "run_002": "enhancing",
    "run_003": "enhancing_plus_necrosis",
}


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def _bh_fdr(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns
    -------
    p_adj : np.ndarray of float — BH-adjusted p-values
    rejected : np.ndarray of bool — True where null hypothesis is rejected
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return p.copy(), np.zeros(n, dtype=bool)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    p_adj = np.minimum(p * n / ranks, 1.0)
    p_adj_monotone = np.minimum.accumulate(p_adj[order][::-1])[::-1]
    p_adj_out = np.empty(n, dtype=float)
    p_adj_out[order] = p_adj_monotone
    return p_adj_out, p_adj_out <= alpha


# ---------------------------------------------------------------------------
# HotspotDetector
# ---------------------------------------------------------------------------

class HotspotDetector(BaseHotspotDetector):
    """
    Detect tumor infiltration hotspots in the perilesional test region.

    Parameters
    ----------
    prob_map_path : Path
        Path to the ensemble probability NIfTI (``prob_tumor_like.nii.gz``).
    brain_mask_flat : np.ndarray, bool
        Flattened brain mask.
    test_region_flat : np.ndarray, bool
        Flattened test region mask (edema + 30 mm band, excluding core).
    edema_flat : np.ndarray, bool
        Flattened edema mask.
    vol_shape_3d : tuple
        3-D volume shape.
    ref_img : nib.Nifti1Image
        Reference image for affine/header.
    run_dir : Path
        Output directory for this run.
    percentiles : list of int
        Probability percentiles used as thresholds (default [90, 95]).
    min_cluster_size : int
    min_dist_from_core_mm : float
    max_clusters : int
    fdr_alpha : float
    config : dict, optional
    """

    def __init__(
        self,
        prob_map_path: Path,
        brain_mask_flat: np.ndarray,
        test_region_flat: np.ndarray,
        edema_flat: np.ndarray,
        vol_shape_3d: Tuple[int, int, int],
        ref_img: nib.Nifti1Image,
        run_dir: Path,
        percentiles: Optional[List[int]] = None,
        min_cluster_size: int = _MIN_CLUSTER_VOXELS,
        min_dist_from_core_mm: float = _MIN_DIST_FROM_CORE_MM,
        max_clusters: int = _MAX_CLUSTERS,
        fdr_alpha: float = 0.05,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            run_dir=run_dir,
            brain_mask_flat=brain_mask_flat,
            test_region_flat=test_region_flat,
            edema_flat=edema_flat,
            vol_shape_3d=vol_shape_3d,
            ref_img=ref_img,
            config=config,
        )
        self.prob_map_path         = Path(prob_map_path)
        self.percentiles           = percentiles or [90, 95]
        self.min_cluster_size      = int(min_cluster_size)
        self.min_dist_from_core_mm = float(min_dist_from_core_mm)
        self.max_clusters          = int(max_clusters)
        self.fdr_alpha             = float(fdr_alpha)

        aff = ref_img.affine
        self.voxel_size_mm: Tuple[float, float, float] = (
            float(np.linalg.norm(aff[:3, 0])),
            float(np.linalg.norm(aff[:3, 1])),
            float(np.linalg.norm(aff[:3, 2])),
        )
        self._voxel_vol_ml: float = (
            self.voxel_size_mm[0] * self.voxel_size_mm[1] * self.voxel_size_mm[2]
        ) / 1000.0

        # Additional per-channel probability maps
        self._prob_maps: Dict[str, Optional[np.ndarray]] = {}
        # Tumor core (optional — for distance filter)
        self.tumor_core_flat: Optional[np.ndarray] = None
        # Training masks (optional — for overlay NIfTI)
        self.train_pos_flat: Optional[np.ndarray] = None
        self.train_neg_flat: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Optional setters
    # ------------------------------------------------------------------

    def set_channel_prob_maps(self, prob_maps: Dict[str, Optional[np.ndarray]]) -> None:
        """
        Register per-channel probability maps for the three-channel qualifier.

        Parameters
        ----------
        prob_maps : dict
            Keys: ``"run_001"`` (necrosis), ``"run_002"`` (enhancing),
            ``"run_003"`` (enhancing_plus_necrosis).
            Values: flat float32 arrays or None.
        """
        self._prob_maps = dict(prob_maps)

    def set_tumor_core_flat(self, tumor_core_flat: np.ndarray) -> None:
        """Set flattened tumor core mask for distance filter."""
        self.tumor_core_flat = np.asarray(tumor_core_flat, dtype=bool).ravel()

    def set_training_masks(
        self,
        train_pos_flat: np.ndarray,
        train_neg_flat: np.ndarray,
    ) -> None:
        """Set training masks used for the training-region overlay NIfTI output."""
        self.train_pos_flat = np.asarray(train_pos_flat, dtype=bool).ravel()
        self.train_neg_flat = np.asarray(train_neg_flat, dtype=bool).ravel()

    # ------------------------------------------------------------------
    # Joint probability map
    # ------------------------------------------------------------------

    def compute_joint_probability_map(
        self, prob_maps: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Average available channel maps into a joint probability map.

        Each voxel in the test region receives a single joint score equal to
        the mean across all registered channel maps. This score is used for
        percentile thresholding, connected-component labelling, and the
        cluster z-test. Individual channel scores are retained separately for
        the three-channel qualifier step.

        Parameters
        ----------
        prob_maps : dict with flat arrays (one per channel)

        Returns
        -------
        np.ndarray, float32 — joint (mean) probability map
        """
        available = [v for v in prob_maps.values() if v is not None]
        if len(available) >= 2:
            return np.mean(np.stack(available, axis=0), axis=0).astype(np.float32)
        if len(available) == 1:
            return available[0].astype(np.float32)
        # No channel maps — load main prob map
        img = nib.load(str(self.prob_map_path))
        return img.get_fdata().ravel().astype(np.float32)

    # ------------------------------------------------------------------
    # Three-channel qualifier
    # ------------------------------------------------------------------

    def _compute_channel_qualifier(
        self, cluster_mask_flat: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute mean probability per channel inside the cluster and classify the
        cluster tissue type.

        Matches original deepneurobold trial_visualizer._compute_channel_qualifier:
        dominant_class = whichever channel has the highest mean probability.
        Values: "necrosis" | "enhancing" | "enhancing_plus_necrosis" | "unknown"

        Returns
        -------
        dict with keys:
            prob_necrosis, prob_enhancing, prob_enh_nec  — mean P per channel
            dominant_class — run label with highest mean P inside the cluster
        """
        # Maps run_name → tissue label (mirrors original _RUN_TO_LABEL)
        _RUN_TO_LABEL_LOCAL: Dict[str, str] = {
            "run_001": "necrosis",
            "run_002": "enhancing",
            "run_003": "enhancing_plus_necrosis",
        }

        result: Dict[str, Any] = {
            "prob_necrosis":  float("nan"),
            "prob_enhancing": float("nan"),
            "prob_enh_nec":   float("nan"),
            "dominant_class": "unknown",
        }
        if not np.any(cluster_mask_flat):
            return result

        channel_values: Dict[str, float] = {}
        for run_name, label in _RUN_TO_LABEL_LOCAL.items():
            prob = self._prob_maps.get(run_name)
            if prob is not None:
                channel_values[label] = float(np.mean(prob[cluster_mask_flat]))

        if "necrosis" in channel_values:
            result["prob_necrosis"]  = channel_values["necrosis"]
        if "enhancing" in channel_values:
            result["prob_enhancing"] = channel_values["enhancing"]
        if "enhancing_plus_necrosis" in channel_values:
            result["prob_enh_nec"]   = channel_values["enhancing_plus_necrosis"]

        # dominant_class = the label with the highest mean probability
        # (identical to original: max(channel_values, key=channel_values.get))
        if channel_values:
            result["dominant_class"] = max(channel_values, key=channel_values.get)

        return result

    # ------------------------------------------------------------------
    # NIfTI helpers
    # ------------------------------------------------------------------

    def _save_mask_nifti(self, mask_flat: np.ndarray, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vol = np.asarray(mask_flat, dtype=bool).reshape(self.vol_shape_3d).astype(np.uint8)
        nib.save(nib.Nifti1Image(vol, self.ref_img.affine, self.ref_img.header), str(out_path))

    def _save_float_nifti(self, data_flat: np.ndarray, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vol = np.asarray(data_flat, dtype=np.float32).reshape(self.vol_shape_3d)
        nib.save(nib.Nifti1Image(vol, self.ref_img.affine, self.ref_img.header), str(out_path))

    def _save_label_nifti(self, data_flat: np.ndarray, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vol = np.asarray(data_flat, dtype=np.uint8).reshape(self.vol_shape_3d)
        nib.save(nib.Nifti1Image(vol, self.ref_img.affine, self.ref_img.header), str(out_path))

    # ------------------------------------------------------------------
    # Core cluster detection
    # ------------------------------------------------------------------

    def detect_clusters(
        self,
        joint_prob_flat: np.ndarray,
        test_region_flat: np.ndarray,
        tumor_core_flat: Optional[np.ndarray],
        percentile: int,
    ) -> pd.DataFrame:
        """
        Threshold → connected components → size/distance/FDR filters → rank.

        Parameters
        ----------
        joint_prob_flat : np.ndarray
        test_region_flat : np.ndarray, bool
        tumor_core_flat : np.ndarray or None
        percentile : int
            Probability percentile used as threshold (e.g. 90 or 95).

        Returns
        -------
        pd.DataFrame ranked by score (may be empty).
        """
        n_vox = len(self.brain_mask_flat)

        # Threshold is computed adaptively from the joint map within the test
        # region, so it reflects the local probability distribution rather than
        # the global brain distribution.
        search_mask = test_region_flat & self.brain_mask_flat
        test_region_probs = joint_prob_flat[search_mask]

        if test_region_probs.size == 0:
            print(f"  [p{percentile}] test region is empty — skipping.")
            return pd.DataFrame()

        thr = float(np.percentile(test_region_probs, percentile))
        print(f"  [p{percentile}] threshold={thr:.4f} (from test region, joint map)")

        # Connected components on joint prob map
        initial_mask = (joint_prob_flat >= thr) & search_mask
        mask_3d = initial_mask.reshape(self.vol_shape_3d)
        structure = ndimage.generate_binary_structure(3, 2)
        labeled, n_features = ndimage.label(mask_3d, structure=structure)
        print(f"  [p{percentile}] n_raw_clusters={n_features}")

        # Distance map from tumor core (for FILTER 2)
        if tumor_core_flat is not None:
            core_3d = tumor_core_flat.reshape(self.vol_shape_3d).astype(bool)
            dist_to_core = ndimage.distance_transform_edt(~core_3d, sampling=self.voxel_size_mm)
            dist_to_core[core_3d] = 0.0
        else:
            dist_to_core = None

        # Background statistics for z-test (computed from joint map)
        bg_probs = joint_prob_flat[search_mask & ~initial_mask]
        bg_mean  = float(np.mean(bg_probs)) if bg_probs.size > 0 else 0.0
        bg_std   = max(float(np.std(bg_probs)) if bg_probs.size > 0 else 1.0, 1e-9)

        candidates: List[Dict[str, Any]] = []

        if n_features > 0:
            sizes = ndimage.sum(mask_3d, labeled, range(1, n_features + 1))
            for ci, size in enumerate(sizes, start=1):

                # FILTER 1: minimum cluster size
                if size < self.min_cluster_size:
                    continue

                cluster_mask   = (labeled == ci)
                cluster_flat   = cluster_mask.ravel()
                # Use joint map for per-cluster statistics
                cluster_probs  = joint_prob_flat[cluster_flat]
                mean_prob      = float(np.mean(cluster_probs))
                max_prob       = float(np.max(cluster_probs))

                if dist_to_core is not None:
                    min_dist = float(np.min(dist_to_core[cluster_mask]))
                else:
                    min_dist = float("nan")

                # FILTER 2: minimum distance from tumor core
                if not np.isnan(min_dist) and min_dist < self.min_dist_from_core_mm:
                    continue

                # Z-test
                z_stat = (mean_prob - bg_mean) / (bg_std / max(np.sqrt(int(size)), 1))
                p_raw  = float(scipy_stats.norm.sf(z_stat))

                # Adaptive percentile rank within test region
                pct_rank = float(scipy_stats.percentileofscore(test_region_probs, mean_prob))

                # Three-channel qualifier (matches original trial_visualizer)
                channel_q = self._compute_channel_qualifier(cluster_flat)

                edema_3d = self.edema_flat.reshape(self.vol_shape_3d)
                candidates.append({
                    "cluster_id":       ci,
                    "size_voxels":      int(size),
                    "volume_ml":        round(int(size) * self._voxel_vol_ml, 3),
                    "mean_prob":        mean_prob,
                    "max_prob":         max_prob,
                    "min_dist_mm":      min_dist,
                    "p_raw":            p_raw,
                    "pct_rank_in_test": round(pct_rank, 2),
                    "prob_necrosis":    channel_q["prob_necrosis"],
                    "prob_enhancing":   channel_q["prob_enhancing"],
                    "prob_enh_nec":     channel_q["prob_enh_nec"],
                    "dominant_class":   channel_q["dominant_class"],
                    "in_edema":         bool(np.any(cluster_mask & edema_3d)),
                    "in_ring":          bool(np.any(cluster_mask & ~edema_3d)),
                })

        if not candidates:
            print(f"  [p{percentile}] 0 clusters passed filters.")
            return pd.DataFrame()

        df = pd.DataFrame(candidates)

        # Benjamini-Hochberg FDR correction
        p_adj, rejected = _bh_fdr(df["p_raw"].values, alpha=self.fdr_alpha)
        df["p_bh_adj"] = p_adj
        df["sig_fdr"]  = rejected

        # Score and rank
        df["score"] = df["mean_prob"] * df["size_voxels"] / (df["min_dist_mm"] + 1.0)
        df.sort_values("score", ascending=False, inplace=True)
        df = df.head(self.max_clusters).reset_index(drop=True)
        df["rank"] = df.index + 1

        return df

    # ------------------------------------------------------------------
    # Full detection pipeline
    # ------------------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:
        """
        Run the full hotspot detection pipeline and save NIfTI outputs.

        Returns
        -------
        dict
            Summary statistics per percentile.
        """
        return self.detect()

    def detect(self) -> Dict[str, Any]:
        """
        Load probability map, compute joint map, detect and save hotspots.

        Returns
        -------
        dict with per-percentile hotspot counts and file paths.
        """
        out_dir = Path(self.run_dir) / "ensemble"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load main probability map
        if not self.prob_map_path.exists():
            raise FileNotFoundError(f"Probability map not found: {self.prob_map_path}")
        prob_flat = nib.load(str(self.prob_map_path)).get_fdata().ravel().astype(np.float32)

        if len(prob_flat) != len(self.brain_mask_flat):
            raise RuntimeError(
                f"Probability map length mismatch: "
                f"{len(prob_flat)} vs brain_mask {len(self.brain_mask_flat)}"
            )

        # Joint probability map from all available channels
        joint_prob_flat = self.compute_joint_probability_map(self._prob_maps or {})
        if len(joint_prob_flat) != len(prob_flat):
            joint_prob_flat = prob_flat  # fallback

        # Save joint map so it can be opened in FSLeyes
        self._save_float_nifti(joint_prob_flat, out_dir / "prob_joint_three_channel.nii.gz")
        print(f"[hotspot] saved joint prob map: prob_joint_three_channel.nii.gz")

        # Save individual channel maps in the ensemble folder for direct comparison
        # (restricted to test region so healthy tissue is suppressed)
        test_mask = self.test_region_flat & self.brain_mask_flat
        n_vox = len(self.brain_mask_flat)
        for run_name, fname in [
            ("run_001", "prob_necrosis_in_test_region.nii.gz"),
            ("run_002", "prob_enhancing_in_test_region.nii.gz"),
            ("run_003", "prob_enh_nec_in_test_region.nii.gz"),
        ]:
            chan_prob = self._prob_maps.get(run_name)
            if chan_prob is not None:
                out_flat = np.zeros(n_vox, dtype=np.float32)
                out_flat[test_mask] = chan_prob[test_mask]
                self._save_float_nifti(out_flat, out_dir / fname)

        # Tumor activity score map: P(enhancing) / (P(enhancing) + P(necrosis) + ε)
        # Only defined within the test region; zero elsewhere.
        pm_nec = self._prob_maps.get("run_001")
        pm_enh = self._prob_maps.get("run_002")
        if pm_nec is not None and pm_enh is not None:
            activity_flat = np.zeros(n_vox, dtype=np.float32)
            activity_flat[test_mask] = (
                pm_enh[test_mask] / (pm_enh[test_mask] + pm_nec[test_mask] + 1e-9)
            ).astype(np.float32)
            self._save_float_nifti(activity_flat, out_dir / "tumor_activity_score.nii.gz")
            print(f"[hotspot] saved tumor_activity_score.nii.gz"
                  f"  (1.0=active tumor, 0.0=necrosis, within test region)")

        summary: Dict[str, Any] = {}

        for p in self.percentiles:
            print(f"\n[hotspot] percentile={p}")

            df = self.detect_clusters(
                joint_prob_flat=joint_prob_flat,
                test_region_flat=self.test_region_flat,
                tumor_core_flat=self.tumor_core_flat,
                percentile=p,
            )

            # Save CSV
            csv_path = out_dir / f"hotspot_ranking_{p}.csv"
            df.to_csv(csv_path, index=False)

            valid_ids = df["cluster_id"].tolist() if not df.empty else []
            n_clusters = len(valid_ids)

            print(f"  [p{p}] {n_clusters} hotspot region(s) after all filters")
            if not df.empty:
                for _, row in df.iterrows():
                    act = row.get("tumor_activity_score", float("nan"))
                    act_str = f"{act:.2f}" if not np.isnan(act) else "n/a"
                    print(
                        f"    rank {int(row['rank'])}: "
                        f"{int(row['size_voxels'])} vox / {row['volume_ml']:.2f} ml  "
                        f"prob={row['mean_prob']:.3f}  "
                        f"dist={row['min_dist_mm']:.1f} mm  "
                        f"pct_rank={row['pct_rank_in_test']:.1f}  "
                        f"type={row['dominant_class']} (activity={act_str})  "
                        f"sig_fdr={row['sig_fdr']}"
                    )

            # Rebuild labeled volume for NIfTI outputs
            search_mask  = self.test_region_flat & self.brain_mask_flat
            thr          = float(np.percentile(joint_prob_flat[search_mask], p))
            initial_mask = (joint_prob_flat >= thr) & search_mask
            mask_3d      = initial_mask.reshape(self.vol_shape_3d)
            structure    = ndimage.generate_binary_structure(3, 2)
            labeled, _   = ndimage.label(mask_3d, structure=structure)

            n_vox = len(self.brain_mask_flat)

            # NIfTI 1: binary mask
            final_flat = np.isin(labeled, valid_ids).ravel() if valid_ids else np.zeros(n_vox, dtype=bool)
            self._save_mask_nifti(final_flat, out_dir / f"hotspot_{p}.nii.gz")

            # NIfTI 2: ranked labels (1/2/3)
            ranked_flat = np.zeros(n_vox, dtype=np.uint8)
            if not df.empty:
                for _, row in df.iterrows():
                    cm = np.isin(labeled, [int(row["cluster_id"])]).ravel()
                    ranked_flat[cm] = int(row["rank"])
            self._save_label_nifti(ranked_flat, out_dir / f"hotspot_{p}_ranked.nii.gz")

            # NIfTI 3: training overlay (1/2/3=rank, 4=train+, 5=train-)
            if self.train_pos_flat is not None and self.train_neg_flat is not None:
                overlay = np.zeros(n_vox, dtype=np.uint8)
                overlay[self.train_neg_flat] = 5
                overlay[self.train_pos_flat] = 4
                if not df.empty:
                    for _, row in df.iterrows():
                        cm = np.isin(labeled, [int(row["cluster_id"])]).ravel()
                        overlay[cm] = int(row["rank"])
                self._save_label_nifti(overlay, out_dir / f"hotspot_{p}_with_training_overlay.nii.gz")

            # NIfTIs 4-6: per-channel prob maps restricted to hotspot voxels
            if not df.empty:
                cluster_union = np.isin(labeled, valid_ids).ravel()
                for run_name, (label_name, fname) in {
                    "run_001": ("necrosis",               f"hotspot_{p}_prob_necrosis.nii.gz"),
                    "run_002": ("enhancing",              f"hotspot_{p}_prob_enhancing.nii.gz"),
                    "run_003": ("enhancing_plus_necrosis", f"hotspot_{p}_prob_enh_nec.nii.gz"),
                }.items():
                    chan_prob = self._prob_maps.get(run_name)
                    if chan_prob is not None:
                        chan_flat = np.zeros(n_vox, dtype=np.float32)
                        chan_flat[cluster_union] = chan_prob[cluster_union]
                        self._save_float_nifti(chan_flat, out_dir / fname)

            summary[f"p{p}_n_clusters"]       = n_clusters
            summary[f"p{p}_hotspot_voxels"]   = int(np.sum(final_flat))
            summary[f"p{p}_ranking_csv"]      = str(csv_path)
            summary[f"p{p}_binary_nifti"]     = str(out_dir / f"hotspot_{p}.nii.gz")
            summary[f"p{p}_ranked_nifti"]     = str(out_dir / f"hotspot_{p}_ranked.nii.gz")

        print(f"\n[hotspot] detection complete. output: {out_dir}")
        return summary

    def validate(self) -> bool:
        return self.prob_map_path.exists()
