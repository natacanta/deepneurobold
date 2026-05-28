"""
analysis.clustering.engine
==========================
Concrete clustering engines — ports ``deepneurobold/analysis/clustering/engines.py``.

Engines
-------
- BoldClustering        : unsupervised (KMeans / GMM / HDBSCAN) on BOLD features
- MultiModalClustering  : unsupervised on BOLD + structural intensities + coordinates
- GuidedClustering      : supervised multinomial logistic regression using seed regions
- GuidedIterative       : thin alias of GuidedClustering (single-pass for now)
- ClusteringEngine      : public dispatcher that implements BaseClustering ABC
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import BaseClustering
from .features import extract_bold_features
from .labels import build_masks_from_seg as _build_masks_from_seg

# Optional HDBSCAN
try:
    import hdbscan as _hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    _hdbscan = None  # type: ignore
    HAS_HDBSCAN = False

# Optional SciPy morphology
try:
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        distance_transform_edt,
        generate_binary_structure,
    )
    HAS_SCIPY = True
except Exception:
    binary_dilation = None           # type: ignore
    binary_erosion = None            # type: ignore
    distance_transform_edt = None    # type: ignore
    generate_binary_structure = None # type: ignore
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cap_pca_components(n_samples: int, n_features: int, cap: int = 10) -> int:
    return max(1, min(cap, n_features, n_samples - 1))


def _labels_1d_to_3d(labels: np.ndarray, mask3d: np.ndarray) -> np.ndarray:
    out = np.zeros(mask3d.size, dtype=np.int16)
    out[mask3d.ravel()] = labels.astype(np.int16) + 1
    return out.reshape(mask3d.shape)


def _clip_robust(X: np.ndarray, zlimit: float = 6.0) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Z  = np.clip((X - mu) / sd, -zlimit, zlimit)
    return Z * sd + mu


def _save_nifti(ref_img: nib.Nifti1Image, data: np.ndarray, path: Path,
                dtype=np.float32) -> None:
    _eprint(f"[SAVE] {path}")
    nib.save(nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header), str(path))


def _ensure_same_shape(name: str, arr: np.ndarray, ref_shape: tuple) -> None:
    if tuple(arr.shape) != tuple(ref_shape):
        raise RuntimeError(
            f"[SHAPE MISMATCH] {name}: got {arr.shape}, expected {ref_shape}. "
            "Ensure all *_inT1*.nii.gz share the same grid."
        )


def _summarize_bool(name: str, m: np.ndarray) -> None:
    _eprint(f"[MASK] {name}: shape={m.shape}, dtype={m.dtype}, sum={int(m.sum())}")


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Cluster quality metrics
# ---------------------------------------------------------------------------

def _cluster_quality_metrics(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 50_000,
    seed: int = 42,
) -> Dict[str, float]:
    out = {
        "silhouette":         float("nan"),
        "calinski_harabasz":  float("nan"),
        "davies_bouldin":     float("nan"),
    }
    y = np.asarray(y).reshape(-1)
    if X.shape[0] != y.shape[0]:
        return out
    if np.unique(y).size < 2 or np.unique(y).size >= X.shape[0]:
        return out
    idx = np.arange(X.shape[0])
    if X.shape[0] > max_samples:
        idx = np.random.RandomState(seed).choice(idx, size=max_samples, replace=False)
    Xs, ys = X[idx], y[idx]
    for key, fn in [
        ("silhouette",        lambda: silhouette_score(Xs, ys, metric="euclidean")),
        ("calinski_harabasz", lambda: calinski_harabasz_score(Xs, ys)),
        ("davies_bouldin",    lambda: davies_bouldin_score(Xs, ys)),
    ]:
        try:
            out[key] = float(fn())
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Core clustering fit
# ---------------------------------------------------------------------------

def _fit_cluster(
    X: np.ndarray,
    k: int,
    algo: str = "gmm",
    seed: int = 42,
    restarts: int = 1,
) -> np.ndarray:
    algo     = algo.lower().strip()
    restarts = max(1, int(restarts))

    if algo == "hdbscan":
        if not HAS_HDBSCAN:
            raise RuntimeError("hdbscan not installed. Use gmm or kmeans.")
        mcs    = max(20, int(0.005 * len(X)))
        hb     = _hdbscan.HDBSCAN(min_cluster_size=mcs, allow_single_cluster=True)
        return hb.fit_predict(X)

    best_labels, best_score, best_run = None, -np.inf, -1
    for r in range(restarts):
        s = seed + r
        if algo == "kmeans":
            labels = KMeans(n_clusters=k, n_init=10, max_iter=300,
                            random_state=s).fit_predict(X)
        elif algo == "gmm":
            labels = GaussianMixture(
                n_components=k, covariance_type="full", random_state=s,
                n_init=3, reg_covar=1e-6, max_iter=500, init_params="kmeans",
            ).fit_predict(X)
        else:
            raise ValueError(f"Unsupported algo: {algo}")

        ch = _cluster_quality_metrics(X, labels, max_samples=20_000, seed=s).get(
            "calinski_harabasz", float("nan")
        )
        _eprint(
            f"[CLUSTER] {algo.upper()} restart {r+1}/{restarts} seed={s} "
            f"CH={ch:.3f} sizes={dict(zip(*np.unique(labels, return_counts=True)))}"
        )
        if np.isfinite(ch) and ch > best_score:
            best_score, best_labels, best_run = ch, labels, r

    if best_labels is None:
        raise RuntimeError(f"{algo} failed to produce a valid clustering.")
    _eprint(f"[CLUSTER] selected restart {best_run+1}/{restarts} CH={best_score:.3f}")
    return best_labels


# ---------------------------------------------------------------------------
# Image search helpers
# ---------------------------------------------------------------------------

def _find_in_t1(t1_dir: Path, kind: str) -> Optional[Path]:
    table: Dict[str, List[str]] = {
        "t1":    ["t1_inT1_ants_geo.nii.gz", "T1_inT1_ants_geo.nii.gz",
                  "t1_inT1_ants.nii.gz", "T1_inT1_ants.nii.gz",
                  "t1_brain.nii.gz", "t1.nii.gz"],
        "t1ce":  ["t1ce_inT1_ants_geo.nii.gz", "T1CE_inT1_ants_geo.nii.gz",
                  "t1ce_inT1_ants.nii.gz", "T1CE_inT1_ants.nii.gz"],
        "t2":    ["t2_inT1_ants_geo.nii.gz", "T2_inT1_ants_geo.nii.gz",
                  "t2_inT1_ants.nii.gz", "T2_inT1_ants.nii.gz"],
        "flair": ["flair_inT1_ants_geo.nii.gz", "FLAIR_inT1_ants_geo.nii.gz",
                  "flair_inT1_ants.nii.gz", "FLAIR_inT1_ants.nii.gz"],
    }
    for name in table.get(kind, []):
        p = t1_dir / name
        if p.exists():
            return p
    return None


def _z_from_img_path(p: Optional[Path], mask3d: np.ndarray) -> Optional[np.ndarray]:
    if p is None or not p.exists():
        return None
    arr = np.asarray(nib.load(str(p), mmap=True).dataobj)
    if arr.shape != mask3d.shape:
        _eprint(f"[WARN] Shape mismatch for {p.name}; skipping z-map.")
        return None
    mu = arr[mask3d].mean()
    sd = arr[mask3d].std() + 1e-6
    z  = ((arr - mu) / sd).astype(np.float32)
    return z


# ---------------------------------------------------------------------------
# ROI / CSF helpers
# ---------------------------------------------------------------------------

def _load_precomputed_masks(mask_dir: Path, ref_shape: tuple) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not mask_dir.exists():
        return out

    def _bool(name: str) -> Optional[np.ndarray]:
        p = mask_dir / name
        if not p.exists():
            return None
        arr = np.asarray(nib.load(str(p), mmap=True).dataobj)
        if arr.shape != ref_shape:
            _eprint(f"[WARN] {name} shape={arr.shape} != ref={ref_shape}")
            return None
        return (arr > 0).astype(bool)

    for key, name in [
        ("brain_healthy", "t1_brain_healthy.nii.gz"),
        ("healthy_mask",  "healthy_mask.nii.gz"),
    ]:
        m = _bool(name)
        if m is not None:
            out[key] = m

    csf = _bool("csf_mask_inT1.nii.gz")
    if csf is None:
        pve_p = mask_dir / "csf_healthy_pve.nii.gz"
        if pve_p.exists():
            arr = np.asarray(nib.load(str(pve_p), mmap=True).dataobj, dtype=np.float32)
            if arr.shape == ref_shape:
                csf = (arr >= 0.5)
    if csf is not None:
        out["csf_mask"] = csf.astype(bool)

    for key, fname in [("gm_pve", "gm_healthy_pve.nii.gz"),
                        ("wm_pve", "wm_healthy_pve.nii.gz")]:
        p = mask_dir / fname
        if p.exists():
            arr = np.asarray(nib.load(str(p), mmap=True).dataobj, dtype=np.float32)
            if arr.shape == ref_shape:
                out[key] = np.clip(arr, 0.0, 1.0)

    gm_p = out.get("gm_pve")
    wm_p = out.get("wm_pve")
    if gm_p is not None and wm_p is not None:
        parenchyma = (gm_p + wm_p) >= 0.3
        out["gmwm_parenchyma"] = parenchyma.astype(bool)
        if "brain_healthy" not in out:
            out["brain_healthy"] = parenchyma.astype(bool)

    return out


def _estimate_csf_mask(
    pre: Dict[str, np.ndarray],
    brain_mask: np.ndarray,
    csf_thr: float = 0.7,
    remove_csf: bool = True,
) -> np.ndarray:
    if not remove_csf:
        return np.zeros_like(brain_mask, dtype=bool)
    csf = pre.get("csf_mask")
    if csf is not None:
        return csf.astype(bool) & brain_mask
    csf_p = pre.get("csf_pve")
    if csf_p is not None:
        return ((csf_p > csf_thr) & brain_mask).astype(bool)
    return np.zeros_like(brain_mask, dtype=bool)


def _coords_from_mask(mask3d: np.ndarray, weight: float = 0.2) -> Tuple[np.ndarray, List[str]]:
    ijk   = np.argwhere(mask3d).astype(np.float32)
    scale = np.array(mask3d.shape, dtype=np.float32)[None, :]
    return (ijk / scale) * float(weight), ["x_norm", "y_norm", "z_norm"]


# ---------------------------------------------------------------------------
# Internal base for standalone engines
# ---------------------------------------------------------------------------

class _ClusteringBase:
    """
    Shared infrastructure for BoldClustering, MultiModalClustering, GuidedClustering.

    Constructor mirrors the original deepneurobold.BaseClustering.
    """

    DEFAULT_TUMOR_LABEL_MAP = {"NECROSIS": 1, "CE": 3, "EDEMA": 2}

    def __init__(
        self,
        pdir: Path,
        *,
        k: int = 4,
        seed: int = 42,
        cfg: Optional[dict] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        self.pdir     = Path(pdir)
        self.pre      = self.pdir / "PREPROCESSING"
        self.t1_dir   = self.pre / "t1"
        self.mask_dir = self.pre / "mask"
        self.bold_dir = self.pre / "bold"
        self.k        = int(k)
        self.seed     = int(seed)
        self.cfg      = cfg or {}
        self.posthoc  = bool(self.cfg.get("posthoc", False))

        self.mask_path = self._find_brain_mask()

        trial_name  = self.cfg.get("trial", "trial1")
        roi_name    = self.cfg.get("roi") or "brain"
        method_name = self.cfg.get("method", "generic")
        self.trial_name  = trial_name
        self.roi_name    = roi_name
        self.method_name = method_name

        if save_dir is not None:
            self.out = save_dir
        else:
            self.out = (
                self.pdir / "Analysis" / "Clusters"
                / trial_name / f"ROI_{roi_name}" / method_name
            )
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolvers / loaders
    # ------------------------------------------------------------------

    def _find_brain_mask(self) -> Optional[Path]:
        for p in [
            self.t1_dir / "t1_brain_mask.nii.gz",
            self.t1_dir / "t1_brain_mask.nii",
            self.mask_dir / "t1_brain_mask.nii.gz",
            self.mask_dir / "t1_brain_mask.nii",
        ]:
            if p.exists():
                return p
        return None

    def _find_segmentation_in_t1(self) -> Optional[Path]:
        for name in ("Segmentation_in_T1.nii.gz", "Segmentation_in_T1.nii",
                     "Segmentation.nii.gz", "Segmentation.nii"):
            p = self.mask_dir / name
            if p.exists():
                return p
        return None

    def _load_bold_4d(self, mmap: bool = False) -> Tuple[nib.Nifti1Image, np.ndarray]:
        for name in ("BOLD_brain_inT1.nii.gz", "BOLD_inT1.nii.gz"):
            p = self.bold_dir / name
            if p.exists():
                img = nib.load(str(p))
                arr = np.asanyarray(img.dataobj) if mmap else np.asarray(img.dataobj)
                return img, arr
        raise FileNotFoundError(f"No BOLD_inT1 4D file found under {self.bold_dir}")

    def _load_mask_array(self, shape3d: tuple) -> np.ndarray:
        if self.mask_path and self.mask_path.exists():
            img = nib.load(str(self.mask_path))
            m   = np.asarray(img.dataobj) > 0
            if m.shape != shape3d:
                _eprint(f"[WARN] Mask shape {m.shape} != data {shape3d}; using full volume.")
                m = np.ones(shape3d, dtype=bool)
        else:
            _eprint(f"[WARN] No brain mask found for {self.pdir.name}; using full volume.")
            m = np.ones(shape3d, dtype=bool)

        # Extend with any voxel that is segmented
        seg_p = self._find_segmentation_in_t1()
        if seg_p:
            lab = np.asarray(nib.load(str(seg_p), mmap=True).dataobj, dtype=np.int32)
            if lab.shape == shape3d:
                m = m | (lab > 0)
        return m.astype(bool)

    # ------------------------------------------------------------------
    # Output saving
    # ------------------------------------------------------------------

    def _save_outputs(
        self,
        ref_img: nib.Nifti1Image,
        labels3d: np.ndarray,
        features2d: np.ndarray,
        meta: Dict,
    ) -> None:
        out_map  = self.out / f"cluster_k{self.k}.nii.gz"
        out_npz  = self.out / f"features_k{self.k}.npz"
        out_json = self.out / f"report_k{self.k}.json"

        nib.save(
            nib.Nifti1Image(labels3d.astype(np.int16), ref_img.affine, ref_img.header),
            str(out_map),
        )
        np.savez_compressed(out_npz, features=features2d)
        out_json.write_text(json.dumps(meta, indent=2))

        _eprint(f"[{_timestamp()}] Saved: {out_map}")

        summary_csv = self.out.parent / "FullConfigurationSummary.csv"
        row = {
            "Patient": self.pdir.name,
            "Trial": self.out.parent.name,
            "Method": self.out.name,
            "k": self.k,
            "Algo": meta.get("algo", ""),
            "Mode": meta.get("mode", ""),
            "Voxels": meta.get("n_vox", 0),
            "PCA_comp": meta.get("pca_components", 0),
            "Elapsed_sec": round(meta.get("elapsed_sec", 0), 2),
        }
        if summary_csv.exists():
            with open(summary_csv, "r", newline="") as fh:
                headers = next(csv.reader(fh), None) or list(row.keys())
        else:
            headers = list(row.keys())
        with open(summary_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=headers)
            if not summary_csv.exists() or summary_csv.stat().st_size == 0:
                w.writeheader()
            w.writerow({h: row.get(h, meta.get(h, "")) for h in headers})


# ---------------------------------------------------------------------------
# 1) BoldClustering
# ---------------------------------------------------------------------------

class BoldClustering(_ClusteringBase):
    """Unsupervised BOLD-only clustering (KMeans / GMM / HDBSCAN)."""

    def __init__(
        self,
        *args,
        roi: str = "brain",
        algo: str = "gmm",
        scale_feats: bool = True,
        chunk_size: Optional[int] = None,
        pca_cap: int = 10,
        zclip: Optional[float] = 6.0,
        coord_weight: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.roi          = roi
        self.algo         = algo
        self.scale_feats  = bool(scale_feats)
        self.chunk_size   = chunk_size
        self.pca_cap      = int(pca_cap)
        self.zclip        = zclip
        self.coord_weight = float(coord_weight)
        self._dbg: Dict[str, np.ndarray] = {}

    def _roi_mask(self, brain_mask: np.ndarray) -> np.ndarray:
        roi_name = (self.roi or "brain").lower().strip()
        if roi_name == "brain":
            roi = brain_mask.astype(bool)
        else:
            masks, _ = _build_masks_from_seg(self.pdir, erosion_iters=0, save_out=False)
            if roi_name == "tumor":
                roi = masks["tumor"].astype(bool) & brain_mask
            elif roi_name in ("edema", "flair_minus_ce"):
                roi = masks["edema"].astype(bool) & brain_mask
            elif roi_name in ("tumor_plus_edema", "tumor+edema", "tumor_edema"):
                roi = (masks["tumor"].astype(bool) | masks["edema"].astype(bool)) & brain_mask
            else:
                _eprint(f"[WARN] Unknown ROI '{roi_name}' → using brain")
                roi = brain_mask.astype(bool)
            if not roi.any():
                raise RuntimeError(f"BoldClustering ROI '{roi_name}' is empty.")

        pre = _load_precomputed_masks(self.mask_dir, brain_mask.shape)
        csf = _estimate_csf_mask(pre, brain_mask)
        roi = roi & ~csf
        self._dbg = {"roi_no_csf": roi, "csf_heuristic": csf}
        return roi

    def run(self) -> None:
        t0 = time.time()
        ref_img, bold = self._load_bold_4d(mmap=True)
        brain_mask    = self._load_mask_array(bold.shape[:3])
        roi_mask      = self._roi_mask(brain_mask)
        if not roi_mask.any():
            raise RuntimeError("Empty ROI.")

        self.out.mkdir(parents=True, exist_ok=True)

        feats  = extract_bold_features(
            bold, roi_mask.ravel(),
            dt=1.8, chunk_size=self.chunk_size or self.cfg.get("chunk_size"),
        )
        X_bold = feats["features"].astype(np.float32, copy=False)
        names  = feats["names"].tolist()

        blocks, bnames = [], []
        Xb = StandardScaler().fit_transform(X_bold) if self.scale_feats else X_bold
        blocks.append(Xb); bnames.extend(names)

        if self.coord_weight > 0:
            Xc, nc = _coords_from_mask(roi_mask, weight=self.coord_weight)
            blocks.append(Xc); bnames.extend(nc)

        X = np.concatenate(blocks, axis=1)
        if self.zclip and self.zclip > 0:
            X = _clip_robust(X, zlimit=float(self.zclip))

        vt = VarianceThreshold(threshold=1e-8)
        X  = vt.fit_transform(X)
        kept_names = [n for n, k in zip(bnames, vt.variances_ > 1e-8) if k]

        n_comp = _cap_pca_components(X.shape[0], X.shape[1], cap=self.pca_cap)
        pca    = PCA(n_components=n_comp, random_state=self.seed)
        Xr     = pca.fit_transform(X)

        restarts = int(self.cfg.get("cluster_restarts", 1))
        labels   = _fit_cluster(Xr, self.k, algo=self.algo,
                                 seed=self.seed, restarts=restarts)

        labels3d = _labels_1d_to_3d(labels, roi_mask)
        metrics  = _cluster_quality_metrics(Xr, labels)
        uniq, counts = np.unique(labels, return_counts=True)
        meta = {
            "mode": "bold", "algo": self.algo, "k": self.k, "roi": self.roi,
            "features": kept_names, "n_vox": int(labels.shape[0]),
            "pca_components": int(n_comp),
            "pca_var_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "cluster_sizes": {int(u): int(c) for u, c in zip(uniq, counts)},
            "elapsed_sec": float(time.time() - t0), "metrics": metrics,
        }
        self._save_outputs(ref_img, labels3d, X, meta)


# ---------------------------------------------------------------------------
# 2) MultiModalClustering
# ---------------------------------------------------------------------------

class MultiModalClustering(_ClusteringBase):
    """Unsupervised BOLD + structural + spatial clustering."""

    def __init__(
        self,
        *args,
        algo: str = "gmm",
        scale_feats: bool = True,
        chunk_size: Optional[int] = None,
        pca_cap: int = 10,
        zclip: Optional[float] = 6.0,
        roi: str = "brain",
        coord_weight: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.algo         = algo
        self.scale_feats  = bool(scale_feats)
        self.chunk_size   = chunk_size
        self.pca_cap      = int(pca_cap)
        self.zclip        = zclip
        self.roi          = roi
        self.coord_weight = float(coord_weight)

    def _select_roi_mask(self, brain_mask: np.ndarray) -> np.ndarray:
        roi_name = (self.roi or "brain").lower().strip()
        if roi_name == "brain":
            roi = brain_mask.astype(bool)
        else:
            masks, _ = _build_masks_from_seg(self.pdir, erosion_iters=0, save_out=False)
            if roi_name == "tumor":
                roi = masks["tumor"].astype(bool) & brain_mask
            elif roi_name in ("edema", "flair_minus_ce"):
                roi = masks["edema"].astype(bool) & brain_mask
            elif roi_name in ("tumor_plus_edema", "tumor+edema", "tumor_edema"):
                roi = (masks["tumor"] | masks["edema"]).astype(bool) & brain_mask
            else:
                _eprint(f"[WARN] Unknown ROI '{roi_name}' → using brain")
                roi = brain_mask.astype(bool)
            if not roi.any():
                raise RuntimeError(f"MultiModalClustering ROI '{roi_name}' is empty.")

        pre = _load_precomputed_masks(self.mask_dir, brain_mask.shape)
        csf = _estimate_csf_mask(pre, brain_mask)
        return (roi & ~csf).astype(bool)

    def run(self) -> None:
        t0 = time.time()
        ref_img, bold = self._load_bold_4d(mmap=True)
        brain_mask    = self._load_mask_array(bold.shape[:3])
        roi_mask      = self._select_roi_mask(brain_mask)
        self.out.mkdir(parents=True, exist_ok=True)

        feats  = extract_bold_features(
            bold, roi_mask.ravel(),
            dt=1.8, chunk_size=self.chunk_size or self.cfg.get("chunk_size"),
        )
        X_bold = feats["features"].astype(np.float32)
        names  = feats["names"].tolist()

        # Structural z-maps
        add_maps, add_names = [], []
        for nm, kind in [("t1", "t1"), ("t1ce", "t1ce"), ("t2", "t2"), ("flair", "flair")]:
            z = _z_from_img_path(_find_in_t1(self.t1_dir, kind), brain_mask)
            if z is not None:
                add_maps.append(z.ravel()[roi_mask.ravel()].reshape(-1, 1))
                add_names.append(f"z_{nm}")

        Xc, nc = _coords_from_mask(roi_mask, weight=self.coord_weight)

        blocks, bnames = [], []
        Xb = StandardScaler().fit_transform(X_bold) if self.scale_feats else X_bold
        blocks.append(Xb); bnames.extend(names)
        if add_maps:
            Xs = StandardScaler().fit_transform(np.concatenate(add_maps, axis=1)) \
                if self.scale_feats else np.concatenate(add_maps, axis=1)
            blocks.append(Xs); bnames.extend(add_names)
        blocks.append(Xc); bnames.extend(nc)

        X = np.concatenate(blocks, axis=1)
        if self.zclip and self.zclip > 0:
            X = _clip_robust(X, float(self.zclip))

        vt = VarianceThreshold(threshold=1e-8)
        X  = vt.fit_transform(X)
        kept_names = [n for n, k in zip(bnames, vt.variances_ > 1e-8) if k]

        np.savez(self.out / "features_multi_input.npz",
                 X=X, feature_names=np.array(kept_names),
                 roi_mask=roi_mask.astype(np.uint8))

        n_comp = _cap_pca_components(X.shape[0], X.shape[1], cap=self.pca_cap)
        pca    = PCA(n_components=n_comp, random_state=self.seed)
        Xr     = pca.fit_transform(X)

        restarts = int(self.cfg.get("cluster_restarts", 1))
        labels   = _fit_cluster(Xr, self.k, algo=self.algo, seed=self.seed, restarts=restarts)

        labels3d = _labels_1d_to_3d(labels, roi_mask)
        uniq, counts = np.unique(labels, return_counts=True)
        metrics = _cluster_quality_metrics(Xr, labels)
        meta = {
            "mode": "multi", "algo": self.algo, "k": self.k, "roi": self.roi,
            "features": kept_names, "n_vox": int(labels.shape[0]),
            "pca_components": int(n_comp),
            "cluster_sizes": {int(u): int(c) for u, c in zip(uniq, counts)},
            "elapsed_sec": float(time.time() - t0), "metrics": metrics,
        }
        self._save_outputs(ref_img, labels3d, X, meta)


# ---------------------------------------------------------------------------
# 3) GuidedClustering
# ---------------------------------------------------------------------------

class GuidedClustering(_ClusteringBase):
    """
    Supervised multinomial logistic regression using seed regions.

    Global class order: ['gm', 'wm', 'necro', 'enh', 'edema']
    Probability maps are saved per class plus a synthetic 'healthy' = gm+wm.
    """

    CLASS_ORDER: List[str] = ["gm", "wm", "necro", "enh", "edema"]

    def __init__(
        self,
        *args,
        scale_feats: bool = True,
        chunk_size: Optional[int] = None,
        pca_cap: int = 10,
        zclip: Optional[float] = 6.0,
        erosion_iters: int = 1,
        outside_constraint: bool = True,
        outside_dilate_iters: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.scale_feats          = bool(scale_feats)
        self.chunk_size           = chunk_size
        self.pca_cap              = int(pca_cap)
        self.zclip                = zclip
        self.erosion_iters        = int(erosion_iters)
        self.outside_constraint   = bool(outside_constraint)
        self.outside_dilate_iters = int(outside_dilate_iters)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_brain_mask(self, bold_shape: tuple) -> np.ndarray:
        p = self.t1_dir / "t1_brain_mask.nii.gz"
        if p.exists():
            arr = np.asarray(nib.load(str(p), mmap=True).dataobj)
            m   = (arr > 0)
            if m.shape == bold_shape:
                return m.astype(bool)
        return self._load_mask_array(bold_shape)

    # ------------------------------------------------------------------
    # Feature stack
    # ------------------------------------------------------------------

    def _stack_feats(
        self,
        bold: np.ndarray,
        brain_mask: np.ndarray,
        use_mask_flat: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        use_mask_flat = np.asarray(use_mask_flat, dtype=bool).ravel()

        feats = extract_bold_features(
            bold, use_mask_flat, dt=1.8,
            chunk_size=self.chunk_size or self.cfg.get("chunk_size"),
        )
        X     = feats["features"].astype(np.float32, copy=False)
        names = feats["names"].tolist()

        add_cols, add_names = [], []
        for nm, kind in [("z_t1", "t1"), ("z_t1ce", "t1ce"),
                         ("z_t2", "t2"), ("z_flair", "flair")]:
            z3d = _z_from_img_path(_find_in_t1(self.t1_dir, kind), brain_mask)
            if z3d is None:
                continue
            add_cols.append(z3d.ravel()[use_mask_flat].reshape(-1, 1))
            add_names.append(nm)

        # Optional supervised tumor-probability map
        sup_p = self.pdir / "Analysis" / "Supervised" / "supervised_prob_tumor_like.nii.gz"
        if sup_p.exists():
            try:
                sup = np.clip(
                    np.asarray(nib.load(str(sup_p)).dataobj, dtype=np.float32), 0.0, 1.0
                )
                _ensure_same_shape("supervised_prob_tumor_like", sup, brain_mask.shape)
                add_cols.append(sup.ravel()[use_mask_flat].reshape(-1, 1))
                add_names.append("sup_tumorprob")
            except Exception as e:
                _eprint(f"[WARN] supervised map skipped: {e}")

        # Signed distance to tumor boundary
        try:
            masks_all, _ = _build_masks_from_seg(self.pdir, erosion_iters=0, save_out=False)
            tumor = (masks_all["tumor"] & brain_mask)
            if HAS_SCIPY and distance_transform_edt is not None:
                dist_out = distance_transform_edt(~tumor)
                dist_in  = distance_transform_edt(tumor)
                sd = dist_out.astype(np.float32)
                sd[tumor] = -dist_in[tumor].astype(np.float32)
                add_cols.append(0.25 * sd.ravel()[use_mask_flat].reshape(-1, 1))
                add_names.append("signed_dist_tumor_x0.25")
        except Exception as e:
            _eprint(f"[WARN] distance prior skipped: {e}")

        if add_cols:
            X = np.concatenate([X, *add_cols], axis=1)
            names.extend(add_names)

        return X, names

    # ------------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------------

    def _make_seeds(
        self,
        erosion_iters: int,
        brain_mask: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        masks_seg, _ = _build_masks_from_seg(
            self.pdir, erosion_iters=erosion_iters, save_out=True
        )
        pre       = _load_precomputed_masks(self.mask_dir, brain_mask.shape)
        gm_pve    = pre.get("gm_pve")
        wm_pve    = pre.get("wm_pve")
        brain_h   = pre.get("brain_healthy", brain_mask.copy()).astype(bool)
        hlthy     = pre.get("healthy_mask")
        h_base    = (hlthy.astype(bool) & brain_h if hlthy is not None else brain_h.copy())
        h_base   &= ~masks_seg["tumor"].astype(bool)

        gm_thr, wm_thr = 0.7, 0.7
        gm_seed = ((gm_pve >= gm_thr) & h_base) if gm_pve is not None else np.zeros_like(brain_mask, dtype=bool)
        wm_seed = ((wm_pve >= wm_thr) & h_base) if wm_pve is not None else np.zeros_like(brain_mask, dtype=bool)

        if HAS_SCIPY and generate_binary_structure is not None and binary_erosion is not None and erosion_iters > 0:
            st      = generate_binary_structure(rank=3, connectivity=1)
            gm_seed = binary_erosion(gm_seed, structure=st, iterations=erosion_iters) if gm_seed.any() else gm_seed
            wm_seed = binary_erosion(wm_seed, structure=st, iterations=erosion_iters) if wm_seed.any() else wm_seed

        gm_seed &= brain_mask
        wm_seed &= brain_mask

        seeds = {
            "gm":    gm_seed,
            "wm":    wm_seed,
            "necro": masks_seg["seed_necro"].astype(bool) & brain_mask,
            "enh":   masks_seg["seed_enh"].astype(bool)   & brain_mask,
            "edema": masks_seg["seed_edema"].astype(bool) & brain_mask,
        }
        counts = {k: int((seeds[k]).sum()) for k in self.CLASS_ORDER}
        _eprint(f"[SEEDS] counts (BEFORE fallback) = {counts}")

        # Fallback for empty tumor seeds
        if any(counts[c] == 0 for c in ["necro", "enh", "edema"]):
            try:
                raw, _ = _build_masks_from_seg(self.pdir, erosion_iters=0, save_out=False)
                for cname in ["necro", "enh", "edema"]:
                    if counts[cname] > 0:
                        continue
                    raw_s = raw.get(f"seed_{cname}", np.zeros_like(brain_mask, dtype=bool))
                    raw_s = raw_s.astype(bool) & brain_mask & ~gm_seed & ~wm_seed
                    if raw_s.sum() > 0:
                        seeds[cname] = raw_s
            except Exception as e:
                _eprint(f"[WARN] seed fallback failed: {e}")

        counts = {k: int(seeds[k].sum()) for k in self.CLASS_ORDER}
        _eprint(f"[SEEDS] counts (AFTER fallback) = {counts}")
        return seeds, counts

    def _build_train_mask_and_y(
        self,
        seeds: Dict[str, np.ndarray],
        brain_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        train_flat = np.zeros(brain_mask.size, dtype=bool)
        for cname in self.CLASS_ORDER:
            train_flat |= (seeds[cname] & brain_mask).ravel()

        class_map = np.full(brain_mask.size, -1, dtype=np.int16)
        for ci, cname in enumerate(self.CLASS_ORDER):
            class_map[(seeds[cname] & brain_mask).ravel()] = ci

        train_idx = np.flatnonzero(train_flat)
        y_train   = class_map[train_idx].astype(np.int64)

        if (y_train < 0).any():
            raise RuntimeError(f"GUIDED: {int((y_train < 0).sum())} unlabeled train voxels.")
        return train_flat, y_train

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> None:
        t0 = time.time()
        ref_img, bold = self._load_bold_4d(mmap=True)
        brain_mask    = self._load_brain_mask(bold.shape[:3])
        self.out.mkdir(parents=True, exist_ok=True)

        pre     = _load_precomputed_masks(self.mask_dir, brain_mask.shape)
        csf     = _estimate_csf_mask(pre, brain_mask)
        roi_full = brain_mask & ~csf

        seeds, counts = self._make_seeds(self.erosion_iters, brain_mask)
        train_flat, y_train = self._build_train_mask_and_y(seeds, brain_mask)
        uniq = np.unique(y_train) if y_train.size else np.array([], dtype=int)

        if uniq.size < 2 and self.erosion_iters > 0:
            _eprint("[WARN] < 2 seed classes after erosion → retrying with erosion=0")
            seeds, counts = self._make_seeds(0, brain_mask)
            train_flat, y_train = self._build_train_mask_and_y(seeds, brain_mask)
            uniq = np.unique(y_train) if y_train.size else np.array([], dtype=int)

        if uniq.size < 2:
            raise RuntimeError("GUIDED: insufficient seed classes for multinomial LR.")

        X_train_raw, feat_names = self._stack_feats(bold, brain_mask, train_flat)

        if self.zclip and self.zclip > 0:
            X_train_raw = _clip_robust(X_train_raw, float(self.zclip))

        vt = VarianceThreshold(threshold=1e-8)
        X_train_vt = vt.fit_transform(X_train_raw)
        kept_names = [n for n, k in zip(feat_names, vt.variances_ > 1e-8) if k]

        scaler = StandardScaler() if self.scale_feats else None
        X_train_s = scaler.fit_transform(X_train_vt) if scaler else X_train_vt

        n_comp  = _cap_pca_components(X_train_s.shape[0], X_train_s.shape[1], cap=self.pca_cap)
        pca     = PCA(n_components=n_comp, random_state=self.seed)
        X_train_r = pca.fit_transform(X_train_s)

        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=500, n_jobs=-1, C=1.0,
        )
        clf.fit(X_train_r, y_train)
        class_to_col = {int(c): i for i, c in enumerate(clf.classes_)}

        # Full ROI prediction
        roi_flat = roi_full.ravel()
        X_all_raw, _ = self._stack_feats(bold, brain_mask, roi_flat)
        if self.zclip and self.zclip > 0:
            X_all_raw = _clip_robust(X_all_raw, float(self.zclip))
        X_all_vt = vt.transform(X_all_raw)
        X_all_s  = scaler.transform(X_all_vt) if scaler else X_all_vt
        X_all_r  = pca.transform(X_all_s)
        probs    = clf.predict_proba(X_all_r)

        # Hard constraint: suppress tumor probabilities outside dilated tumor
        if self.outside_constraint and HAS_SCIPY and binary_dilation is not None:
            try:
                masks_all, _ = _build_masks_from_seg(self.pdir, erosion_iters=0, save_out=False)
                tumor = (masks_all["tumor"] & brain_mask).astype(bool)
                st    = generate_binary_structure(rank=3, connectivity=1)
                tumor_dil = binary_dilation(tumor, structure=st,
                                            iterations=max(1, self.outside_dilate_iters))
                outside   = (~tumor_dil) & brain_mask
                roi_idx   = np.flatnonzero(roi_flat)
                out_idx   = np.flatnonzero(roi_flat & outside.ravel())
                if out_idx.size > 0:
                    rel = np.searchsorted(roi_idx, out_idx)
                    rel = rel[(rel >= 0) & (rel < probs.shape[0])]
                    tumor_cols = [
                        class_to_col[self.CLASS_ORDER.index(c)]
                        for c in ["necro", "enh", "edema"]
                        if self.CLASS_ORDER.index(c) in class_to_col
                    ]
                    if tumor_cols:
                        sub = probs[rel]
                        sub[:, tumor_cols] *= 0.05
                        probs[rel] = sub / (sub.sum(axis=1, keepdims=True) + 1e-8)
            except Exception as e:
                _eprint(f"[WARN] outside constraint failed: {e}")

        roi_idx = np.flatnonzero(roi_flat)
        y_pred_global = clf.classes_[probs.argmax(axis=1).astype(np.int16)].astype(np.int16)

        labels_full = np.zeros(brain_mask.size, dtype=np.int16)
        labels_full[roi_idx] = y_pred_global + 1
        labels3d = labels_full.reshape(brain_mask.shape)
        _save_nifti(ref_img, labels3d, self.out / "guided_labels.nii.gz", dtype=np.int16)

        # Per-class probability maps
        for cid_global, cname in enumerate(self.CLASS_ORDER):
            vol = np.zeros(brain_mask.size, dtype=np.float32)
            if cid_global in class_to_col:
                vol[roi_idx] = probs[:, class_to_col[cid_global]].astype(np.float32)
            _save_nifti(ref_img, vol.reshape(brain_mask.shape),
                        self.out / f"guided_prob_{cname}.nii.gz", dtype=np.float32)

        # Synthetic healthy = gm + wm
        gm_id, wm_id = self.CLASS_ORDER.index("gm"), self.CLASS_ORDER.index("wm")
        if gm_id in class_to_col and wm_id in class_to_col:
            h_vol = np.zeros(brain_mask.size, dtype=np.float32)
            h_vol[roi_idx] = (probs[:, class_to_col[gm_id]] +
                              probs[:, class_to_col[wm_id]]).astype(np.float32)
            _save_nifti(ref_img, h_vol.reshape(brain_mask.shape),
                        self.out / "guided_prob_healthy.nii.gz", dtype=np.float32)

        metrics = _cluster_quality_metrics(X_train_r, y_train)
        uniq_pred, counts_pred = np.unique(y_pred_global, return_counts=True)
        meta = {
            "mode": "guided", "classes": self.CLASS_ORDER,
            "synthetic_maps": ["healthy = gm+wm"],
            "n_train_vox": int(train_flat.sum()),
            "seed_counts": counts,
            "algo": "logreg_multinomial",
            "pca_components": int(n_comp),
            "pca_var_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "n_roi_vox": int(roi_idx.size),
            "cluster_sizes_global": {int(u): int(c) for u, c in zip(uniq_pred, counts_pred)},
            "elapsed_sec": float(time.time() - t0),
            "metrics_on_seeds": metrics,
            "erosion_iters": self.erosion_iters,
            "outside_constraint": self.outside_constraint,
        }
        (self.out / "guided_meta.json").write_text(json.dumps(meta, indent=2))
        _eprint(f"[{_timestamp()}] GuidedClustering done in {meta['elapsed_sec']:.1f}s")


# ---------------------------------------------------------------------------
# 4) GuidedIterative (alias)
# ---------------------------------------------------------------------------

class GuidedIterative(GuidedClustering):
    """
    Iterative GUIDED engine — currently single-pass GuidedClustering.

    Accepts extra knobs (max_loops, thr_add_tumor, thr_add_healthy) so that
    existing CLI callers do not break.
    """

    def __init__(
        self,
        *args,
        n_iters: int = 1,
        max_loops: int = 1,
        thr_add_tumor: float = 0.75,
        thr_add_healthy: float = 0.90,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_iters         = max(1, max_loops if max_loops > 0 else n_iters)
        self.thr_add_tumor   = float(thr_add_tumor)
        self.thr_add_healthy = float(thr_add_healthy)

    def run(self) -> None:
        _eprint(f"[GUIDED-ITER] n_iters={self.n_iters} (single-pass behaviour)")
        super().run()


# ---------------------------------------------------------------------------
# Public BaseClustering implementation
# ---------------------------------------------------------------------------

_ENGINE_MAP: Dict[str, type] = {
    "bold":           BoldClustering,
    "multi":          MultiModalClustering,
    "guided":         GuidedClustering,
    "guided_iter":    GuidedIterative,
    "guided_iterative": GuidedIterative,
}


class ClusteringEngine(BaseClustering):
    """
    Public dispatcher that implements the ``BaseClustering`` ABC.

    Reads ``config["mode"]`` (default: ``"bold"``) to select the internal
    engine.  All other config keys are passed through to the engine.

    Parameters
    ----------
    trial_dir : Path
        Patient directory (passed as *pdir* to the internal engine).
    config : dict, optional
        Keys forwarded to the internal engine:
        ``mode``, ``k``, ``seed``, ``roi``, ``algo``, ``trial``,
        ``method``, ``pca_cap``, ``zclip``, ``erosion_iters``, etc.
    """

    def __init__(self, trial_dir, config=None) -> None:
        super().__init__(trial_dir=trial_dir, config=config)
        self.pdir = self.trial_dir       # alias used by internal helpers

    def cluster(self) -> Dict[str, Any]:
        mode = str(self.get_config("mode") or "bold").lower().strip()
        if mode not in _ENGINE_MAP:
            raise ValueError(
                f"Unknown clustering mode '{mode}'. "
                f"Available: {sorted(_ENGINE_MAP)}"
            )
        engine_cls = _ENGINE_MAP[mode]

        k    = int(self.get_config("k") or 4)
        seed = int(self.get_config("seed") or 42)
        cfg  = dict(self.config or {})

        engine = engine_cls(
            self.pdir,
            k=k,
            seed=seed,
            cfg=cfg,
        )
        engine.run()
        return {"mode": mode, "pdir": str(self.pdir), "out": str(engine.out)}
