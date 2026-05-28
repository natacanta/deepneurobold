"""
data.sampling
=============
Train/validation split strategies for voxel-wise supervised learning.

Supports three modes:
  ``voxel_random``  — balanced random sampling, no spatial holdout
  ``patch_holdout`` — spatially contiguous patches held out for validation
  ``grid_block``    — regular grid blocks; one per class held out for validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from deepneurobold.core.base import BaseComponent

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
SamplingMode = Literal["voxel_random", "patch_holdout", "grid_block"]


@dataclass(frozen=True)
class SampleSplit:
    """Container for a train/validation index split."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    meta: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def _sample_equal(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    n_per_class: Optional[int],
    rs: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise RuntimeError("Cannot sample: pos or neg pool is empty.")
    if n_per_class is None:
        n = min(pos_idx.size, neg_idx.size)
    else:
        n = min(int(n_per_class), pos_idx.size, neg_idx.size)
        if n <= 0:
            raise ValueError("n_per_class must be > 0.")
    return rs.choice(pos_idx, size=n, replace=False), rs.choice(neg_idx, size=n, replace=False)


def _sample_neg_stratified(
    neg_idx: np.ndarray,
    bg_gm_flat: np.ndarray,
    bg_wm_flat: np.ndarray,
    n_neg: int,
    rs: np.random.RandomState,
    gm_fraction: float = 1.0 / 3.0,
) -> np.ndarray:
    """Stratified background sampling: 1/3 GM + 2/3 WM by default.

    Splits ``neg_idx`` (the post-holdout negative pool, already excluding CSF
    if ``build_train_masks`` was called with SynthSeg) into the GM-resident
    and WM-resident subsets, then samples ``round(n_neg * gm_fraction)`` from
    GM and the remainder from WM. Voxels that fall in neither GM nor WM are
    excluded from the draw (this should only happen if SynthSeg failed in
    those regions; in practice such voxels were also excluded from the
    healthy pool upstream).

    If either subset is too small to meet the target fraction, the deficit
    is back-filled from the other subset so the total still equals the
    requested ``n_neg``.

    Returns the selected flat indices (shape ``(n_neg,)``).
    """
    if neg_idx.size == 0:
        raise RuntimeError("Cannot sample: neg pool empty.")
    if not (0.0 < float(gm_fraction) < 1.0):
        raise ValueError(f"gm_fraction must be in (0, 1), got {gm_fraction}")

    gm_mask = np.asarray(bg_gm_flat, dtype=bool)
    wm_mask = np.asarray(bg_wm_flat, dtype=bool)
    in_gm = gm_mask[neg_idx]
    in_wm = wm_mask[neg_idx]
    gm_pool = neg_idx[in_gm & ~in_wm]
    wm_pool = neg_idx[in_wm & ~in_gm]

    target_gm = int(round(int(n_neg) * float(gm_fraction)))
    target_wm = int(n_neg) - target_gm

    # Back-fill if one tissue pool is short.
    pick_gm = min(target_gm, gm_pool.size)
    pick_wm = min(target_wm, wm_pool.size)
    deficit = int(n_neg) - (pick_gm + pick_wm)
    if deficit > 0:
        if wm_pool.size - pick_wm >= deficit:
            pick_wm += deficit
        elif gm_pool.size - pick_gm >= deficit:
            pick_gm += deficit
        else:
            raise RuntimeError(
                f"_sample_neg_stratified: not enough voxels — "
                f"requested={n_neg} gm_pool={gm_pool.size} wm_pool={wm_pool.size}"
            )

    out_gm = rs.choice(gm_pool, size=pick_gm, replace=False) if pick_gm else np.zeros(0, dtype=np.int64)
    out_wm = rs.choice(wm_pool, size=pick_wm, replace=False) if pick_wm else np.zeros(0, dtype=np.int64)
    return np.concatenate([out_gm, out_wm]).astype(np.int64)


def _coords_from_flat(idx: np.ndarray, vol_shape_3d: Tuple[int, int, int]) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).ravel()
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.column_stack(np.unravel_index(idx, vol_shape_3d)).astype(np.int64)


def _centers_from_mask(mask_flat: np.ndarray, vol_shape_3d: Tuple[int, int, int]) -> np.ndarray:
    idx = np.flatnonzero(mask_flat)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.column_stack(np.unravel_index(idx, vol_shape_3d)).astype(np.int64)


def _cube_indices(center_xyz: np.ndarray, vol_shape_3d: Tuple[int, int, int], half_width: int) -> np.ndarray:
    x0, y0, z0 = int(center_xyz[0]), int(center_xyz[1]), int(center_xyz[2])
    xs = np.arange(max(0, x0 - half_width), min(vol_shape_3d[0], x0 + half_width + 1))
    ys = np.arange(max(0, y0 - half_width), min(vol_shape_3d[1], y0 + half_width + 1))
    zs = np.arange(max(0, z0 - half_width), min(vol_shape_3d[2], z0 + half_width + 1))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    lin = np.ravel_multi_index((gx.ravel(), gy.ravel(), gz.ravel()), vol_shape_3d)
    return lin.astype(np.int64)


def _grid_thin_indices(
    idx: np.ndarray,
    vol_shape_3d: Tuple[int, int, int],
    rs: np.random.RandomState,
    min_step_vox: int,
    max_keep: Optional[int] = None,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).ravel()
    if idx.size == 0 or min_step_vox <= 1:
        return idx
    coords = np.column_stack(np.unravel_index(idx, vol_shape_3d)).astype(np.int64)
    cells = (coords // int(min_step_vox)).astype(np.int64)
    key = cells[:, 0] * (10**8) + cells[:, 1] * (10**4) + cells[:, 2]
    perm = rs.permutation(idx.size)
    key_p, idx_p = key[perm], idx[perm]
    seen: set = set()
    keep = []
    for k, ii in zip(key_p, idx_p):
        kk = int(k)
        if kk in seen:
            continue
        seen.add(kk)
        keep.append(int(ii))
        if max_keep is not None and len(keep) >= int(max_keep):
            break
    return np.asarray(keep, dtype=np.int64)


def _build_holdout_mask(
    tumor_train_flat: np.ndarray,
    healthy_train_flat: np.ndarray,
    vol_shape_3d: Tuple[int, int, int],
    rs: np.random.RandomState,
    holdout_frac: float,
    patch_half_width: int,
    max_centers_per_class: int = 5000,
) -> np.ndarray:
    pos_centers = _centers_from_mask(tumor_train_flat, vol_shape_3d)
    neg_centers = _centers_from_mask(healthy_train_flat, vol_shape_3d)
    if pos_centers.shape[0] == 0 or neg_centers.shape[0] == 0:
        raise RuntimeError("patch_holdout cannot run: empty class mask.")
    n_pos_cent = max(1, int(round(pos_centers.shape[0] * holdout_frac)))
    n_pos_cent = min(n_pos_cent, max_centers_per_class, pos_centers.shape[0])
    n_neg_cent = min(n_pos_cent, max_centers_per_class, neg_centers.shape[0])
    pos_c = pos_centers[rs.choice(pos_centers.shape[0], size=n_pos_cent, replace=False)]
    neg_c = neg_centers[rs.choice(neg_centers.shape[0], size=n_neg_cent, replace=False)]
    holdout_mask = np.zeros(int(np.prod(vol_shape_3d)), dtype=bool)
    for c in pos_c:
        holdout_mask[_cube_indices(c, vol_shape_3d, patch_half_width)] = True
    for c in neg_c:
        holdout_mask[_cube_indices(c, vol_shape_3d, patch_half_width)] = True
    return holdout_mask


def _voxel_size_mm_to_steps(
    voxel_size_mm: Tuple[float, float, float],
    block_mm: float,
    buffer_mm: float,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    vx, vy, vz = float(voxel_size_mm[0]), float(voxel_size_mm[1]), float(voxel_size_mm[2])
    if vx <= 0 or vy <= 0 or vz <= 0:
        raise ValueError(f"Invalid voxel_size_mm={voxel_size_mm}")
    sx = max(1, int(round(float(block_mm) / vx)))
    sy = max(1, int(round(float(block_mm) / vy)))
    sz = max(1, int(round(float(block_mm) / vz)))
    bx = min(max(0, int(round(float(buffer_mm) / vx))), max(0, sx // 2 - 1))
    by = min(max(0, int(round(float(buffer_mm) / vy))), max(0, sy // 2 - 1))
    bz = min(max(0, int(round(float(buffer_mm) / vz))), max(0, sz // 2 - 1))
    return (sx, sy, sz), (bx, by, bz)


def _cell_key(coords_xyz: np.ndarray, step_xyz: Tuple[int, int, int]) -> np.ndarray:
    sx, sy, sz = int(step_xyz[0]), int(step_xyz[1]), int(step_xyz[2])
    cells = (coords_xyz // np.array([sx, sy, sz], dtype=np.int64)).astype(np.int64)
    key = cells[:, 0] * 10**8 + cells[:, 1] * 10**4 + cells[:, 2]
    return key.astype(np.int64)


def _is_in_core(
    coords_xyz: np.ndarray,
    step_xyz: Tuple[int, int, int],
    buffer_xyz: Tuple[int, int, int],
) -> np.ndarray:
    sx, sy, sz = int(step_xyz[0]), int(step_xyz[1]), int(step_xyz[2])
    bx, by, bz = int(buffer_xyz[0]), int(buffer_xyz[1]), int(buffer_xyz[2])
    rx = coords_xyz[:, 0] % sx
    ry = coords_xyz[:, 1] % sy
    rz = coords_xyz[:, 2] % sz
    return (rx >= bx) & (rx < (sx - bx)) & (ry >= by) & (ry < (sy - by)) & (rz >= bz) & (rz < (sz - bz))


# ---------------------------------------------------------------------------
# Standalone function (mirrors original deepneurobold API)
# ---------------------------------------------------------------------------

def make_split(
    tumor_train_flat: np.ndarray,
    healthy_train_flat: np.ndarray,
    vol_shape_3d: Tuple[int, int, int],
    mode: SamplingMode,
    seed: int,
    n_per_class: Optional[int] = 20000,
    holdout_frac: float = 0.10,
    patch_half_width: int = 2,
    voxel_size_mm: Optional[Tuple[float, float, float]] = None,
    cube_size_cm: Optional[float] = None,
    max_retries: int = 6,
    min_step_vox: Optional[int] = None,
    grid_block_mm: Optional[float] = None,
    grid_buffer_mm: Optional[float] = None,
    bg_gm_flat: Optional[np.ndarray] = None,
    bg_wm_flat: Optional[np.ndarray] = None,
    bg_gm_fraction: float = 1.0 / 3.0,
) -> SampleSplit:
    """
    Create a reproducible train/validation split.

    Parameters
    ----------
    tumor_train_flat, healthy_train_flat : np.ndarray, bool
        Flattened masks for positive and negative voxels.
    vol_shape_3d : tuple(int, int, int)
    mode : SamplingMode
    seed : int
    n_per_class : int or None
    holdout_frac : float
    patch_half_width : int
    voxel_size_mm : tuple, optional
    cube_size_cm : float, optional
    max_retries : int
    min_step_vox : int, optional
    grid_block_mm, grid_buffer_mm : float, optional
    bg_gm_flat, bg_wm_flat : np.ndarray, optional
        Flat boolean masks of GM and WM tissue (from SynthSeg). When BOTH
        are provided, the negative class is drawn with ``bg_gm_fraction``
        from GM and ``1 - bg_gm_fraction`` from WM.
        CSF should already be excluded from ``healthy_train_flat`` upstream
        (see ``build_train_masks(exclude_csf_from_background=True)``).
    bg_gm_fraction : float
        Fraction of negative samples drawn from GM. Default 1/3.

    Returns
    -------
    SampleSplit
    """
    stratified_bg = bg_gm_flat is not None and bg_wm_flat is not None
    rs = _rng(seed)
    print("[sampling] make_split starting...")
    print(f"  Tumor voxels available:   {int(np.sum(tumor_train_flat))}")
    print(f"  Healthy voxels available: {int(np.sum(healthy_train_flat))}")
    print(f"  n_per_class: {n_per_class}  holdout_frac: {holdout_frac}")
    print(f"  mode: {mode}  seed: {seed}")

    effective_patch_half_width = int(patch_half_width)
    meta: Dict[str, Any] = {}

    if cube_size_cm is not None and voxel_size_mm is not None:
        cube_mm = float(cube_size_cm) * 10.0
        voxel_mm = float(np.mean(voxel_size_mm))
        cube_vox = max(1, int(round(cube_mm / voxel_mm)))
        effective_patch_half_width = max(1, cube_vox // 2)
        meta.update({
            "cube_size_cm": float(cube_size_cm),
            "voxel_size_mm": tuple(float(x) for x in voxel_size_mm),
            "patch_half_width_vox": int(effective_patch_half_width),
        })

    pos_pool = np.flatnonzero(tumor_train_flat)
    neg_pool = np.flatnonzero(healthy_train_flat)

    if pos_pool.size == 0 or neg_pool.size == 0:
        raise RuntimeError(f"Cannot sample: empty class pool. pos={pos_pool.size}, neg={neg_pool.size}")

    if min_step_vox is not None:
        ms = int(min_step_vox)
        if ms > 1:
            pos_pool = _grid_thin_indices(pos_pool, vol_shape_3d, rs, min_step_vox=ms)
            neg_pool = _grid_thin_indices(neg_pool, vol_shape_3d, rs, min_step_vox=ms)
            meta["min_step_vox"] = ms
            if pos_pool.size == 0 or neg_pool.size == 0:
                raise RuntimeError("Cannot sample after thinning: empty class pool.")

    # ---- voxel_random ----
    if mode == "voxel_random":
        pos_s, neg_s = _sample_equal(pos_pool, neg_pool, n_per_class, rs)
        if stratified_bg:
            neg_s = _sample_neg_stratified(
                neg_pool, bg_gm_flat, bg_wm_flat,
                n_neg=neg_s.size, rs=rs, gm_fraction=bg_gm_fraction,
            )
            meta["bg_stratified"] = {"gm_fraction": float(bg_gm_fraction)}
        train_idx = np.concatenate([pos_s, neg_s]).astype(np.int64)
        train_y = np.concatenate([np.ones(pos_s.size, dtype=np.int64), np.zeros(neg_s.size, dtype=np.int64)])
        perm = rs.permutation(train_idx.size)
        return SampleSplit(
            train_idx=train_idx[perm], val_idx=np.zeros((0,), dtype=np.int64),
            train_y=train_y[perm], val_y=np.zeros((0,), dtype=np.int64), meta=meta,
        )

    # ---- grid_block ----
    if mode == "grid_block":
        if voxel_size_mm is None:
            raise ValueError("grid_block requires voxel_size_mm.")
        if grid_block_mm is None:
            raise ValueError("grid_block requires grid_block_mm.")
        block_mm = float(grid_block_mm)
        buffer_mm = float(grid_buffer_mm) if grid_buffer_mm is not None else (0.25 * block_mm)
        step_xyz, buffer_xyz = _voxel_size_mm_to_steps(voxel_size_mm, block_mm=block_mm, buffer_mm=buffer_mm)
        meta.update({
            "grid_block_mm": block_mm, "grid_buffer_mm": buffer_mm,
            "grid_step_xyz": tuple(int(x) for x in step_xyz),
            "grid_buffer_xyz": tuple(int(x) for x in buffer_xyz),
            "voxel_size_mm": tuple(float(x) for x in voxel_size_mm),
        })
        pos_coords = _coords_from_flat(pos_pool, vol_shape_3d)
        neg_coords = _coords_from_flat(neg_pool, vol_shape_3d)
        pos_key = _cell_key(pos_coords, step_xyz)
        neg_key = _cell_key(neg_coords, step_xyz)
        common_cells = np.intersect1d(np.unique(pos_key), np.unique(neg_key))
        if common_cells.size < 2:
            raise RuntimeError(f"grid_block: not enough common cells. common_cells={common_cells.size}")
        if not (0.0 < float(holdout_frac) < 1.0):
            raise ValueError("holdout_frac must be in (0, 1).")
        n_val_cells = min(max(1, int(round(common_cells.size * float(holdout_frac)))), common_cells.size - 1)
        val_cells = rs.choice(common_cells, size=n_val_cells, replace=False)
        pos_in_val = np.isin(pos_key, val_cells)
        neg_in_val = np.isin(neg_key, val_cells)
        val_pos_pool = pos_pool[pos_in_val & _is_in_core(pos_coords, step_xyz, buffer_xyz)]
        val_neg_pool = neg_pool[neg_in_val & _is_in_core(neg_coords, step_xyz, buffer_xyz)]
        train_pos_pool = pos_pool[~pos_in_val]
        train_neg_pool = neg_pool[~neg_in_val]
        if val_pos_pool.size == 0 or val_neg_pool.size == 0:
            raise RuntimeError("grid_block produced empty validation pool.")
        if train_pos_pool.size == 0 or train_neg_pool.size == 0:
            raise RuntimeError("grid_block produced empty training pool.")
        actual_n = n_per_class
        if n_per_class is not None:
            actual_n = min(int(n_per_class), train_pos_pool.size, train_neg_pool.size)
        pos_s, neg_s = _sample_equal(train_pos_pool, train_neg_pool, actual_n, rs)
        if stratified_bg:
            neg_s = _sample_neg_stratified(
                train_neg_pool, bg_gm_flat, bg_wm_flat,
                n_neg=neg_s.size, rs=rs, gm_fraction=bg_gm_fraction,
            )
            meta["bg_stratified"] = {"gm_fraction": float(bg_gm_fraction)}
        train_idx = np.concatenate([pos_s, neg_s]).astype(np.int64)
        train_y = np.concatenate([np.ones(pos_s.size, dtype=np.int64), np.zeros(neg_s.size, dtype=np.int64)])
        n_val = min(val_pos_pool.size, val_neg_pool.size)
        val_idx = np.concatenate([
            rs.choice(val_pos_pool, size=n_val, replace=False),
            rs.choice(val_neg_pool, size=n_val, replace=False),
        ]).astype(np.int64)
        val_y = np.concatenate([np.ones(n_val, dtype=np.int64), np.zeros(n_val, dtype=np.int64)])
        perm_tr = rs.permutation(train_idx.size)
        perm_va = rs.permutation(val_idx.size)
        meta.update({
            "val_cells_count": int(n_val_cells), "common_cells_count": int(common_cells.size),
        })
        return SampleSplit(
            train_idx=train_idx[perm_tr], val_idx=val_idx[perm_va],
            train_y=train_y[perm_tr], val_y=val_y[perm_va], meta=meta,
        )

    # ---- patch_holdout ----
    if mode != "patch_holdout":
        raise ValueError(f"Unknown sampling mode: {mode}")

    if not (0.0 < float(holdout_frac) < 1.0):
        raise ValueError("holdout_frac must be in (0, 1).")

    last_stats: Dict[str, Any] = {}
    for t in range(int(max_retries)):
        frac_t = min(0.50, float(holdout_frac) * (1.0 + 0.35 * t))
        hw_t = int(effective_patch_half_width + t)
        holdout_mask = _build_holdout_mask(
            tumor_train_flat=tumor_train_flat, healthy_train_flat=healthy_train_flat,
            vol_shape_3d=vol_shape_3d, rs=rs, holdout_frac=frac_t, patch_half_width=hw_t,
        )
        val_pos = np.flatnonzero(holdout_mask & tumor_train_flat)
        val_neg = np.flatnonzero(holdout_mask & healthy_train_flat)
        pos_pool_train = np.flatnonzero((~holdout_mask) & tumor_train_flat)
        neg_pool_train = np.flatnonzero((~holdout_mask) & healthy_train_flat)
        if min_step_vox is not None and int(min_step_vox) > 1:
            ms = int(min_step_vox)
            val_pos = _grid_thin_indices(val_pos, vol_shape_3d, rs, min_step_vox=ms)
            val_neg = _grid_thin_indices(val_neg, vol_shape_3d, rs, min_step_vox=ms)
            pos_pool_train = _grid_thin_indices(pos_pool_train, vol_shape_3d, rs, min_step_vox=ms)
            neg_pool_train = _grid_thin_indices(neg_pool_train, vol_shape_3d, rs, min_step_vox=ms)
        last_stats = {
            "retry": int(t), "holdout_frac_effective": float(frac_t),
            "patch_half_width_effective": int(hw_t),
            "val_pos_pool": int(val_pos.size), "val_neg_pool": int(val_neg.size),
            "train_pos_pool": int(pos_pool_train.size), "train_neg_pool": int(neg_pool_train.size),
        }
        if val_pos.size == 0 or val_neg.size == 0:
            continue
        if pos_pool_train.size == 0 or neg_pool_train.size == 0:
            continue
        actual_n = n_per_class
        if n_per_class is not None:
            actual_n = min(int(n_per_class), pos_pool_train.size, neg_pool_train.size)
        pos_s, neg_s = _sample_equal(pos_pool_train, neg_pool_train, actual_n, rs)
        if stratified_bg:
            neg_s = _sample_neg_stratified(
                neg_pool_train, bg_gm_flat, bg_wm_flat,
                n_neg=neg_s.size, rs=rs, gm_fraction=bg_gm_fraction,
            )
            last_stats["bg_stratified"] = {"gm_fraction": float(bg_gm_fraction)}
        train_idx = np.concatenate([pos_s, neg_s]).astype(np.int64)
        train_y = np.concatenate([np.ones(pos_s.size, dtype=np.int64), np.zeros(neg_s.size, dtype=np.int64)])
        n_val = min(val_pos.size, val_neg.size)
        val_idx = np.concatenate([
            rs.choice(val_pos, size=n_val, replace=False),
            rs.choice(val_neg, size=n_val, replace=False),
        ]).astype(np.int64)
        val_y = np.concatenate([np.ones(n_val, dtype=np.int64), np.zeros(n_val, dtype=np.int64)])
        perm_tr = rs.permutation(train_idx.size)
        perm_va = rs.permutation(val_idx.size)
        return SampleSplit(
            train_idx=train_idx[perm_tr], val_idx=val_idx[perm_va],
            train_y=train_y[perm_tr], val_y=val_y[perm_va],
            meta={**meta, **last_stats},
        )

    raise RuntimeError(
        f"patch_holdout failed after {max_retries} retries. Last stats: {last_stats}"
    )


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class TrainValSampler(BaseComponent):
    """
    Create reproducible train/validation splits from voxel index pools.

    Parameters
    ----------
    mode : str
        ``"grid_block"``, ``"patch_holdout"``, or ``"voxel_random"``.
    n_per_class : int
    seed : int
    config : dict, optional
    """

    VALID_MODES = ("grid_block", "patch_holdout", "voxel_random")

    def __init__(
        self,
        mode: str = "grid_block",
        n_per_class: int = 20_000,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="TrainValSampler", config=config)
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        self.mode = mode
        self.n_per_class = n_per_class
        self.seed = seed

    def validate(self) -> bool:
        return self.mode in self.VALID_MODES

    def split(
        self,
        tumor_flat: np.ndarray,
        healthy_flat: np.ndarray,
        vol_shape_3d: Tuple[int, int, int],
        **kwargs: Any,
    ) -> SampleSplit:
        """
        Generate a train/validation split.

        Parameters
        ----------
        tumor_flat, healthy_flat : np.ndarray, bool
        vol_shape_3d : tuple(int, int, int)
        **kwargs : forwarded to :func:`make_split`

        Returns
        -------
        SampleSplit
        """
        return make_split(
            tumor_train_flat=tumor_flat,
            healthy_train_flat=healthy_flat,
            vol_shape_3d=vol_shape_3d,
            mode=self.mode,  # type: ignore[arg-type]
            seed=self.seed,
            n_per_class=self.n_per_class,
            **kwargs,
        )
