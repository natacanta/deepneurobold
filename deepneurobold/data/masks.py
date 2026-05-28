"""
data.masks
==========
Segmentation mask loader and training mask builder.

Provides:
- ``load_segmentation_masks`` — load raw enh/necro/edema masks from disk
- ``build_train_masks``       — derive training-ready pos/neg masks with
                                safety margins and the 3 cm in-between test region
- ``SegmentationMasks``       — class interface wrapping both functions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from .base import BaseDataLoader

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PositiveDef = Literal["enhancing_only", "enhancing_plus_necrosis", "necrosis", "enhancing"]
NegativeDef = Literal["healthy_strict", "healthy_controlled"]


# ---------------------------------------------------------------------------
# SynthSeg FreeSurfer label scheme -> GM / WM / CSF groupings
# ---------------------------------------------------------------------------
# SynthSeg v1 follows the standard FreeSurfer "aseg" label numbering.  We map
# each integer label to one of three tissue classes used for stratified
# background sampling (train negatives drawn from 1/3 GM + 2/3 WM,
# CSF excluded).
#
# Label families:
#   - Cortex + deep gray-matter nuclei -> GM
#   - Cerebral / cerebellar white matter + brain stem -> WM
#   - Ventricles + CSF compartments -> CSF (excluded from training)
#
# Any label not listed below is ignored (counts as "other", neither GM/WM/CSF).
SYNTHSEG_GM_LABELS = frozenset({
    3,  42,            # Left/Right Cerebral Cortex
    8,  47,            # Left/Right Cerebellum Cortex
    10, 49,            # Left/Right Thalamus
    11, 50,            # Left/Right Caudate
    12, 51,            # Left/Right Putamen
    13, 52,            # Left/Right Pallidum
    17, 53,            # Left/Right Hippocampus
    18, 54,            # Left/Right Amygdala
    26, 58,            # Left/Right Accumbens
    28, 60,            # Left/Right VentralDC
})

SYNTHSEG_WM_LABELS = frozenset({
    2,  41,            # Left/Right Cerebral White Matter
    7,  46,            # Left/Right Cerebellum White Matter
    16,                # Brain stem
})

SYNTHSEG_CSF_LABELS = frozenset({
    4,  43,            # Left/Right Lateral Ventricle
    5,  44,            # Left/Right Inferior Lateral Ventricle
    14, 15,            # 3rd / 4th Ventricle
    24,                # CSF
})


def synthseg_to_tissue_masks(
    seg_3d: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Convert a SynthSeg FreeSurfer-style integer label volume into 3 binary
    tissue masks: GM, WM, CSF.

    Parameters
    ----------
    seg_3d : np.ndarray
        3D integer array with FreeSurfer-style label IDs (typically the contents
        of ``results/02_synthseg/Patient_XX/segmentation.nii.gz``).

    Returns
    -------
    dict with keys ``'gm'``, ``'wm'``, ``'csf'`` (each a 3D bool ndarray with
    the same shape as ``seg_3d``).
    """
    seg = np.asarray(seg_3d)
    gm  = np.isin(seg, list(SYNTHSEG_GM_LABELS))
    wm  = np.isin(seg, list(SYNTHSEG_WM_LABELS))
    csf = np.isin(seg, list(SYNTHSEG_CSF_LABELS))
    return {"gm": gm, "wm": wm, "csf": csf}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mm_to_iters(mm: float, voxel_size_mm: Tuple[float, float, float]) -> int:
    """Convert a physical distance in mm to binary_dilation iterations.

    Uses the *mean* voxel size with round() — identical to original deepneurobold
    masks.py _mm_to_iters:
        vmean = float(np.mean([vx, vy, vz]))
        iters = int(round(float(mm) / vmean))
        return max(1, iters)
    """
    vx, vy, vz = float(voxel_size_mm[0]), float(voxel_size_mm[1]), float(voxel_size_mm[2])
    if vx <= 0 or vy <= 0 or vz <= 0:
        raise ValueError(f"Invalid voxel_size_mm={voxel_size_mm}")
    vmean = float(np.mean([vx, vy, vz]))
    return max(1, int(round(float(mm) / vmean)))


def load_synthseg_seg(
    patient_dir: Path,
    ref_img: Optional[nib.Nifti1Image] = None,
    project_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Load the SynthSeg parcellation for a patient and align it to ``ref_img``.

    Looks under ``<project_root>/results/02_synthseg/<patient_id>/segmentation.nii.gz``.
    If ``ref_img`` is provided and the SynthSeg grid does not match it, the seg
    is resampled (nearest-neighbour) onto the reference grid.

    Parameters
    ----------
    patient_dir : Path
        Path to the patient's data dir (used to derive the patient ID and the
        project root if ``project_root`` is omitted).
    ref_img : nib.Nifti1Image, optional
        Reference image whose grid the seg should match (typically the T1 used
        by the rest of the pipeline).
    project_root : Path, optional
        Project root. Defaults to ``patient_dir.parent.parent`` (i.e. assumes
        the canonical ``<root>/data/Patient_XX/`` layout).

    Returns
    -------
    np.ndarray of int32, same 3D shape as ``ref_img``, or ``None`` if the
    SynthSeg seg file is missing.
    """
    patient_dir = Path(patient_dir)
    pid = patient_dir.name
    if project_root is None:
        project_root = patient_dir.parent.parent
    seg_path = Path(project_root) / "results" / "02_synthseg" / pid / "segmentation.nii.gz"
    if not seg_path.exists():
        return None
    seg_img = nib.load(str(seg_path))
    # If a 4D BOLD image is passed as ref, project it to its spatial 3D slab
    # (resample_from_to only accepts matching ndim and a 4x4 affine).
    if ref_img is not None and ref_img.ndim > 3:
        ref_3d_arr = np.asarray(ref_img.get_fdata())[..., 0]
        ref_img = nib.Nifti1Image(ref_3d_arr, ref_img.affine[:4, :4], ref_img.header)
    if ref_img is not None and (
        seg_img.shape[:3] != ref_img.shape[:3]
        or not np.allclose(seg_img.affine, ref_img.affine, atol=1e-3)
    ):
        from nibabel.processing import resample_from_to
        seg_img = resample_from_to(seg_img, ref_img, order=0)
    return np.asarray(seg_img.get_fdata()).astype(np.int32)


def _normalize_mask_keys(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Ensure masks dict always has keys 'enh', 'necro', 'edema'."""
    if "necro" not in masks:
        if "nec" in masks:
            masks["necro"] = masks["nec"]
        elif "necrosis" in masks:
            masks["necro"] = masks["necrosis"]

    if "edema" not in masks:
        if "ed" in masks:
            masks["edema"] = masks["ed"]
        elif "edema_mask" in masks:
            masks["edema"] = masks["edema_mask"]

    if "enh" not in masks:
        if "enhancing" in masks:
            masks["enh"] = masks["enhancing"]
        elif "et" in masks:
            masks["enh"] = masks["et"]

    return masks


# ---------------------------------------------------------------------------
# Standalone functions (mirrors original deepneurobold API)
# ---------------------------------------------------------------------------

def load_segmentation_masks(
    patient_dir: Path,
    use_brats: bool,
) -> Tuple[Dict[str, np.ndarray], nib.Nifti1Image]:
    """
    Load tumor segmentation masks.

    Returns
    -------
    masks : dict with keys ``'enh'``, ``'necro'``, ``'edema'``
    ref_img : nib.Nifti1Image aligned with the masks
    """
    from deepneurobold.analysis.clustering.labels import build_masks_from_seg

    masks, ref_img = build_masks_from_seg(Path(patient_dir), use_brats=use_brats)
    if not isinstance(ref_img, nib.Nifti1Image):
        raise TypeError(
            f"build_masks_from_seg returned ref_img of type {type(ref_img).__name__}, "
            f"expected nib.Nifti1Image"
        )
    masks = _normalize_mask_keys(masks)
    return masks, ref_img


def build_train_masks(
    brain_flat: np.ndarray,
    masks: Dict[str, np.ndarray],
    positive: Union[PositiveDef, str],
    negative: NegativeDef,
    margin_iters: int = 2,
    controlled_outer_iters: int = 12,
    inbetween_outer_mm: float = 30.0,
    voxel_size_mm: Optional[Tuple[float, float, float]] = None,
    inbetween_outer_iters: Optional[int] = None,
    inbetween_include_edema: bool = True,
    synthseg_seg: Optional[np.ndarray] = None,
    exclude_csf_from_background: bool = True,
) -> Dict[str, Any]:
    """
    Build flat boolean masks for supervised training.

    The *test region* (perilesional zone) comprises the edema plus a ~3 cm
    band around the tumor core, excluding the core itself. Hotspot detection
    thresholds are computed within this region rather than from the full brain.

    Parameters
    ----------
    synthseg_seg : np.ndarray, optional
        3D integer label volume from SynthSeg (FreeSurfer-style aseg labels).
        When provided, ``gm_flat`` / ``wm_flat`` / ``csf_flat`` masks are added
        to the output dict so downstream samplers can perform stratified
        background draws (e.g. 1/3 GM + 2/3 WM).
    exclude_csf_from_background : bool
        If True and ``synthseg_seg`` is provided, CSF voxels are removed from
        ``healthy_train_flat`` (negatives never drawn from ventricles).

    Returns
    -------
    dict with keys:
        ``tumor_core_flat``, ``tumor_margin_flat``, ``tumor_ring_flat``,
        ``healthy_candidate_flat``, ``healthy_train_flat``,
        ``tumor_train_flat``, ``train_mask_flat``,
        ``edema_flat``, ``test_region_flat``, ``band3cm_flat``,
        ``controlled_outer_iters``, ``inbetween_outer_iters``,
        ``inbetween_outer_mm``, ``inbetween_include_edema``.
        When ``synthseg_seg`` is provided, also ``gm_flat``, ``wm_flat``,
        ``csf_flat``, ``healthy_train_gm_flat``, ``healthy_train_wm_flat``.
    """
    masks = _normalize_mask_keys(masks)

    if "enh" not in masks or "necro" not in masks or "edema" not in masks:
        raise KeyError(
            f"Masks dict must contain keys: 'enh', 'necro', 'edema'. "
            f"Got: {sorted(masks.keys())}"
        )

    enh_3d   = np.asarray(masks["enh"]).astype(bool)
    necro_3d = np.asarray(masks["necro"]).astype(bool)
    edema_3d = np.asarray(masks["edema"]).astype(bool)

    vol_shape = enh_3d.shape
    if brain_flat.size != int(np.prod(vol_shape)):
        raise RuntimeError(
            f"Shape mismatch: brain_flat has {brain_flat.size} voxels "
            f"but masks have shape {vol_shape} ({int(np.prod(vol_shape))} voxels)."
        )

    brain_3d = np.asarray(brain_flat).reshape(vol_shape).astype(bool)

    # Restrict to brain
    enh_3d   = enh_3d   & brain_3d
    necro_3d = necro_3d & brain_3d
    edema_3d = edema_3d & brain_3d

    all_tumor_3d = enh_3d | necro_3d

    # Tumor core depends on positive definition
    pos_lower = positive.lower().strip()
    if pos_lower in ("necrosis", "necro"):
        tumor_core_3d = necro_3d
    elif pos_lower in ("enhancing", "enhancing_only"):
        tumor_core_3d = enh_3d
    elif pos_lower == "enhancing_plus_necrosis":
        tumor_core_3d = enh_3d | necro_3d
    else:
        raise ValueError(f"Unknown positive definition: {positive}")

    # Core wins over edema
    edema_3d = edema_3d & (~tumor_core_3d)

    # Margin
    if margin_iters < 0:
        raise ValueError("margin_iters must be >= 0")
    if margin_iters == 0:
        tumor_margin_3d = tumor_core_3d.copy()
    else:
        tumor_margin_3d = binary_dilation(tumor_core_3d, iterations=int(margin_iters)).astype(bool)
    tumor_margin_3d = tumor_margin_3d & brain_3d

    tumor_ring_3d = tumor_margin_3d & (~tumor_core_3d)

    # Healthy pools
    healthy_candidate_3d = brain_3d & (~all_tumor_3d) & (~edema_3d)
    healthy_base_3d      = healthy_candidate_3d & (~tumor_margin_3d)

    if negative == "healthy_strict":
        healthy_train_3d = healthy_base_3d

    elif negative == "healthy_controlled":
        if controlled_outer_iters <= margin_iters:
            raise ValueError("controlled_outer_iters must be > margin_iters for a non-empty controlled band.")
        outer_3d = binary_dilation(
            tumor_core_3d, iterations=int(controlled_outer_iters)
        ).astype(bool) & brain_3d
        band_3d = outer_3d & (~tumor_margin_3d)
        healthy_train_3d = healthy_base_3d & band_3d
        if np.count_nonzero(healthy_train_3d) == 0:
            healthy_train_3d = healthy_base_3d  # fallback
    else:
        raise ValueError(f"Unknown negative definition: {negative}")

    # ------------------------------------------------------------------
    # Optional SynthSeg tissue stratification:
    # CSF is excluded from the healthy pool, GM/WM masks are exposed for
    # stratified sampling downstream.
    # ------------------------------------------------------------------
    gm_3d = wm_3d = csf_3d = None
    healthy_train_gm_3d = healthy_train_wm_3d = None
    if synthseg_seg is not None:
        synthseg_seg = np.asarray(synthseg_seg)
        if synthseg_seg.shape != vol_shape:
            raise RuntimeError(
                f"synthseg_seg shape {synthseg_seg.shape} does not match "
                f"masks volume shape {vol_shape}"
            )
        tissues = synthseg_to_tissue_masks(synthseg_seg)
        gm_3d  = tissues["gm"]  & brain_3d
        wm_3d  = tissues["wm"]  & brain_3d
        csf_3d = tissues["csf"] & brain_3d

        if bool(exclude_csf_from_background):
            healthy_train_3d = healthy_train_3d & (~csf_3d)

        healthy_train_gm_3d = healthy_train_3d & gm_3d
        healthy_train_wm_3d = healthy_train_3d & wm_3d

    tumor_train_3d = tumor_core_3d
    train_mask_3d  = healthy_train_3d | tumor_train_3d

    # In-between test region (~3 cm band around core + edema, excluding core)
    if inbetween_outer_iters is None:
        if voxel_size_mm is None:
            raise ValueError(
                "To build the 3 cm in-between zone you must provide either "
                "inbetween_outer_iters or voxel_size_mm."
            )
        inbetween_outer_iters = _mm_to_iters(float(inbetween_outer_mm), voxel_size_mm)

    if int(inbetween_outer_iters) < 0:
        raise ValueError("inbetween_outer_iters must be >= 0")

    if int(inbetween_outer_iters) == 0:
        outer3_3d = tumor_core_3d.copy()
    else:
        outer3_3d = binary_dilation(
            tumor_core_3d, iterations=int(inbetween_outer_iters)
        ).astype(bool)
    outer3_3d = outer3_3d & brain_3d

    band3_3d = outer3_3d & (~tumor_core_3d)

    if bool(inbetween_include_edema):
        test_region_3d = (edema_3d | band3_3d) & brain_3d
    else:
        test_region_3d = band3_3d & brain_3d

    out: Dict[str, Any] = {
        "tumor_core_flat":       tumor_core_3d.ravel(),
        "tumor_margin_flat":     tumor_margin_3d.ravel(),
        "tumor_ring_flat":       tumor_ring_3d.ravel(),
        "healthy_candidate_flat":healthy_candidate_3d.ravel(),
        "healthy_train_flat":    healthy_train_3d.ravel(),
        "tumor_train_flat":      tumor_train_3d.ravel(),
        "train_mask_flat":       train_mask_3d.ravel(),
        "edema_flat":            edema_3d.ravel(),
        "controlled_outer_iters":int(controlled_outer_iters),
        "test_region_flat":      test_region_3d.ravel(),
        "band3cm_flat":          band3_3d.ravel(),
        "inbetween_outer_iters": int(inbetween_outer_iters),
        "inbetween_outer_mm":    float(inbetween_outer_mm),
        "inbetween_include_edema": bool(inbetween_include_edema),
    }
    if synthseg_seg is not None:
        out.update({
            "gm_flat":               gm_3d.ravel(),
            "wm_flat":               wm_3d.ravel(),
            "csf_flat":              csf_3d.ravel(),
            "healthy_train_gm_flat": healthy_train_gm_3d.ravel(),
            "healthy_train_wm_flat": healthy_train_wm_3d.ravel(),
        })
    return out


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------

class SegmentationMasks(BaseDataLoader):
    """
    Load tumor segmentation masks (necrosis, enhancing, edema) and derive
    training-ready positive/negative masks.

    Parameters
    ----------
    patient_dir : Path
    use_brats : bool
    config : dict, optional
    """

    def __init__(
        self,
        patient_dir: Path,
        use_brats: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(patient_dir=patient_dir, config=config)
        self.use_brats = use_brats
        self._masks: Optional[Dict[str, np.ndarray]] = None
        self._ref_img: Optional[nib.Nifti1Image] = None

    def load(self) -> Dict[str, Any]:
        """
        Load raw segmentation masks from disk.

        Returns
        -------
        dict with keys ``'masks'`` (enh/necro/edema) and ``'ref_img'``.
        """
        self._masks, self._ref_img = load_segmentation_masks(
            self.patient_dir, self.use_brats
        )
        return {"masks": self._masks, "ref_img": self._ref_img}

    def build_train_masks(
        self,
        brain_flat: np.ndarray,
        positive: str,
        negative: str,
        margin_iters: int = 2,
        controlled_outer_iters: int = 12,
        voxel_size_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        inbetween_outer_mm: float = 30.0,
        inbetween_outer_iters: Optional[int] = None,
        inbetween_include_edema: bool = True,
    ) -> Dict[str, Any]:
        """Build training-ready positive and negative masks with safety margins."""
        if self._masks is None:
            self.load()
        return build_train_masks(
            brain_flat=brain_flat,
            masks=self._masks,  # type: ignore[arg-type]
            positive=positive,
            negative=negative,
            margin_iters=margin_iters,
            controlled_outer_iters=controlled_outer_iters,
            inbetween_outer_mm=inbetween_outer_mm,
            voxel_size_mm=voxel_size_mm,
            inbetween_outer_iters=inbetween_outer_iters,
            inbetween_include_edema=inbetween_include_edema,
        )
