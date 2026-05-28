# DeepNeuroBOLD

**Preoperative BOLD-fMRI mapping of glioma infiltration with voxel-wise machine learning.**

A clinical research pipeline that predicts glioma infiltration patterns from
hypoxia-targeted BOLD-fMRI in patients with newly diagnosed glioblastoma. The
pipeline trains a voxel-wise classifier to distinguish tumour tissue from
healthy parenchyma and projects it onto the peritumoural edema, producing
**infiltration probability maps**, **ranked hotspot clusters**, and **hotspot
phenotypes** (compact, multifocal, diffuse).

---

## Motivation

Standard contrast-enhanced MRI shows the enhancing tumour and the surrounding
T2/FLAIR hyperintensity (edema), but it does not show the microscopic
infiltration in the peritumoural brain that drives recurrence. Resting-state
BOLD-fMRI under a hypoxia challenge carries vascular information altered by
infiltrating tumour. This pipeline learns the contrast from BOLD time series
in tumour vs healthy voxels and projects the learned probability onto the
peritumoural edema of every new patient.

---

## Pipeline overview

```
                     PREPROCESSING                       (external step)
                DICOM to NIfTI, BET, ANTs T1
                registration, BOLD to T1, tissue
                segmentation (SynthSeg optional)
                             │
                             ▼
              TRAINING MASK CONSTRUCTION
              tumour core, edema, 3 cm test region,
              GM/WM background (CSF excluded)
                             │
                             ▼
                 VOXEL-WISE CLASSIFIER
              Spatial grid-block CV (K=4, 20 mm blocks,
              30 mm buffer). Random Forest by default;
              Linear SVM and L2 Logistic Regression as
              baselines (DeLong-tested).
                             │
                             ▼
                PROBABILITY MAP (whole brain)
                             │
                             ▼
                 HOTSPOT DETECTION
              Peritumoural edema, adaptive p95 (or p90
              fallback), 150 mm^3 minimum, top-3 clusters.
                             │
                             ▼
                  HOTSPOT PHENOTYPING
              Compact / multifocal / diffuse based on
              sphericity + BOLD time-series coherence.
```

---

## Repository layout

```
deepneurobold/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example                          site-specific paths template
│
├── deepneurobold/                        Python package
│   ├── config/paths.py                   env-var-based path resolution
│   ├── data/                             loaders, masks, sampling
│   ├── preprocessing/                    BOLD to T1, tissue segmentation
│   ├── features/                         time-series features
│   ├── classifiers/                      RF / SVM / LR wrappers
│   ├── analysis/
│   │   ├── consensus_tumor.py            multi-source consensus core
│   │   ├── bold_quality.py               tSNR, DVARS, drift, spikes
│   │   ├── bold_quality_classifier.py    per-voxel quality RF (LOO)
│   │   └── hotspot/
│   │       ├── detector.py               adaptive-threshold clustering
│   │       └── phenotype.py              sphericity + coherence rule
│   └── experiment/
│       ├── spec.py                       experiment specification
│       ├── runner.py                     single-classifier driver
│       ├── classifier_comparison.py      RF + SVM + LR on shared splits
│       └── classifier_stats.py           DeLong, calibration, DCA
│
└── run/
    ├── __init__.py
    ├── config.py                          shared runtime configuration
    ├── run_experiment.py                  driver 1: single-patient pipeline
    ├── run_classifier_comparison.py       driver 2: reproduce paper analysis
    └── slurm/
        └── run_supervised.sh              SLURM launcher template
```

---

## Installation

```bash
git clone https://github.com/<your-username>/deepneurobold.git
cd deepneurobold
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**External tools expected on `$PATH`** (typically provided through an
Apptainer / Singularity image on HPC):

- **FSL** (`flirt`, `bet2`, `fslmaths`, `fslmerge`, `fslsplit`, `fast`, `fslcpgeom`)
- **ANTs** Python bindings (`antspyx`) for registration
- **dcm2niix** for DICOM conversion
- **SynthSeg** for brain parcellation (optional; pipeline falls back to
  uniform background sampling if absent)

Python 3.9+ recommended.

---

## Configuration

All filesystem locations are resolved from environment variables at runtime,
so no site-specific path is baked into the code.

Copy the template and fill in your own values:

```bash
cp .env.example .env
```

`.env`:

```
DNB_DATA_ROOT=/path/to/patient/data
DNB_OUTPUT_ROOT=/path/to/pipeline/output
DNB_FOLLOWUP_ROOT=/path/to/registered/followups   # optional
DNB_ENV_PYTHON=/path/to/venv/bin/python           # optional (SLURM only)
DNB_SURGERY_DATES_CSV=/path/to/surgery_dates.csv  # optional
```

Source the file before running any driver:

```bash
set -a && source .env && set +a
```

**Expected data layout under `DNB_DATA_ROOT`:**

```
$DNB_DATA_ROOT/
    Patient_01/
        T1/                                  raw T1 DICOM series
        T1CE/                                raw post-gadolinium T1 series
        T2/                                  raw T2 series
        FLAIR/                               raw FLAIR series
        BOLD/                                raw BOLD-fMRI series
        Respiract/                           gas paradigm log (optional)
        PREPROCESSING/                       (produced by driver 1)
            t1/                              registered T1 + brain mask
            bold/                            registered BOLD + smoothing
            mask/                            tumour segmentation
```

---

## Usage

### Driver 1: run the full pipeline on one patient

```bash
python -m run.run_experiment --patient-dir $DNB_DATA_ROOT/Patient_01
```

Preprocesses the patient (BOLD to T1, smoothing, tissue segmentation), trains
the voxel-wise classifier under spatial grid-block cross-validation, projects
the probability map, extracts hotspots and classifies the phenotype. Writes
all outputs under `$DNB_OUTPUT_ROOT/Patient_01/`.

### Driver 2: reproduce the paper analysis on the full cohort

```bash
python -m run.run_classifier_comparison --patient-dir $DNB_DATA_ROOT/Patient_01
```

Same as driver 1 but runs three classifiers (Random Forest, Linear SVM with
Platt calibration, L2 Logistic Regression) on identical spatial splits and
reports the DeLong pairwise comparison used in the paper.

### On HPC (SLURM)

The `run/slurm/run_supervised.sh` template shows how to wrap driver 2 in a
SLURM array job (one patient per array task).

---

## Method summary

### Voxel-wise classifier

Positive class: tumour core (necrosis and/or enhancing).
Negative class: healthy parenchyma, drawn 1/3 from grey matter and 2/3 from
white matter when SynthSeg is available (uniform otherwise). CSF is always
excluded from training.

Cross-validation is spatial: the brain is partitioned into 20 mm cubic
grid-blocks; four folds are assembled by round-robin over blocks and a 30 mm
buffer separates train from test to prevent spatial leakage.

Three classifiers are compared under identical splits: Random Forest,
Linear SVM (with Platt scaling), L2 Logistic Regression. Random Forest is
the production classifier.

### Hotspot detection

Applied on the ensemble probability map restricted to the peritumoural
edema. Voxels above an adaptive threshold `p95` of the edema distribution
(falling back to `p90` when `p95` yields no supra-threshold cluster) are
retained. Connected components (18-connectivity) are filtered by minimum
volume of 150 mm^3 and ranked by

```
score(C) = mean_p(C) * volume(C) / (distance_to_core(C) + 1)
```

The top three clusters per patient enter downstream analyses.

### Hotspot phenotyping

Each patient is assigned one of `focal`, `multifocal`, `diffuse` or
`empty / subthreshold` following a rule that combines the sphericity of
the largest component and the mean intra-component BOLD time-series
coherence.

---

## Reproducibility

- All random processes are seeded (`seed=42` unless overridden).
- Spatial CV splits are cached by the SHA-1 of the split definition, so
  re-runs are byte-identical unless the split definition changes.
- The two drivers share the same configuration surface (env vars +
  command-line arguments); no hidden state.

---

## Citation

Manuscript in preparation. Until publication, please cite this repository
directly.

Key methodological references:

- Billot B. et al. *SynthSeg*, MedIA 2023 — brain parcellation.
- DeLong E.R. et al. *Comparing the areas under two or more correlated
  receiver operating characteristic curves*, Biometrics 1988.
- Breiman L. *Random Forests*, Machine Learning 2001.

---

## License

MIT License (see `LICENSE`).

Patient data is **excluded** from the repository. Any site running this
pipeline must comply with its own institutional data-protection and ethics
requirements.

---

## Contact

For questions about the pipeline, please open an issue on this repository.
Corresponding author details are listed in the associated publication.
