from deepneurobold.config.paths import DATA_ROOT, PYTHON_BIN
import os
from pathlib import Path
import torch

def config_file():
    args = {}

    # ====== PATHS ======
    args["base"] = str(DATA_ROOT.parent.parent)
    args["repo_neurobold"] = f"{args['base']}/deepneurobold"
    args["patients_dir"] = f"{args['base']}/<DATA_ROOT>"

    # ====== ENVIRONMENT ======
    args["conda_env"] = str(Path(PYTHON_BIN).parent.parent)
    args["partition"] = "standard"

    # ====== FSL CONFIG ======
    FSLDIR = "<FSLDIR>"
    os.environ["FSLDIR"] = FSLDIR
    os.environ["PATH"] = f"{FSLDIR}/bin:" + os.environ.get("PATH", "")
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    args["fsldir"] = FSLDIR
    args["fsl_outputtype"] = "NIFTI_GZ"

    # ====== SCRATCH / TMP ======
    tmp_guess = os.environ.get("SLURM_TMPDIR") or f"/sctmp/{os.environ.get('USER','tmp')}"
    os.environ.setdefault("TMPDIR", tmp_guess)

    # ====== THREADS ======
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        os.environ["DNB_PAR"] = slurm_cpus
        ants_threads = int(slurm_cpus)
    else:
        os.environ.setdefault("DNB_PAR", "6")
        ants_threads = int(os.environ.get("DNB_PAR", "6"))

    # ====== BOLD→T1 KNOBS ======
    os.environ.setdefault("DNB_BATCH", "300")
    os.environ.setdefault("DNB_USE_PAR", "1")
    os.environ.setdefault("DNB_INTERP", "nearestneighbour")
    os.environ.setdefault("DNB_INPLACE", "0")
    os.environ.setdefault("DNB_FORCE_TMEAN", "0")

    # ====== BLAS / OMP ======
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # ====== dcm2niix ======
    DCM2NIIX_PATH = str(Path(FSLDIR) / "bin" / "dcm2niix")
    if not Path(DCM2NIIX_PATH).exists():
        print(f"[WARN] dcm2niix not found at {DCM2NIIX_PATH}.")
    args["dcm2niix"] = DCM2NIIX_PATH

    # ====== SHIFT_CORR ======
    args["shift_corr"] = {
        "et_time_col": "MR Time(s)",
        "et_o2_col": "PO2 (mmHg)",
        "et_skiprows": 1,
        "shift_mode": "xcorr_edge",
        "edge_smooth_k": 7,
        "allow_autoshift_et": 1,
    }

    # ====== ANTs ======
    args["ants_transform"] = "SyN"
    args["ants_interp"] = "linear"
    args["ants_winsorize"] = "0.005,0.995"
    args["ants_histmatch"] = True
    args["ants_threads"] = ants_threads

    # ====== DEVICE ======
    cuda_available = torch.cuda.is_available()
    args["device"] = torch.device("cuda" if cuda_available else "cpu")

    # ====== LOG ======
    print("===================================")
    print("DeepNeuroBOLD v2 — Configuration")
    print("===================================")
    print(f"Base path        : {args['base']}")
    print(f"Patients dir     : {args['patients_dir']}")
    print(f"Repo dir         : {args['repo_neurobold']}")
    print(f"FSLDIR           : {args['fsldir']}")
    print(f"TEMP dir (TMPDIR): {os.environ.get('TMPDIR')}")
    print(f"BATCH/PAR        : {os.environ.get('DNB_BATCH')}/{os.environ.get('DNB_PAR')} (use_par={os.environ.get('DNB_USE_PAR')})")
    print(f"Conda env        : {args['conda_env']}")
    print(f"ANTs threads     : {args['ants_threads']}")
    print(f"Device           : {'GPU (CUDA)' if cuda_available else 'CPU only'}")
    print("===================================\n")

    return args
