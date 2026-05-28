# Environment
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=1

# Paths
PAT=${DNB_DATA_ROOT%/*}/Patient_01
T1="$PAT/PREPROCESSING/t1/t1_reoriented.nii.gz"
BOLD4D="$PAT/PREPROCESSING/bold/BOLD.nii.gz"
MASK="$PAT/PREPROCESSING/t1/t1_brain_mask.nii.gz"

# Sanity checks
command -v flirt
command -v fslmaths

# 1) 3D mean
fslmaths "$BOLD4D" -Tmean "$PAT/PREPROCESSING/bold/BOLD_mean.nii.gz"

# 2) Affine registration (mean BOLD -> T1)
flirt -in "$PAT/PREPROCESSING/bold/BOLD_mean.nii.gz" -ref "$T1" \
      -omat "$PAT/PREPROCESSING/bold/bold2t1.mat" \
      -out "$PAT/PREPROCESSING/bold/BOLD_mean_inT1.nii.gz" \
      -dof 6 -cost normmi -interp trilinear

# 3) Apply to the full 4D series
flirt -in "$BOLD4D" -ref "$T1" \
      -applyxfm -init "$PAT/PREPROCESSING/bold/bold2t1.mat" \
      -out "$PAT/PREPROCESSING/bold/BOLD_inT1.nii.gz" -interp trilinear

# 4) Apply brain mask
fslmaths "$PAT/PREPROCESSING/bold/BOLD_inT1.nii.gz" -mas "$MASK" \
         "$PAT/PREPROCESSING/bold/BOLD_brain_inT1.nii.gz"
fslmaths "$PAT/PREPROCESSING/bold/BOLD_mean_inT1.nii.gz" -mas "$MASK" \
         "$PAT/PREPROCESSING/bold/BOLD_mean_brain_inT1.nii.gz"
