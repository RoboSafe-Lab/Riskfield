#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=03:00:00
#SBATCH --job-name=sev_pet --output=/users/cw3005/riskfield/sev_pet.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=48 RF_DT=0.08 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing
RF_DATASET=ind RF_DIMS=scene_dims_ind.npz RF_LABELS=conflict_labels_pet.npz \
python -u scripts/severity_eval2.py 2>&1 | grep -E "RESULT|rho|n="
echo SEV_PET_DONE
