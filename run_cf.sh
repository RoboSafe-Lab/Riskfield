#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=03:00:00
#SBATCH --job-name=cftab --output=/users/cw3005/riskfield/cftab.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=48 RF_N_SCENES=20 RF_STRIDE=50 RF_GAMMA=0.95 RF_BRAKE=0.7 RF_ACCEL=1.3
export RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_DT=0.08 RF_DIMS=scene_dims_ind.npz
python -u scripts/cf_table.py 2>&1 | grep -E "RESULT|==="
echo CFTAB_DONE
