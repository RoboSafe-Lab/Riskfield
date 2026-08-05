#!/bin/bash
#SBATCH --job-name=rf-calibj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
PROJECT_DIR="${HOME}/riskfield"
CONDA_ENV="${CONDA_ENV:-riskflow}"
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-na}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export RF_CKPT_EGO="${RF_CKPT_EGO:-serialized/riskflow_ind_8.pt}"
export RF_CKPT_JOINT="${RF_CKPT_JOINT:-serialized/riskflow_ind_7.pt}"
export RF_GRID="${RF_GRID:-64}"
export RF_PER_SCENE="${RF_PER_SCENE:-2000}"
export RF_MAX_SCENES="${RF_MAX_SCENES:-800}"
python -u scripts/calibrate_joint.py
echo "DONE."
