#!/bin/bash
#SBATCH --job-name=rf-eval-wm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Job (a): authoritative location-aware evaluation of the all-33 world-model
# checkpoint. Prints metrics to stdout (no wandb collapse).
set -euo pipefail
PROJECT_DIR="${HOME}/riskfield"
CONDA_ENV="${CONDA_ENV:-riskflow}"

export RF_CKPT="${RF_CKPT:-serialized/riskflow_ind_1.pt}"
export RF_SITE_SCOPE="${RF_SITE_SCOPE:-all}"
export RF_USE_WORLD_MODEL="${RF_USE_WORLD_MODEL:-1}"

echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-na}"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
python scripts/eval_report_all.py
echo "EVAL DONE"
