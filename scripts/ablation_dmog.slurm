#!/bin/bash
#SBATCH --job-name=rf-ablation-nowm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Job (b): no-world-model ablation on all 33 recordings (same config as the
# world-model run, RF_USE_WORLD_MODEL=0). After training, run the standalone
# evaluator on the freshly serialized checkpoint for clean metrics that can be
# placed side-by-side with the world-model numbers.
set -euo pipefail
PROJECT_DIR="${HOME}/riskfield"
CONDA_ENV="${CONDA_ENV:-riskflow}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RF_USE_WORLD_MODEL=0
export RF_SITE_SCOPE=all

echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-na}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
test -f data/08_recordingMeta.csv || { echo "ERROR: data/ missing"; exit 1; }

# Snapshot existing checkpoints so we can identify the new one.
before=$(ls serialized/*.pt 2>/dev/null | sort || true)
python main.py
after=$(ls serialized/*.pt 2>/dev/null | sort || true)
newckpt=$(comm -13 <(echo "$before") <(echo "$after") | tail -1)
[ -z "$newckpt" ] && newckpt=$(ls -t serialized/*.pt | head -1)
echo "ABLATION checkpoint: $newckpt"

# Clean authoritative metrics for the no-WM model.
RF_CKPT="$newckpt" RF_USE_WORLD_MODEL=0 RF_SITE_SCOPE=all \
  python scripts/eval_report_all.py
echo "ABLATION DONE"
