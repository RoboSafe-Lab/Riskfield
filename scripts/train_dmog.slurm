#!/bin/bash
#SBATCH --job-name=riskflow-wm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Train the world-model-conditioned RiskFlow on the DMOG Slurm cluster.
# Submit from the project root:  sbatch scripts/train_dmog.slurm
set -euo pipefail

PROJECT_DIR="${HOME}/riskfield"
CONDA_ENV="${CONDA_ENV:-riskflow}"         # dedicated env; override: CONDA_ENV=foo sbatch ...
export WANDB_MODE="${WANDB_MODE:-offline}" # offline by default; set online if logged in

echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-na}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python -c 'import torch;print("torch",torch.__version__,"cuda_avail",torch.cuda.is_available())'

cd "${PROJECT_DIR}"
# main.py expects the InD data at ./data (InD(root="data")).
test -f data/08_recordingMeta.csv || { echo "ERROR: data/ missing on cluster"; exit 1; }

python main.py
echo "DONE. Serialized model under ${PROJECT_DIR}/serialized/"
