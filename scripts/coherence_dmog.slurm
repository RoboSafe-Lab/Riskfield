#!/bin/bash
#SBATCH --job-name=rf-coherence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Temporal-coherence measurement answering Reviewer 2 (ITSC 2026 #1201):
# "the risk tensor stacks marginal densities at each horizon step, not a joint
#  distribution over the trajectory ... high-density regions at step k and k+1
#  need not correspond to any coherent trajectory. The authors acknowledge this
#  briefly but do not analyze how much it matters in practice."
#
# The paper claims "Temporally coherent rollout: yes" in the positioning table
# (main.tex:122) against PORA, and asserts it in the abstract and contributions,
# but never measures it. This produces the number.
#
# The control is built in: the ablation family riskflow_ind_0..6 was trained on
# an identical pipeline, schedule and budget, WITHOUT a map encoder, so these
# rows differ only in the conditioning. ind_2 has no world model at all -- its
# flow sees only the broadcast context and the flow's internal time index, which
# is exactly the per-step-independent construction the reviewer describes.
#
#   ind_2  no world model      <- the reviewer's strawman, made concrete
#   ind_1  naive world model   <- unconstrained latent rollout (hurts accuracy)
#   ind_5  residual world mdl  <- the controlled counterpart to ind_2
#   ind_8  deployed ego model  <- carries the map; backs the table's checkmark
#
# Read the GAP line: accel_ratio(model/GT) is dimensionless, so it is comparable
# across models even though the underlying units are normalized. 1.0 = decodes
# as smooth as real driving; >>1 = the fixed-z trajectory jitters, i.e. the
# per-step densities do NOT line up into one coherent path.
set -euo pipefail

PROJECT_DIR="${HOME}/riskfield"
CONDA_ENV="${CONDA_ENV:-riskflow}"

echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-na}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
test -f data/08_recordingMeta.csv || { echo "ERROR: data/ missing"; exit 1; }

export RF_DATASET=ind RF_SITE_SCOPE=all RF_SCENE_LEVEL=0
export RF_MAX_SCENES="${RF_MAX_SCENES:-1500}"
export RF_Z_SAMPLES="${RF_Z_SAMPLES:-16}"
export RF_SEED="${RF_SEED:-0}"

run () {   # run <label> <ckpt> <use_world_model> <use_map> [scene_level]
  echo ""
  echo "######## $1  ($2  UWM=$3 MAP=$4 SCENE=${5:-0})"
  # Never let one checkpoint's mismatch abort the rest (set -e is on).
  RF_CKPT="serialized/$2" RF_USE_WORLD_MODEL="$3" RF_USE_MAP="$4" \
  RF_SCENE_LEVEL="${5:-0}" python scripts/coherence_metric.py \
    || echo "SKIPPED $1 -- see the ABORT line above"
}

run "no world model (per-step independent)" riskflow_ind_2.pt 0 0
run "residual world model"                  riskflow_ind_5.pt 1 0
run "multi-agent action-cond. (ablation ours)" riskflow_ind_6.pt 1 0 1
run "deployed ego model (+map)"             riskflow_ind_8.pt 1 1

# NOT measurable: riskflow_ind_0/ind_1 (naive world model). Those predate the
# residual head, so the current WorldModel has 8 world_model.residual.* params
# the checkpoint cannot fill; loading anyway would measure a partly random
# model. The guard in coherence_metric.py refuses it.

echo ""
echo "COHERENCE DONE -- compare the GAP accel_ratio lines across the four blocks."
