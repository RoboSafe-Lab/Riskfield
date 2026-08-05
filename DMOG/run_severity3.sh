#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=06:00:00
#SBATCH --job-name=sev3 --output=/users/cw3005/riskfield/sev3.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=48 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing

echo "=== InD SEVERITY ==="
RF_DATASET=ind RF_DT=0.08 RF_STRIDE=20 RF_DIMS=scene_dims_ind.npz \
RF_LABELS=conflict_labels_pet.npz \
python -u scripts/severity_eval2.py 2>&1 | grep -E "RESULT|==="

echo "=== AD4CHE SEVERITY ==="
RF_DATASET=ad4che RF_DT=0.0667 RF_STRIDE=5 RF_DIMS=scene_dims_ad4che.npz \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
RF_LABELS=conflict_labels_ad4che_pet.npz \
python -u scripts/severity_eval2.py 2>&1 | grep -E "RESULT|==="

echo "=== rounD SEVERITY ==="
RF_DATASET=round RF_DT=0.08 RF_STRIDE=2 RF_DIMS=scene_dims_round.npz \
RF_CKPT_EGO=serialized/riskflow_round_ego.pt RF_CKPT_JOINT=serialized/riskflow_round_joint.pt \
RF_LABELS=conflict_labels_round_pet.npz \
python -u scripts/severity_eval2.py 2>&1 | grep -E "RESULT|==="
echo SEV3_DONE
