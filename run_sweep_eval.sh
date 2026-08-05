#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=03:00:00
#SBATCH --job-name=sweep_eval --output=/users/cw3005/riskfield/sweep_eval.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=48 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_USE_SCORE_CACHE=1

echo "=== InD ==="
RF_DATASET=ind RF_DT=0.08 RF_STRIDE=20 RF_DIMS=scene_dims_ind.npz \
RF_LABEL_FILES=conflict_labels_petloose.npz,conflict_labels_pet.npz,conflict_labels_petstrict.npz \
python -u scripts/eval_conflict.py 2>&1 | grep RESULT
cp conflict_eval.json sweep_ind.json

echo "=== AD4CHE ==="
RF_DATASET=ad4che RF_DT=0.0667 RF_STRIDE=5 RF_DIMS=scene_dims_ad4che.npz \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
RF_LABEL_FILES=conflict_labels_ad4che_petloose.npz,conflict_labels_ad4che_pet.npz,conflict_labels_ad4che_petstrict.npz \
python -u scripts/eval_conflict.py 2>&1 | grep RESULT
cp conflict_eval.json sweep_ad4che.json

echo "=== rounD ==="
RF_DATASET=round RF_DT=0.08 RF_STRIDE=2 RF_DIMS=scene_dims_round.npz \
RF_CKPT_EGO=serialized/riskflow_round_ego.pt RF_CKPT_JOINT=serialized/riskflow_round_joint.pt \
RF_LABEL_FILES=conflict_labels_round_petloose.npz,conflict_labels_round_pet.npz,conflict_labels_round_petstrict.npz \
python -u scripts/eval_conflict.py 2>&1 | grep RESULT
cp conflict_eval.json sweep_round.json
echo SWEEP_EVAL_DONE
