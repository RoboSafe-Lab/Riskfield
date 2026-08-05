#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=04:00:00
#SBATCH --job-name=margin0 --output=/users/cw3005/riskfield/margin0.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GT_GEOM=1 RF_MODE=motion RF_VMIN=2.0 RF_TAU_PET=1.0 RF_MARGIN=0

echo "=== generate margin-0 honest labels (strided to match eval) ==="
RF_DATASET=ind RF_DT=0.08 RF_STRIDE=20 RF_OUT=conflict_labels_honest_m0.npz python -u scripts/conflict_labels.py 2>&1 | grep RESULT
RF_DATASET=ad4che RF_DT=0.0667 RF_STRIDE=5 RF_OUT=conflict_labels_ad4che_honest_m0.npz python -u scripts/conflict_labels.py 2>&1 | grep RESULT
RF_DATASET=round RF_DT=0.08 RF_STRIDE=2 RF_OUT=conflict_labels_round_honest_m0.npz python -u scripts/conflict_labels.py 2>&1 | grep RESULT

echo "=== threshold PET<1.0 ==="
python scripts/_make_pet_m0.py

echo "=== eval vs margin-0 PET labels (cached scores) ==="
RF_GRID=48 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_DATASET=ind RF_DT=0.08 RF_STRIDE=20 RF_DIMS=scene_dims_ind.npz RF_LABEL_FILES=conflict_labels_pet_m0.npz python -u scripts/eval_conflict.py 2>&1 | grep RESULT
RF_GRID=48 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_DATASET=ad4che RF_DT=0.0667 RF_STRIDE=5 RF_DIMS=scene_dims_ad4che.npz RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt RF_LABEL_FILES=conflict_labels_ad4che_pet_m0.npz python -u scripts/eval_conflict.py 2>&1 | grep RESULT
RF_GRID=48 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_DATASET=round RF_DT=0.08 RF_STRIDE=2 RF_DIMS=scene_dims_round.npz RF_CKPT_EGO=serialized/riskflow_round_ego.pt RF_CKPT_JOINT=serialized/riskflow_round_joint.pt RF_LABEL_FILES=conflict_labels_round_pet_m0.npz python -u scripts/eval_conflict.py 2>&1 | grep RESULT
echo MARGIN0_DONE
