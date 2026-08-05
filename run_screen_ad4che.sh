#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=03:00:00
#SBATCH --job-name=screenad --output=/users/cw3005/riskfield/screenad.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=48 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000
RF_DATASET=ad4che RF_DT=0.0667 RF_STRIDE=5 RF_DIMS=scene_dims_ad4che.npz \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
RF_LABELS=conflict_labels_ad4che_pet.npz \
python -u scripts/_screen_ad4che.py 2>&1 | grep -E "RESULT|===|SCREEN_DONE|Error|Traceback"
echo SCREENAD_DONE
