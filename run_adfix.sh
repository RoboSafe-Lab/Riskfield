#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=02:00:00
#SBATCH --job-name=adfix --output=/users/cw3005/riskfield/adfix.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_DT=0.0667 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=1
export RF_DATASET=ad4che RF_DIMS=scene_dims_ad4che.npz
export RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt
export RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt
export RF_SCENE_IDS=1252,1915 RF_TAGS=a1252,a1915 RF_OUTDIR=qual_ad_fix
python -u scripts/qualitative_map.py > adfix.out 2>&1
echo "RC=$?"
grep -E 'ANALYSIS|ESTRIP' adfix.out
echo ADFIX_DONE
