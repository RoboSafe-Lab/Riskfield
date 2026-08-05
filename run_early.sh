#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=02:00:00
#SBATCH --job-name=early --output=/users/cw3005/riskfield/early.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=1
export RF_DATASET=ind RF_DT=0.08 RF_DIMS=scene_dims_ind.npz RF_OUTDIR=qual_early
rm -f re_early.out
RF_SCENE_IDS=54120 RF_TAGS=s54120 RF_ESTRIP_ONSET=15 python -u scripts/qualitative_map.py >> re_early.out 2>&1
RF_SCENE_IDS=42320 RF_TAGS=s42320 RF_ESTRIP_ONSET=26 python -u scripts/qualitative_map.py >> re_early.out 2>&1
RF_SCENE_IDS=47000 RF_TAGS=s47000 RF_ESTRIP_ONSET=16 python -u scripts/qualitative_map.py >> re_early.out 2>&1
RF_SCENE_IDS=56660 RF_TAGS=s56660 RF_ESTRIP_ONSET=17 python -u scripts/qualitative_map.py >> re_early.out 2>&1
grep -E 'ESTRIP|RESULT s' re_early.out
echo EARLY_DONE
