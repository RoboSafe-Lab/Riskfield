#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=01:00:00
#SBATCH --job-name=pair --output=/users/cw3005/riskfield/pair.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=1
export RF_DATASET=ind RF_DT=0.08 RF_DIMS=scene_dims_ind.npz RF_OUTDIR=qual_pair
RF_SCENE_IDS=42320 RF_TAGS=early42320 RF_ESTRIP_ONSET=26 RF_ESTRIP_PAIR=1 python -u scripts/qualitative_map.py > re_pair.out 2>&1
grep -E 'ESTRIP|RESULT s' re_pair.out
echo PAIR_DONE
