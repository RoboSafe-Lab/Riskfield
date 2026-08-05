#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=01:30:00
#SBATCH --job-name=qual --output=/users/cw3005/riskfield/qual.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000
RF_DATASET=ind RF_DT=0.08 RF_DIMS=scene_dims_ind.npz \
RF_SCENE_IDS=44040,56660 RF_TAGS=crit,ncrit RF_OUTDIR=qual_ind_sq \
python -u scripts/qualitative_map.py > re_qual.out 2>&1
echo "RC=$?"
grep -E 'ANALYSIS|RESULT (crit|ncrit)|Traceback|Error' re_qual.out
ls qual_ind_sq/
echo QUAL_DONE
