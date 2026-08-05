#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=03:00:00
#SBATCH --job-name=reenergy --output=/users/cw3005/riskfield/reenergy.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=1
RF_DATASET=ind RF_DT=0.08 RF_DIMS=scene_dims_ind.npz \
RF_SCENE_IDS=44040,56660 RF_TAGS=crit,ncrit RF_OUTDIR=qual_ind_re \
python -u scripts/qualitative_map.py > re_ind.out 2>&1
echo "RC=$?"
grep -E 'ANALYSIS' re_ind.out
ls qual_ind_re/qual_map_crit_energy.png qual_ind_re/qual_map_ncrit_energy.png
echo REENERGY_DONE
