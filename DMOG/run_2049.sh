#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=01:00:00
#SBATCH --job-name=q2049 --output=/users/cw3005/riskfield/q2049.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=0

echo "=== fig 2049 (4-lens: occupancy / velocity / risk / energy) ==="
RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
RF_SCENE_IDS=2049,2238 RF_TAGS=conv,clos RF_OUTDIR=qual_2049 \
python -u scripts/qualitative_map.py > re_2049.out 2>&1
echo "RC=$?"
grep -E 'ANALYSIS|DECOMP|RESULT' re_2049.out
ls qual_2049/
echo Q2049_DONE
