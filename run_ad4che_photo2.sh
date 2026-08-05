#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=01:30:00
#SBATCH --job-name=adphoto2 --output=/users/cw3005/riskfield/adphoto2.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=1 RF_MAPBG=0
RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
RF_SCENE_IDS=2320,1915 RF_TAGS=a2320,a1915 RF_OUTDIR=qual_ad_photo2 \
python -u scripts/qualitative_map.py > re_adphoto2.out 2>&1
echo "RC=$?"
grep -E 'ANALYSIS|DECOMP|RESULT a|Error|Traceback' re_adphoto2.out | head -30
ls qual_ad_photo2/*estrip* 2>/dev/null
echo ADPHOTO2_DONE
