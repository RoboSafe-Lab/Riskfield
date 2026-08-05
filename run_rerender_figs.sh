#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=03:00:00
#SBATCH --job-name=refigs --output=/users/cw3005/riskfield/refigs.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=1

echo "=== fig:ad4che (a1252, a1915) DT=0.0667 ==="
RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
RF_SCENE_IDS=1252,1915 RF_TAGS=a1252,a1915 RF_OUTDIR=qual_ad_re \
python -u scripts/qualitative_map.py > re_ad.out 2>&1
echo "AD_RC=$?"
grep -E 'ANALYSIS|ESTRIP' re_ad.out

echo "=== fig:qualitative (crit=44040, ncrit=56660) DT=0.08 ==="
RF_DATASET=ind RF_DT=0.08 RF_DIMS=scene_dims_ind.npz \
RF_SCENE_IDS=44040,56660 RF_TAGS=crit,ncrit RF_OUTDIR=qual_ind_re \
python -u scripts/qualitative_map.py > re_ind.out 2>&1
echo "IND_RC=$?"
grep -E 'ANALYSIS|ESTRIP' re_ind.out

ls qual_ad_re/ qual_ind_re/
echo REFIGS_DONE
