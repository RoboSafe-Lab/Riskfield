#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=03:00:00
#SBATCH --job-name=adcand --output=/users/cw3005/riskfield/adcand.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_VMETHOD=poly RF_PDEG=3 RF_SEV=closing RF_LOG=1,1000 RF_ESTRIP=1

echo "=== AD4CHE PET-positive candidates (site 17) ==="
RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
RF_SCENE_IDS=2214,2238,2049,1618,1684,2301,2406,2206 \
RF_TAGS=s2214,s2238,s2049,s1618,s1684,s2301,s2406,s2206 RF_OUTDIR=qual_ad_cand \
python -u scripts/qualitative_map.py > re_adcand.out 2>&1
echo "RC=$?"
grep -E 'ANALYSIS|ESTRIP|RESULT' re_adcand.out
ls qual_ad_cand/
echo ADCAND_DONE
