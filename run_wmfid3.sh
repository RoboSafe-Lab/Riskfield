#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=06:00:00
#SBATCH --job-name=wmfid3 --output=/users/cw3005/riskfield/wmfid3.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_MAX_SCENES=400 RF_MANEUVER_Q=0.75

echo "=== InD WMFID (sanity) ==="
RF_DATASET=ind \
RF_CKPT_EGO=serialized/riskflow_ind_8.pt RF_CKPT_JOINT=serialized/riskflow_ind_7.pt \
python -u scripts/wm_fidelity.py 2>&1 | grep -E "RESULT|==="

echo "=== AD4CHE WMFID ==="
RF_DATASET=ad4che \
RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt \
python -u scripts/wm_fidelity.py 2>&1 | grep -E "RESULT|==="

echo "=== rounD WMFID ==="
RF_DATASET=round \
RF_CKPT_EGO=serialized/riskflow_round_ego.pt RF_CKPT_JOINT=serialized/riskflow_round_joint.pt \
python -u scripts/wm_fidelity.py 2>&1 | grep -E "RESULT|==="
echo WMFID3_DONE
