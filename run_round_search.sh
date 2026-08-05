#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=06:00:00
#SBATCH --job-name=rd_search --output=/users/cw3005/riskfield/rd_search.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_SEV=closing RF_LOG=1,1000 RF_CYCLES=6 RF_NEAR=5 RF_FPS=9
export RF_DATASET=round RF_DT=0.08 RF_DIMS=scene_dims_round.npz
export RF_CKPT_EGO=serialized/riskflow_round_ego.pt
export RF_CKPT_JOINT=serialized/riskflow_round_joint.pt

for S in 660 995 985 201 8 34 10 29 996 673; do
  RF_SCENE_IDX=$S RF_OUT=anim_round_s$S.gif python -u scripts/animate_joint.py > rd_$S.out 2>&1
  PK=$(grep -oE 'E=[0-9.e+-]+ J' rd_$S.out | grep -oE '[0-9.e+-]+' | sort -g | tail -1)
  echo "SCENE $S peakmax=$PK"
done
echo RD_SEARCH_DONE
