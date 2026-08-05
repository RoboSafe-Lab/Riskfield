#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=48G --time=01:00:00
#SBATCH --job-name=rdfix --output=/users/cw3005/riskfield/rdfix.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
# rounD page videos: off-map mask + P_ego display floor + cloud-tail floor.
# round_conflict = scene 660 (genuine 240 J conflict);
# round_conflict2 = scene 0 (busy roundabout, entering ego meets circulating
#   traffic) -- replaces zoomed scene 257 whose cloud tail bled off-road.
export RF_GRID=64 RF_SEV=closing RF_LOG=1,1000 RF_CYCLES=6 RF_NEAR=5 RF_FPS=9
export RF_MAPMASK=1 RF_PEGO_FLOOR=0.05 RF_CLOUD_FLOOR=0.25
export RF_DATASET=round RF_DT=0.08 RF_DIMS=scene_dims_round.npz
export RF_CKPT_EGO=serialized/riskflow_round_ego.pt
export RF_CKPT_JOINT=serialized/riskflow_round_joint.pt
FF=/users/cw3005/anaconda3/bin/ffmpeg
for pair in "660 round_conflict" "0 round_conflict2"; do
  set -- $pair; SC=$1; NAME=$2
  echo "==== $NAME (scene $SC) ===="
  RF_SCENE_IDX=$SC RF_OUT=anim_${NAME}.gif python -u scripts/animate_joint.py > rdfix_$SC.out 2>&1
  grep -E 'RESULT saved anim|off-map|Error|Traceback' rdfix_$SC.out | head
  "$FF" -y -i anim_${NAME}.gif -c:v libopenh264 -profile:v constrained_baseline -b:v 1200k -movflags +faststart -pix_fmt yuv420p \
    -vf "scale='min(960,iw)':-2" ${NAME}.mp4 2>conv_${NAME}.err \
    && echo "MP4 $NAME OK size=$(stat -c%s ${NAME}.mp4)" || { echo "MP4 $NAME FAIL"; tail -3 conv_${NAME}.err; }
done
echo RDFIX_DONE
