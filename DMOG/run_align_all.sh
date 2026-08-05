#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=48G --time=03:00:00
#SBATCH --job-name=alignall --output=/users/cw3005/riskfield/alignall.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
# Re-render ALL page animations with a FIXED crop aspect (1.18) so every video
# fills the 760x680 frame identically -> same content/colorbar height -> the
# parallel cards align. Same fixes as before (off-map mask, P_ego floor, cloud
# floor) + H.264 (libopenh264). Scene IDs recovered from run_anim_v2*.sh.
export RF_GRID=64 RF_SEV=closing RF_LOG=1,1000 RF_CYCLES=6 RF_NEAR=5 RF_FPS=9
export RF_MAPMASK=1 RF_PEGO_FLOOR=0.05 RF_CLOUD_FLOOR=0.25 RF_CROP_ASPECT=1.18
FF=/users/cw3005/anaconda3/bin/ffmpeg
EGO_I=serialized/riskflow_ind_8.pt; JNT_I=serialized/riskflow_ind_7.pt
EGO_A=serialized/riskflow_ad4che_ego.pt; JNT_A=serialized/riskflow_ad4che_joint.pt
EGO_R=serialized/riskflow_round_ego.pt; JNT_R=serialized/riskflow_round_joint.pt

mp4 () {  # $1 name
  "$FF" -y -i anim_$1.gif -c:v libopenh264 -profile:v constrained_baseline -b:v 1200k \
    -pix_fmt yuv420p -movflags +faststart -vf "scale='min(960,iw)':-2" $1.mp4 2>cva_$1.err \
    && echo "MP4 $1 OK $("$FF" -hide_banner -i $1.mp4 2>&1 | grep -oE 'Video: [a-z0-9]+')" \
    || { echo "MP4 $1 FAIL"; tail -2 cva_$1.err; }
}

# ---- InD ----
RF_DATASET=ind RF_DT=0.08 RF_DIMS=scene_dims_ind.npz RF_CKPT_EGO=$EGO_I RF_CKPT_JOINT=$JNT_I \
  RF_SCENE_IDX=1500 RF_OUT=anim_ind_conflict.gif python -u scripts/animate_joint.py >a_i1.out 2>&1; grep -E 'saved anim|Traceback' a_i1.out|head -2; mp4 ind_conflict
RF_DATASET=ind RF_DT=0.08 RF_DIMS=scene_dims_ind.npz RF_CKPT_EGO=$EGO_I RF_CKPT_JOINT=$JNT_I \
  RF_SCENE_IDX=56660 RF_OUT=anim_ind_conflict2.gif python -u scripts/animate_joint.py >a_i2.out 2>&1; grep -E 'saved anim|Traceback' a_i2.out|head -2; mp4 ind_conflict2
# ---- AD4CHE ----
RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz RF_CKPT_EGO=$EGO_A RF_CKPT_JOINT=$JNT_A \
  RF_SCENE_IDX=530 RF_OUT=anim_ad4che_site17.gif python -u scripts/animate_joint.py >a_a1.out 2>&1; grep -E 'saved anim|Traceback' a_a1.out|head -2; mp4 ad4che_site17
RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz RF_CKPT_EGO=$EGO_A RF_CKPT_JOINT=$JNT_A \
  RF_SCENE_IDX=190 RF_OUT=anim_ad4che_site15.gif python -u scripts/animate_joint.py >a_a2.out 2>&1; grep -E 'saved anim|Traceback' a_a2.out|head -2; mp4 ad4che_site15
# ---- rounD ----
RF_DATASET=round RF_DT=0.08 RF_DIMS=scene_dims_round.npz RF_CKPT_EGO=$EGO_R RF_CKPT_JOINT=$JNT_R \
  RF_SCENE_IDX=90 RF_OUT=anim_round_conflict.gif python -u scripts/animate_joint.py >a_r1.out 2>&1; grep -E 'saved anim|Traceback' a_r1.out|head -2; mp4 round_conflict
RF_DATASET=round RF_DT=0.08 RF_DIMS=scene_dims_round.npz RF_CKPT_EGO=$EGO_R RF_CKPT_JOINT=$JNT_R \
  RF_SCENE_IDX=0 RF_OUT=anim_round_conflict2.gif python -u scripts/animate_joint.py >a_r2.out 2>&1; grep -E 'saved anim|Traceback' a_r2.out|head -2; mp4 round_conflict2
# ---- counterfactual (AD4CHE scene 2320, three ego-speed scales) ----
for spec in "brake 0.7" "maintain 1.0" "accel 1.3"; do
  set -- $spec; nm=$1; vs=$2
  RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz RF_CKPT_EGO=$EGO_A RF_CKPT_JOINT=$JNT_A \
    RF_SCENE_IDX=2320 RF_VSCALE=$vs RF_TITLE="$nm x$vs" RF_OUT=anim_cf_$nm.gif \
    python -u scripts/animate_joint.py >a_cf$nm.out 2>&1; grep -E 'saved anim|Traceback' a_cf$nm.out|head -2; mp4 cf_$nm
done
echo ALIGNALL_DONE
