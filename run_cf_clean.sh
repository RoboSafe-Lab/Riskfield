#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=02:00:00
#SBATCH --job-name=cfclean --output=/users/cw3005/riskfield/cfclean.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
# clean counterfactual hero = AD4CHE scene 2320 (direction-consistent truck overtake),
# replacing the old scene-530 adjacent-lane/velocity-flip demo. Same render settings as
# run_cf_anim2.sh: grid 64, shared log colorbar 1..1000 J, vscale 0.7/1.0/1.3.
export RF_GRID=64 RF_SEV=closing RF_LOG=1,1000 RF_CYCLES=6 RF_NEAR=5 RF_FPS=9
export RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz
export RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt
export RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt

for spec in "brake 0.7" "maintain 1.0" "accel 1.3"; do
  set -- $spec; name=$1; vs=$2
  echo "CF_${name} (vscale=$vs)"
  RF_SCENE_IDX=2320 RF_VSCALE=$vs RF_TITLE="$name x$vs" RF_OUT=anim_cf2320_$name.gif \
    python -u scripts/animate_joint.py > cf2320_$name.out 2>&1
  grep -E 'peak-risk|saved ' cf2320_$name.out
  echo "PEAKMAX_${name}=$(grep -oE 'E=[0-9.eE+]+ J' cf2320_$name.out | grep -oE '[0-9.eE+]+' | sort -g | tail -1) J"
done

echo "=== ffmpeg convert gif->mp4 ==="
if command -v ffmpeg >/dev/null 2>&1; then FF=ffmpeg; else FF=$(ls /usr/bin/ffmpeg /opt/*/bin/ffmpeg 2>/dev/null | head -1); fi
echo "FF=$FF"
for name in brake maintain accel; do
  "$FF" -y -i anim_cf2320_$name.gif -movflags faststart -pix_fmt yuv420p \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" cf2320_$name.mp4 2>conv_$name.err \
    && echo "MP4_${name}_OK" || { echo "MP4_${name}_FAIL"; tail -3 conv_$name.err; }
done
ls -la cf2320_*.mp4 anim_cf2320_*.gif 2>/dev/null
echo "CF_CLEAN_DONE"
