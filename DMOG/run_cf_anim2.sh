#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=08:00:00
#SBATCH --job-name=cf_anim2 --output=/users/cw3005/riskfield/cf_anim2.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GRID=64 RF_SEV=closing RF_LOG=1,1000 RF_CYCLES=6 RF_NEAR=5 RF_FPS=9
export RF_DATASET=round RF_DT=0.08 RF_DIMS=scene_dims_round.npz
export RF_CKPT_EGO=serialized/riskflow_round_ego.pt
export RF_CKPT_JOINT=serialized/riskflow_round_joint.pt

echo "ROUND_270"
RF_SCENE_IDX=270 RF_OUT=anim_round_c270.gif python -u scripts/animate_joint.py > rc270.out 2>&1
grep -E 'cycle1 |cycle6 |saved anim|invalid' rc270.out

echo "ROUND_257"
RF_SCENE_IDX=257 RF_OUT=anim_round_c257.gif python -u scripts/animate_joint.py > rc257.out 2>&1
grep -E 'cycle1 |cycle6 |saved anim|invalid' rc257.out

# counterfactual: AD4CHE scene 530, three ego-speed scales, fixed log colorbar
export RF_DATASET=ad4che RF_DT=0.0667 RF_DIMS=scene_dims_ad4che.npz
export RF_CKPT_EGO=serialized/riskflow_ad4che_ego.pt
export RF_CKPT_JOINT=serialized/riskflow_ad4che_joint.pt

echo "CF_BRAKE"
RF_SCENE_IDX=530 RF_VSCALE=0.7 RF_TITLE='Brake  -  ego speed x0.7' RF_OUT=anim_cf_brake.gif python -u scripts/animate_joint.py > cfb.out 2>&1
grep -E 'cycle1 |cycle6 |saved anim' cfb.out

echo "CF_MAINTAIN"
RF_SCENE_IDX=530 RF_VSCALE=1.0 RF_TITLE='Maintain  -  baseline' RF_OUT=anim_cf_maintain.gif python -u scripts/animate_joint.py > cfm.out 2>&1
grep -E 'cycle1 |cycle6 |saved anim' cfm.out

echo "CF_ACCEL"
RF_SCENE_IDX=530 RF_VSCALE=1.3 RF_TITLE='Accelerate  -  ego speed x1.3' RF_OUT=anim_cf_accel.gif python -u scripts/animate_joint.py > cfa.out 2>&1
grep -E 'cycle1 |cycle6 |saved anim' cfa.out

echo "CF_ANIM2_DONE"
