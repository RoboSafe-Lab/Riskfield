#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=05:00:00
#SBATCH --job-name=honest_lbl --output=/users/cw3005/riskfield/honest_label.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
export RF_GT_GEOM=1 RF_MODE=motion RF_VMIN=2.0 RF_TAU_PET=1.0 RF_TAU_TTC=1.5

echo "=== InD ==="
RF_DATASET=ind RF_DT=0.08 RF_MARGIN=0.3 RF_OUT=conflict_labels_honest.npz \
python -u scripts/conflict_labels.py 2>&1 | grep RESULT

echo "=== AD4CHE ==="
RF_DATASET=ad4che RF_DT=0.0667 RF_MARGIN=0.3 RF_OUT=conflict_labels_ad4che_honest.npz \
python -u scripts/conflict_labels.py 2>&1 | grep RESULT

echo "=== rounD ==="
RF_DATASET=round RF_DT=0.08 RF_MARGIN=0.5 RF_OUT=conflict_labels_round_honest.npz \
python -u scripts/conflict_labels.py 2>&1 | grep RESULT

echo "=== distributions ==="
python -u scripts/metric_dists.py 2>&1 | grep -E "RESULT|saved|WARN"
echo HONEST_DONE
