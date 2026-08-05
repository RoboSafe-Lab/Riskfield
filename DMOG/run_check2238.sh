#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --mem=32G --time=00:30:00
#SBATCH --job-name=chk2238 --output=/users/cw3005/riskfield/chk2238.log
cd /users/cw3005/riskfield
source /users/cw3005/anaconda3/etc/profile.d/conda.sh
conda activate riskflow
RF_DATASET=ad4che RF_DT=0.0667 RF_SCENE_IDS=2238 \
python -u scripts/_check2238.py 2>&1 | grep -E "RESULT|CHECK_DONE|Error|Traceback"
echo CHK_DONE
