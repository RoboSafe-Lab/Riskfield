#!/usr/bin/env bash
# Deploy the Riskfield project + InD data to the DMOG Slurm cluster and submit
# the training job. Run from the Riskfield/ directory (Git-bash on Windows OK).
#
#   bash scripts/deploy_dmog.sh code     # push code only (fast, ~MBs)
#   bash scripts/deploy_dmog.sh data     # push InD dataset (~1.8 GB, slow)
#   bash scripts/deploy_dmog.sh submit   # sbatch the training job
#   bash scripts/deploy_dmog.sh all      # code + data + submit
set -euo pipefail

KEY="C:/Users/wchen/.ssh/id_alcescluster"
HOST="cw3005@dmog.hw.ac.uk"
REMOTE="riskfield"
SSH="ssh -F none -o BatchMode=yes -i ${KEY} ${HOST}"
SCP="scp -F none -o BatchMode=yes -i ${KEY}"

push_code() {
  echo ">> packing code (excluding dataset/caches)"
  tar --exclude='datasets/inD' --exclude='**/__pycache__' --exclude='*.pt' \
      --exclude='videos' --exclude='wandb' --exclude='.git' \
      -czf /tmp/riskfield_code.tgz .
  $SSH "mkdir -p ~/${REMOTE}"
  $SCP /tmp/riskfield_code.tgz "${HOST}:~/${REMOTE}/"
  $SSH "cd ~/${REMOTE} && tar -xzf riskfield_code.tgz && rm riskfield_code.tgz && echo code-deployed"
}

push_data() {
  echo ">> uploading InD data (~1.8 GB, this is slow)"
  $SSH "mkdir -p ~/${REMOTE}/data"
  # Stream a tar over ssh so it survives one connection and avoids 132 round-trips.
  tar -czf - -C datasets/inD/data . | \
    $SSH "tar -xzf - -C ~/${REMOTE}/data && echo data-deployed"
}

submit() {
  $SSH "cd ~/${REMOTE} && sbatch scripts/train_dmog.slurm"
}

case "${1:-all}" in
  code)   push_code ;;
  data)   push_data ;;
  submit) submit ;;
  all)    push_code; push_data; submit ;;
  *) echo "usage: $0 {code|data|submit|all}"; exit 1 ;;
esac
