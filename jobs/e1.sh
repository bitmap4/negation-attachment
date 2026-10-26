#!/bin/bash
#SBATCH --job-name=neg-e1
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --nodelist=gnode044
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e1_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
echo "node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} start=$(date)"
source jobs/setup_env.sh
J=$SLURM_JOB_ID
"$PY" -m neg.e1_behaviour results/e1_base.json hi_dev fr_dev hi_test fr_test > "logs/e1_base_$J.log" 2>&1 \
  && { tail -40 "logs/e1_base_$J.log"; echo "E1 COMPLETE $(date)"; } \
  || { tail -30 "logs/e1_base_$J.log"; echo "E1 FAILED $(date)"; exit 1; }
