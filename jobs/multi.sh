#!/bin/bash
#SBATCH --job-name=neg
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Usage: sbatch [--gres=gpu:N --cpus-per-task=.. --mem=..] -J <name> jobs/multi.sh <command file>
# Each line of the command file is "<log name> [VAR=value ...] <python module and arguments>".
# Lines are divided between the GPUs of the job; each GPU runs its lines in sequence.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
CMDS=$1; J=$SLURM_JOB_ID
echo "node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} start=$(date)"
source jobs/setup_env.sh
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NG=${#GPUS[@]}
ST="logs/status_$J.txt"; : > "$ST"
for g in $(seq 0 $((NG - 1))); do
  (
    i=0
    while IFS= read -r line || [ -n "$line" ]; do
      [ -z "$line" ] && continue
      if [ $((i % NG)) -eq "$g" ]; then
        name=${line%% *}; cmd=${line#* }; envs=()
        while [[ "${cmd%% *}" == [A-Z_]*=* ]]; do envs+=("${cmd%% *}"); cmd=${cmd#* }; done
        vis=${GPUS[$g]}
        [[ " ${envs[*]-} " == *" ALLGPUS=1 "* ]] && vis=$CUDA_VISIBLE_DEVICES     # a line with ALLGPUS=1 sees all GPUs of the job
        if env ${envs[@]+"${envs[@]}"} CUDA_VISIBLE_DEVICES=$vis "$PY" -m $cmd > "logs/${name}_$J.log" 2>&1; then
          echo "ok $name gpu$g $(date +%T)" | tee -a "$ST"
        else
          echo "FAILED $name gpu$g $(date +%T)" | tee -a "$ST"; tail -6 "logs/${name}_$J.log"
        fi
      fi
      i=$((i + 1))
    done < "$CMDS"
  ) &
done
wait
if grep -q FAILED "$ST"; then echo "JOB FAILED $(date)"; exit 1; fi
echo "JOB COMPLETE $(date)"
