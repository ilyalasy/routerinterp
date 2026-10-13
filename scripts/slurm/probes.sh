#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

models=(olmoe gpt_oss_s64 gpt_oss_s128)

for model in "${models[@]}"; do
    sbatch --job-name="probe-${model}" scripts/slurm/run.sbatch \
        probe --config "configs/${model}.yaml"
done
