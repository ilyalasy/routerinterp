#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

sbatch --gpus=1 --job-name=select-olmoe scripts/slurm/run.sbatch \
    select --config configs/olmoe.yaml --batch-size 256

sbatch --gpus=1 --job-name=select-gpt_oss-s64 scripts/slurm/run.sbatch \
    select --config configs/gpt_oss_s64.yaml --batch-size 128

sbatch --gpus=1 --job-name=select-gpt_oss-s128 scripts/slurm/run.sbatch \
    select --config configs/gpt_oss_s128.yaml --batch-size 128
