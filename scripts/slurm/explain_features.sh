#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

sbatch --job-name=explain-features-olmoe scripts/slurm/run.sbatch \
    explain \
    --config configs/olmoe.yaml \
    --method feature_autointerp \
    --output-dir artifacts/olmoe/explain/feature_autointerp

for model in gpt_oss_s64 gpt_oss_s128; do
    sbatch --job-name="explain-features-${model}" scripts/slurm/run.sbatch \
        explain \
        --config "configs/${model}.yaml" \
        --method feature_autointerp \
        --output-dir "artifacts/${model}/explain/feature_autointerp"
done
