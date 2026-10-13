#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

models=(olmoe gpt_oss_s64 gpt_oss_s128)
methods=(
    routerinterp
    unigram_autointerp
    bigram_autointerp
    unigram_lookup
    bigram_lookup
    expert_activation
    esb
)

for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        sbatch --job-name="explain-${model}-${method}" scripts/slurm/run.sbatch \
            explain \
            --config "configs/${model}.yaml" \
            --method "$method" \
            --output-dir "artifacts/${model}/explain/${method}"
    done
done
