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

score_ids=()
for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        eval_id=$(sbatch --parsable --job-name="eval-${model}-${method}" scripts/slurm/run.sbatch \
            evaluate \
            --config "configs/${model}.yaml" \
            --explanations-dir "artifacts/${model}/explain/${method}" \
            --output-dir "artifacts/${model}/evaluate/${method}")
        if [[ ( "$model" == olmoe || "$model" == gpt_oss_s128 ) && "$method" =~ ^(unigram_lookup|esb|routerinterp)$ ]]; then
            score_ids+=("$eval_id")
        fi
    done
done

sbatch --dependency="afterok:$(IFS=:; echo "${score_ids[*]}")" \
    --job-name="plot-explanation-scores" scripts/slurm/run.sbatch \
    plot-explanation-scores \
    --label "gpt-oss-20b" \
    --evaluate artifacts/gpt_oss_s128/evaluate \
    --label "OLMoE-1B-7B" \
    --evaluate artifacts/olmoe/evaluate \
    --out artifacts/figures/routerinterp_results.png
