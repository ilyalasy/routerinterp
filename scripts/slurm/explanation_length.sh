#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

models=(gpt_oss_s64 gpt_oss_s128)
ns=(1 5 10 20 45)

for model in "${models[@]}"; do
    eval_ids=()
    eval_args=()
    for n in "${ns[@]}"; do
        select_id=$(sbatch --parsable --gpus=1 --job-name="length-select-${model}-n${n}" scripts/slurm/run.sbatch \
            select \
            --config "configs/${model}.yaml" \
            --n "$n" \
            --batch-size 128 \
            --output-dir "artifacts/${model}/explanation_length/select/n${n}")
        explain_id=$(sbatch --parsable --dependency="afterok:${select_id}" \
            --job-name="length-explain-${model}-n${n}" scripts/slurm/run.sbatch \
            explain \
            --config "configs/${model}.yaml" \
            --selection-dir "artifacts/${model}/explanation_length/select/n${n}" \
            --output-dir "artifacts/${model}/explanation_length/explain/n${n}")
        eval_ids+=("$(sbatch --parsable --dependency="afterok:${explain_id}" \
            --job-name="length-eval-${model}-n${n}" scripts/slurm/run.sbatch \
            evaluate \
            --config "configs/${model}.yaml" \
            --explanations-dir "artifacts/${model}/explanation_length/explain/n${n}" \
            --output-dir "artifacts/${model}/explanation_length/evaluate/n${n}")")
        eval_args+=(--evaluate "artifacts/${model}/explanation_length/evaluate/n${n}")
    done
    sbatch --dependency="afterok:$(IFS=:; echo "${eval_ids[*]}")" \
        --job-name="length-plot-${model}" scripts/slurm/run.sbatch \
        plot-explanation-length \
        --out "artifacts/${model}/figures/f1_vs_tokens_avg.png" \
        "${eval_args[@]}"
done
