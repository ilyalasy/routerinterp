#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

models=(olmoe gpt_oss_s64 gpt_oss_s128)
methods=(routing_margin rho_useful cosine)
# Paper Fig 5: attribution vs ρ vs cosine, including the n=45 operating point
# and the "past ~50" regime.
ns=(1 2 5 10 20 30 40 50 100 200)

for model in "${models[@]}"; do
    if [[ "$model" == olmoe ]]; then
        batch_size=256
    else
        batch_size=128
    fi
    select_ids=()
    for method in "${methods[@]}"; do
        for n in "${ns[@]}"; do
            select_ids+=("$(sbatch --parsable --gpus=1 \
                --job-name="sweep-select-${model}-${method}-n${n}" \
                scripts/slurm/run.sbatch \
                select \
                --config "configs/${model}.yaml" \
                --method "$method" \
                --n "$n" \
                --batch-size "$batch_size" \
                --output-dir "artifacts/${model}/feature_selection_sweep/${method}/n${n}")")
        done
    done

    args=()
    for method in "${methods[@]}"; do
        for n in "${ns[@]}"; do
            args+=(--selection "artifacts/${model}/feature_selection_sweep/${method}/n${n}/selections/${method}/n${n}.parquet")
        done
    done

    probe_id=$(sbatch --parsable \
        --dependency="afterok:$(IFS=:; echo "${select_ids[*]}")" \
        --job-name="sweep-probe-${model}" scripts/slurm/run.sbatch \
        probe-selection \
        --config "configs/${model}.yaml" \
        --out "artifacts/${model}/metrics/feature_selection_sweep.parquet" \
        "${args[@]}")
    sbatch --dependency="afterok:${probe_id}" \
        --job-name="sweep-plot-${model}" scripts/slurm/run.sbatch \
        plot-feature-selection-sweep \
        --metrics "artifacts/${model}/metrics/feature_selection_sweep.parquet" \
        --out "artifacts/${model}/figures/feature_selection_sweep.png"
done
