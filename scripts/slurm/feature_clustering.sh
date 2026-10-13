#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

# Figure 2: n=20 features per expert at the deepest tested layer.
# Selection files are the paper n=45 runs; cluster keeps rank < 20.
models=(olmoe gpt_oss_s64)

cluster_ids=()
for model in "${models[@]}"; do
    cluster_ids+=("$(sbatch --parsable --job-name="cluster-${model}" scripts/slurm/run.sbatch \
        cluster \
        --explanations "artifacts/${model}/explain/feature_autointerp/explanations.jsonl" \
        --selection "artifacts/${model}/select/selections/routing_margin/n45.parquet" \
        --out "artifacts/${model}/metrics/feature_clustering.parquet" \
        --n 20)")
done

sbatch --dependency="afterok:${cluster_ids[0]}:${cluster_ids[1]}" \
    --job-name="plot-clusters" scripts/slurm/run.sbatch \
    plot-clusters \
    --label "OLMoE-1B-7B" \
    --metrics artifacts/olmoe/metrics/feature_clustering.parquet \
    --label "GPT-OSS-20B" \
    --metrics artifacts/gpt_oss_s64/metrics/feature_clustering.parquet \
    --out artifacts/figures/clusters_per_expert.png
