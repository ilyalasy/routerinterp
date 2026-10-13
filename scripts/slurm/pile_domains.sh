#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

for model in gpt_oss_s64 gpt_oss_s128; do
    pile_id=$(sbatch --parsable --job-name="pile-domains-${model}" scripts/slurm/run.sbatch \
        pile-domains \
        --collect "artifacts/${model}/collect" \
        --out "artifacts/${model}/metrics/pile_domains.parquet")
    sbatch --dependency="afterok:${pile_id}" \
        --job-name="pile-plot-${model}" scripts/slurm/run.sbatch \
        plot-pile-domains \
        --metrics "artifacts/${model}/metrics/pile_domains.parquet" \
        --out "artifacts/${model}/figures/expert_pile_distribution.png"
done
