#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

sbatch --gpus=1 --job-name=train-sae scripts/slurm/run.sbatch \
    train-sae --config configs/olmoe.yaml --batch-size 512
