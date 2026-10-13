#!/bin/bash
#SBATCH --job-name=routerinterp
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# GPU jobs pass ``sbatch --gpus=1`` from the wrapper scripts. Do not put
# --gpus in this file: explain/eval/plot jobs are CPU.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# Triton JIT (SAE training, and some SAE encode paths) compiles a tiny CUDA
# helper and needs cuda.h. Isambard's toolkit is not on the default include path.
if [[ -z "${CUDA_HOME:-}" && -d /opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6 ]]; then
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/include" ]]; then
    export CPATH="${CUDA_HOME}/include${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    # bin/activate runs `conda activate "$@"` and would swallow the
    # routerinterp command (collect, select, ...).
    source "${CONDA_ROOT:-${HOME}/miniforge3}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

if [[ $# -eq 0 ]]; then
    echo "Pass a routerinterp command" >&2
    exit 2
fi

routerinterp "$@"
