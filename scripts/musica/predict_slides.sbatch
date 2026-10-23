#!/bin/bash
#SBATCH --job-name=tc-predict-slides
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# Usage (from the repo root on MUSICA):
#   sbatch scripts/musica/predict_slides.sbatch \
#       --slides-glob '/data/fs201053/rh02261/data/prostate_mug/*.svs' --num-slides 10
#
# Expects $DATA/tc-env.sh (uv cache, libvips/openslide on LD_LIBRARY_PATH, HF_HOME)
# and the Hugging Face token in $DATA/.hf_token (Virchow2 is gated).

set -euo pipefail

source "$DATA/tc-env.sh"
HF_TOKEN=$(tr -d '[:space:]' < "$DATA/.hf_token")
export HF_TOKEN

uv run --frozen python -m scripts.musica.predict_slides \
    --checkpoint "$DATA/checkpoints/linear_virchow2_lbfgs_0e2230c7.ckpt" \
    --output-dir "$DATA/predictions/${SLURM_JOB_NAME}-${SLURM_JOB_ID}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-16}" \
    "$@"
