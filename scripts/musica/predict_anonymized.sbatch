#!/bin/bash
#SBATCH --job-name=tc-predict-anon
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# Predict slides on MUSICA and export the maps in the geometry and naming of
# their anonymized copies (see export_anonymized.py).
#
# Usage (from the repo root on MUSICA):
#   sbatch scripts/musica/predict_anonymized.sbatch \
#       --slides-glob "$DATA/slides/prostate_mug_linkage/*.svs" --num-slides 8

set -euo pipefail

source "$DATA/tc-env.sh"
HF_TOKEN=$(tr -d '[:space:]' < "$DATA/.hf_token")
export HF_TOKEN

OUT="$DATA/predictions/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

uv run --frozen python -m scripts.musica.predict_slides \
    --checkpoint "$DATA/checkpoints/linear_virchow2_lbfgs_0e2230c7.ckpt" \
    --output-dir "$OUT" \
    --num-workers "${SLURM_CPUS_PER_TASK:-16}" \
    "$@"

uv run --frozen python -m scripts.musica.export_anonymized \
    --predictions-dir "$OUT" \
    --report /data/fs201316/jp50463/anonymize_report.csv \
    --anonymized-dir /data/fs201316/jp50463/annonymized \
    --output-dir "$OUT/anonymized"
