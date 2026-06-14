# Tissue Classification

This repository contains a pipeline for tissue classification in whole-slide
images (WSIs). It includes preprocessing utilities, dataset splitting tools,
machine learning training and evaluation code, Hydra configurations, and
submission scripts for running individual pipeline stages.

## Repository Structure

- `preprocessing/` contains source code for WSI preprocessing, tiling, mask
  generation, embedding extraction, quality control, and related statistics.
- `ml/` contains machine learning code for data loading, model training,
  evaluation, callbacks, and prediction outputs.
- `split/` contains utilities for train/test and k-fold dataset splitting.
- `configs/` contains Hydra configuration files for preprocessing, splitting,
  logging, datasets, and machine learning experiments.
- `scripts/` contains executable submission scripts for running pipeline stages.
- `pyproject.toml` and `uv.lock` define the Python project metadata and locked
  dependencies.
- `LICENSE` contains the MIT License for this repository.

## Setup

The project uses Python 3.12 and `uv` for dependency management.

```bash
uv sync
```

## Usage

Pipeline stages can be run through the corresponding scripts in `scripts/`.
For example:

```bash
uv run python scripts/submit_tiling.py
uv run python scripts/submit_embeddings.py
uv run python scripts/submit_train_linear_probe.py
```

Configuration is managed with Hydra. Base configurations are stored in
`configs/`, with task-specific configurations grouped by pipeline stage.
