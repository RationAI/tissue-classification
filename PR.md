# feat: linear classifier training pipeline on precomputed embeddings

## Summary

Adds an end-to-end ML training pipeline for linear probing on precomputed tile
embeddings. Introduces the embedding dataset preprocessing step, a PyTorch
Lightning training module, and all supporting configs and submission scripts.

## Changes

### Preprocessing
- `preprocessing/_labels.py` — shared label/tissue-prop derivation logic.

### ML training
- `ml/meta_arch.py` — `MetaArch` Lightning module: backbone + decode head +
  CrossEntropyLoss with balanced class weights computed from the train fold.
  Logs per-class metrics, confusion matrices, and per-slide accuracy.
- `ml/data/datasets/embedding_tiles.py` — `EmbeddingTilesDataset`: loads the
  embedding parquet, inner-joins with metadata, and serves `(embedding, label,
  slide_id)` triples. Stays in Arrow for the join to avoid large-list → pandas
  conversion overhead.
- `ml/data/data_module.py` — Lightning `DataModule` wrapping train/val/test splits.
- `ml/callbacks/parquet_prediction_writer.py` — writes model predictions to Parquet.
- `configs/experiment/ml/linear_classifier.yaml` — full experiment config.
- `configs/ml/` — model, data, and trainer sub-configs.
- `scripts/submit_train_linear.py` — MLflow submission script.

## Test plan

- [ ] Run `submit_train_linear.py`; verify training converges and MLflow logs
  loss, macro F1, per-class metrics, and confusion matrix figures.
- [ ] Check class weights are logged under `class_weight/<class>` in MLflow.
- [ ] Confirm `parquet_prediction_writer` produces a valid predictions Parquet
  on the test split.
