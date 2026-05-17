# ProvGigaPath comparison plan

Compare ProvGigaPath against Virchow2 with the same labeled tiles, folds,
thresholds, metric selection, and checkpoint convention.

## Backbone-specific constants

- `embedding_model_name`: `ProvGigaPath`
- `embedding_dim`: `1536`
- `embedding_run_id`: `410c8672471348ceb4c58817f70fa097`
- probe head: `nn.Linear(1536, 7)`
- labeled train/val metadata: same `kfold_split/kfold_tiles.parquet`
- labeled test metadata: same `filter_tiles/test_tiles.parquet`
- selection metric: mean validation `f1_macro` over folds

## Stage 1: k-fold weight-decay sweeps

Run both optimizers over the same candidates:

```bash
uv run python -m ml \
  +experiment=ml/linear_classifier_stratified_group_kfold_provgigapath \
  val_fold=0,1,2,3,4 \
  model.weight_decay=0,1e-5,1e-4,1e-3,1e-2 \
  --multirun

uv run python -m ml \
  +experiment=ml/linear_classifier_lbfgs_stratified_group_kfold_provgigapath \
  val_fold=0,1,2,3,4 \
  model.weight_decay=0,1e-5,1e-4,1e-3,1e-2 \
  --multirun
```

After the sweeps, export full tables:

- `docs/sweep_summary_provgigapath_adamw.csv`
- `docs/sweep_summary_provgigapath_lbfgs.csv`

Pick ProvGigaPath's own best AdamW and LBFGS weight decays by mean validation
`f1_macro`. Do not reuse Virchow2's selected values unless the sweep supports it.

## Stage 2: final training

Train on all folds with the selected weight decay for each optimizer:

```bash
uv run python -m ml \
  +experiment=ml/linear_classifier_final_adamw_provgigapath \
  model.weight_decay=<best_adamw_wd>

uv run python -m ml \
  +experiment=ml/linear_classifier_final_lbfgs_provgigapath \
  model.weight_decay=<best_lbfgs_wd>
```

Use the same checkpoint convention as Virchow2:

```text
mlflow-artifacts:/104/<run_id>/artifacts/checkpoints/last/checkpoint.ckpt
```

## Stage 3: held-out test

Evaluate the final checkpoints on the held-out labeled test split:

```bash
uv run python -m ml \
  +experiment=ml/linear_classifier_test_adamw_provgigapath \
  model.weight_decay=<best_adamw_wd> \
  checkpoint="<adamw_final_checkpoint>"

uv run python -m ml \
  +experiment=ml/linear_classifier_test_lbfgs_provgigapath \
  model.weight_decay=<best_lbfgs_wd> \
  checkpoint="<lbfgs_final_checkpoint>"
```

The ground-truth test configs compute metrics and prediction parquet only; they
do not write TIFF prediction maps.
