# Training Methodology — Linear Probe over Frozen Tile Embeddings

This document records *why* the training method is set up the way it is, with the concrete configuration values and results. It covers the three stages — k-fold cross-validation (train/val) used to tune regularization, the final training run, and the held-out test. Methodology and rationale only.

The same method is run over **two frozen backbones — Virchow2 and ProvGigaPath** — so they can be compared on the same labeled tiles, folds, thresholds, filtering, metric selection, and checkpoint convention. Everything in §1b–§5 and §9 is backbone-independent and identical for both. Only the embedding source, embedding dim, probe head width, experiment config names, and the swept/selected weight decay differ per backbone (§1, §6–§8).

---

## 1. Setup

- **Model**: frozen tile embeddings → a single `nn.Linear(embedding_dim, n_classes)`. Backbone is `Identity` (embeddings are precomputed); only the linear head is trained.
- **Classes**: 7 (`collapse_alterations_to_other`).
- **Objective**: `Linear` → `softmax` → `CrossEntropyLoss`.

This is exactly **multinomial logistic regression** on fixed features. That single fact drives every decision below, and it holds for both backbones — the only thing that changes between them is the input feature space.

### Backbones compared

| | Virchow2 | ProvGigaPath |
|---|---|---|
| `embedding_dim` | 2560 | 1536 |
| Probe head | `nn.Linear(2560, 7)` | `nn.Linear(1536, 7)` |
| `embedding_run_id` | `${dataset.mlflow_artifacts.embedding_run_id}` | `410c8672471348ceb4c58817f70fa097` |
| Embedding preprocessing experiment | `preprocessing/embeddings_virchow2_tissue_tiles_05mpp` | (ProvGigaPath equivalent) |
| Selection metric | mean validation `f1_macro` over folds | mean validation `f1_macro` over folds |

Both backbones consume the **same labeled metadata**: `kfold_split/kfold_tiles.parquet` for train/val and `filter_tiles/test_tiles.parquet` for the held-out test. ProvGigaPath does **not** reuse Virchow2's selected weight decays — it runs its own sweep and picks its own per-optimizer values (§6).

Fixed data settings (all stages, both backbones, `configs/ml/linear_classifier.yaml`):

| Setting | Value |
|---|---|
| `tissue_prop_min` | 0.2 |
| Per-class thresholds | Nerve 0.0, Blood 0.0, Connective-Tissue 0.4, Fat 0.6, Epithelium 0.2, Muscle 0.5, Other 0.5 |
| Embedding dim | 2560 (Virchow2) / 1536 (ProvGigaPath) |
| Loss | `CrossEntropyLoss` (class-weighted) |

---

## 1b. Tile filtering and labeling (dataset load time)

Labels are not stored; they are derived at load time from `roi_coverage_*` columns by `EmbeddingTilesDataset._filter_metadata`. The same pipeline runs for train, val, and test, and is identical for both backbones (only the metadata parquet and the fold filter differ — never the backbone). A tile carries one `roi_coverage_<class>` value per class = fraction of the tile covered by that class's annotation.

Pipeline, in order:

1. **Tissue proportion floor.** Keep tiles where `Σ roi_coverage_* ≥ tissue_prop_min` (0.2). Drops tiles that are mostly background/unannotated.
2. **Single-class filter.** Keep tiles with at most **one** non-zero `roi_coverage_*`. Mixed-class tiles are ambiguous for a single-label objective and are dropped (not split, not multi-labeled).
3. **Per-class threshold on the dominant class.** The dominant class = `argmax` of the `roi_coverage_*` values. Keep the tile only if that dominant coverage `≥ thresholds[dominant_class]`. The per-class thresholds (Nerve 0.0, Blood 0.0, Connective-Tissue 0.4, Fat 0.6, Epithelium 0.2, Muscle 0.5, Other 0.5) set how much of the tile the winning class must occupy — looser for sparse/thin tissue (Nerve, Blood), stricter for classes that need a clear majority (Fat, Muscle).
4. **Label assignment.** Surviving tiles get `label = dominant_class`; mapped to `class_indices` (fixed ordering, §9).
5. **Fold filter** (train/val only): `include_folds` / `exclude_folds` on the `fold` column. Test parquet has no `fold` column and skips this.

Each step raises if it empties the dataset. After filtering, the metadata is inner-joined to the embedding parquet on `(slide_id, x, y)`, so the final sample count is the intersection of "passed all filters" and "has a precomputed embedding". Because filtering is keyed on metadata only, the set of labeled tiles is the same for both backbones; they differ only in which embedding vector is joined in. `tile_counts_after_thresholds.ipynb` reproduces steps 1–4 and reports per-class retention.

This filtering is **fixed across all stages and both backbones, and not swept** — it defines the dataset, not a hyperparameter. Only weight decay is tuned (§6).

---

## 2. The learning problem is convex

Cross-entropy over a linear model with fixed inputs is **convex** in the weights; with L2 regularization the optimum is unique. This is true for both backbones — Virchow2 and ProvGigaPath only change the dimensionality and conditioning of that convex problem, not its nature. Consequences:

- There is nothing to explore — training is a deterministic descent toward one point. Epoch count is not a meaningful quantity to tune or report.
- What matters is (a) how close to the optimum we get — measured by the gradient norm ‖∇L‖ / the training-loss plateau, not by epochs — and (b) how much **regularization** shapes that optimum.

Regularization (L2 weight decay) is therefore the **primary and only swept hyperparameter**. Learning rate and epoch count are fixed by the convergence argument, not tuned.

---

## 3. Optimizers

Two optimizers, both valid because the problem is convex, both run for each backbone:

| | AdamW | LBFGS |
|---|---|---|
| Role | sweep + one final variant | sweep + one final variant |
| Type | first-order mini-batch | second-order, full-batch exact solve |
| `learning_rate` | 1e-4 | 1.0 |
| Batching | `batch_size` 1024, shuffle, drop_last | `batch_size` 1e9 (whole set = one batch), no shuffle, no drop_last |
| Extra | — | `max_iter` 100, `history_size` 100, `line_search_fn` strong_wolfe, `tolerance_grad` 1e-7, `tolerance_change` 1e-9 |

LBFGS solves the convex objective in far fewer iterations than first-order descent and lands essentially on the analytic optimum, so the final fit is fast. It is run full-batch by design: the convex objective is defined over all samples at once; mini-batching would defeat the exact-solve rationale. AdamW keeps normal mini-batching and reaches the same optimum if run to convergence. The lower-dimensional ProvGigaPath feature space (1536 vs 2560) is a better-conditioned convex problem, so its weight-decay sensitivity may differ from Virchow2 — this is exactly what the per-backbone sweep in §6 measures.

---

## 4. Convergence criterion, not epoch budget

`max_epochs` is a safety cap, not a target (500 for AdamW via `trainer/default`, 10 for the LBFGS sweep/final). The model runs until the convex objective stops improving:

- Epoch count is not tuned or reported.
- The signal is the training loss / gradient norm flattening.
- Once it plateaus the optimum is reached; further steps cannot help because a convex objective has nothing more to improve toward.

---

## 5. Early-stopping strategy

Config (`configs/ml/trainer/default.yaml`):

```yaml
early_stopping:
  monitor: train/loss_epoch
  mode: min
  patience: 1
  min_delta: 1.0e-4
model_checkpoint:
  monitor: train/loss_epoch
  mode: min
  save_top_k: 1
max_epochs: 500
```

- **Monitor `train/loss_epoch`, not val.** The aim at fit time is to reach the optimum of the *regularized* convex training objective. Generalization is controlled separately, by the regularization strength chosen in cross-validation (§6). The stopping signal is "has the training objective converged" — a proxy for ‖∇L‖ → 0.
- **`patience: 1` is deliberate.** For a convex objective the loss decreases monotonically toward the unique optimum. The first epoch that fails to improve it by `min_delta` (1e-4) means the descent has flattened *at* the optimum — it cannot start improving again, by the nature of the problem. There is no second basin to wait for, so larger patience would only burn epochs at a converged point. `patience: 1` stops exactly when ‖∇L‖ has effectively vanished. This is "run until convergence, don't count epochs" in operational form.
- `save_top_k: 1` on the same metric ⇒ the checkpoint is the best-converged state, consistent with the stopping rule.

The AdamW final reuses this exact rule. The LBFGS final relies on LBFGS's own `tolerance_grad`/`tolerance_change` convergence (full-batch, ≤10 epochs). Identical for both backbones.

---

## 6. Stage 1 — K-fold cross-validation (train/val): tuning weight decay

**Purpose**: choose the L2 weight decay. This is the *only* hyperparameter search, and it is done here on train/val folds, **independently per backbone**.

- Metadata = `kfold_tiles.parquet`; one fold held out as val (`exclude_folds=[val_fold]` for train, `include_folds=[val_fold]` for val).
- Split strategies: `stratified` and `stratified_group` (group = no slide leakage across folds).
- For each weight-decay candidate `{0, 1e-5, 1e-4, 1e-3, 1e-2}`, train every fold to convergence (same early-stopping rule), evaluate on the held-out fold. Selection = maximum mean validation `f1_macro` across folds; ties are broken by lower `f1_macro` std, then by the simpler/lower weight decay.

### Metric decision for final training

The metric used to decide which sweep point gets retrained before test is **mean validation `f1_macro` over the stratified-group folds**. The task is imbalanced and the hard classes (especially Connective-Tissue and Epithelium) matter, so `f1_macro` is the most appropriate model-selection metric: it gives each class equal influence and penalizes poor precision/recall balance. `acc_macro` is kept as a supporting metric, and validation loss is used diagnostically for convergence/overfitting, but neither decides the final weight decay.

This selection metric is separate from the final training checkpoint metric. In Stage 2 there is no validation fold left; the model is retrained on all training folds with the selected weight decay, and `train/loss_epoch` remains the convergence/checkpoint monitor. The held-out test split is used only after this decision is fixed.

### 6.1 Virchow2 sweep

Results (mean ± std over folds; full per-class tables in `docs/sweep_summary_{adamw,lbfgs}.csv`):

#### AdamW sweep

| weight_decay | val loss | acc_macro | f1_macro |
|---|---|---|---|
| 0     | 0.448 ± 0.396 | 0.8981 ± 0.039 | 0.8483 ± 0.051 |
| 1e-5  | 0.437 ± 0.375 | 0.8983 ± 0.039 | 0.8481 ± 0.053 |
| 1e-4  | 0.448 ± 0.398 | 0.9011 ± 0.037 | 0.8476 ± 0.056 |
| **1e-3**  | **0.436 ± 0.396** | **0.9006 ± 0.033** | **0.8486 ± 0.053** |
| 1e-2  | 0.410 ± 0.337 | 0.9007 ± 0.035 | 0.8465 ± 0.056 |

AdamW is essentially flat across weight decay (well-conditioned, low sensitivity). By the selected criterion, mean validation `f1_macro`, **wd = 1e-3** is chosen as the best point on a flat curve.

#### LBFGS sweep

| weight_decay | val loss | acc_macro | f1_macro |
|---|---|---|---|
| 0     | 22.241 ± 9.174 | 0.8396 ± 0.056 | 0.8222 ± 0.028 |
| 1e-5  | 0.734 ± 0.286  | 0.8721 ± 0.049 | 0.8356 ± 0.040 |
| 1e-4  | 0.449 ± 0.162  | 0.8898 ± 0.044 | 0.8468 ± 0.042 |
| 1e-3  | 0.295 ± 0.089  | 0.9070 ± 0.038 | 0.8550 ± 0.048 |
| **1e-2**  | **0.227 ± 0.076**  | **0.9208 ± 0.034** | **0.8556 ± 0.058** |

LBFGS is highly sensitive: with no regularization the exact solve diverges (loss ≈ 22, ill-posed). Regularization is mandatory and monotonically helps up to **wd = 1e-2** by the selected criterion, mean validation `f1_macro` (also best loss and acc_macro — driven mainly by the hard Connective-Tissue class: f1 0.52 → 0.68 across the sweep). This confirms the convexity prediction: the unregularized exact solution overfits the 2560-d space; L2 makes it well-posed.

**Chosen (Virchow2):** AdamW final `weight_decay = 1e-3`; LBFGS final `weight_decay = 1e-2`.

### 6.2 ProvGigaPath sweep

Same candidates, same folds, same early-stopping rule; ProvGigaPath embeddings (`embedding_run_id = 410c8672471348ceb4c58817f70fa097`, 1536-d). Selection metric = **mean validation `f1_macro` over folds**. Do not reuse Virchow2's selected values unless the sweep supports it.

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

Export full tables to `docs/sweep_summary_provgigapath_adamw.csv` and `docs/sweep_summary_provgigapath_lbfgs.csv`.

#### AdamW sweep

| weight_decay | val loss | acc_macro | f1_macro |
|---|---|---|---|
| 0     | 0.482 ± 0.356 | 0.8714 ± 0.033 | 0.8199 ± 0.034 |
| 1e-5  | 0.486 ± 0.351 | 0.8710 ± 0.034 | 0.8195 ± 0.034 |
| **1e-4**  | 0.500 ± 0.384 | 0.8712 ± 0.034 | **0.8200 ± 0.034** |
| 1e-3  | 0.496 ± 0.368 | 0.8706 ± 0.035 | 0.8199 ± 0.038 |
| 1e-2  | 0.496 ± 0.353 | 0.8700 ± 0.033 | 0.8188 ± 0.036 |

AdamW is nearly flat for ProvGigaPath, with only a 0.0012 spread in mean validation `f1_macro` across the whole grid. By the selected criterion, **wd = 1e-4** is the best point; it is effectively tied with `wd = 0` and `wd = 1e-3`, but has the highest rounded mean `f1_macro`.

#### LBFGS sweep

| weight_decay | val loss | acc_macro | f1_macro |
|---|---|---|---|
| 0     | 30.713 ± 21.176 | 0.7980 ± 0.045 | 0.7881 ± 0.025 |
| 1e-5  | 1.025 ± 0.427 | 0.8407 ± 0.038 | 0.8081 ± 0.029 |
| **1e-4**  | 0.574 ± 0.208 | 0.8649 ± 0.033 | **0.8171 ± 0.032** |
| 1e-3  | 0.387 ± 0.112 | 0.8831 ± 0.029 | **0.8171 ± 0.041** |
| 1e-2  | 0.337 ± 0.084 | 0.8941 ± 0.029 | 0.8089 ± 0.056 |

LBFGS again needs regularization: `wd = 0` gives a very large validation loss and weak `f1_macro`. Validation loss and `acc_macro` keep improving through `wd = 1e-2`, but the selection metric peaks earlier. `wd = 1e-4` and `wd = 1e-3` are tied in rounded mean validation `f1_macro`; **wd = 1e-4** is chosen by the lower `f1_macro` std tie-breaker.

**Chosen (ProvGigaPath):** AdamW final `weight_decay = 1e-4`; LBFGS final `weight_decay = 1e-4`.

---

## 7. Stage 2 — Final training

Retrain on **all** training folds (no held-out fold), with the weight decay chosen for that backbone in Stage 1.

### Virchow2 final

| | AdamW final | LBFGS final |
|---|---|---|
| Config | `linear_classifier_final_adamw` | `linear_classifier_final_lbfgs` |
| `weight_decay` | 1e-3 | 1e-2 |
| `learning_rate` | 1e-4 | 1.0 |
| Trainer | `trainer/default` (early stop, max 500) | `max_epochs` 10, full-batch |
| Stop rule | train/loss_epoch, patience 1 | LBFGS tolerances |
| Checkpoint | `last` (converged) | `last` |

### ProvGigaPath final

```bash
uv run python -m ml \
  +experiment=ml/linear_classifier_final_adamw_provgigapath \
  model.weight_decay=1e-4

uv run python -m ml \
  +experiment=ml/linear_classifier_final_lbfgs_provgigapath \
  model.weight_decay=1e-4
```

Same trainer / stop rule / checkpoint convention as Virchow2. Checkpoint path convention:

```text
mlflow-artifacts:/104/<run_id>/artifacts/checkpoints/last/checkpoint.ckpt
```

Both optimizers reach essentially the same convex optimum per backbone; LBFGS gets there faster. Validation here is a sanity check on the fully-trained model — tuning already happened in Stage 1.

---

## 8. Stage 3 — Test

Evaluate the chosen final checkpoint on the **held-out test split** (`test_tiles.parquet`), untouched by Stages 1–2. Unbiased generalization estimate only — no fitting, no selection. The ground-truth test configs compute metrics and a prediction parquet only; they do not write TIFF prediction maps.

### 8.1 Virchow2

#### AdamW final — test metrics

| Metric | Value |
|---|---|
| acc_macro | 0.9307 |
| f1_macro | 0.8735 |
| slide_acc_mean | 0.9119 |
| slide_acc_median | 0.9764 |
| slide_acc_min | 0.2484 |

Per-class (acc / f1): Blood 0.992/0.977, Connective-Tissue 0.858/0.653, Epithelium 0.859/0.738, Fat 0.911/0.887, Muscle 0.968/0.931, Nerve 0.972/0.954, Other 0.955/0.975.

#### LBFGS final — test metrics

_Pending — fill from the LBFGS test run (same checkpoint convention)._

### 8.2 ProvGigaPath

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

#### AdamW final — test metrics

_Pending — fill from the ProvGigaPath AdamW test run._

#### LBFGS final — test metrics

_Pending — fill from the ProvGigaPath LBFGS test run._

### 8.3 Backbone comparison

_Pending — compare Virchow2 vs ProvGigaPath on the held-out test (acc_macro, f1_macro, per-class f1, slide-level accuracy), once both backbones have completed Stages 1–3. Comparison is fair by construction: identical labeled tiles, folds, thresholds, filtering, selection metric, and checkpoint convention; only the frozen feature space differs._

---

## 9. Class mapping convention

Classes follow `class_indices` (`collapse_alterations_to_other`), 7 classes, identical ordering across all three stages, both backbones, and all downstream consumers (metrics, exported masks, reporting). The class index space is fixed at training time and must not drift.

---

## Summary of the rationale

| Decision | Reason |
|---|---|
| Linear + softmax + CE | multinomial logistic regression — convex (both backbones) |
| Don't tune/report epochs | convex ⇒ single optimum; epochs meaningless |
| Watch ‖∇L‖ / loss plateau | true convergence signal |
| LBFGS for final | convex ⇒ exact full-batch solve, fast |
| Weight decay = only swept hyperparameter | regularization mandatory (LBFGS diverges at wd=0) |
| Select wd by mean validation `f1_macro` | class-balanced model selection before the held-out test |
| Tune wd on k-fold train/val | unbiased regularization selection, no slide leakage |
| Sweep wd per backbone independently | different feature dim/conditioning ⇒ different optimal regularization |
| AdamW wd 1e-3 / LBFGS wd 1e-2 (Virchow2) | best sweep points (AdamW flat, LBFGS monotone) |
| ProvGigaPath wd | AdamW wd 1e-4 / LBFGS wd 1e-4 by mean validation f1_macro |
| Same tiles/folds/thresholds/metric/checkpoint across backbones | fair backbone comparison; only feature space differs |
| `patience: 1` on train loss | convex loss cannot re-improve after plateau |
| Separate held-out test | unbiased generalization, report only |
