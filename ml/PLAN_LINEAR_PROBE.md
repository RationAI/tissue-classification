# Linear-Probe Training: Implementation Plan

Scope of the first PR: **train + k-fold validation only**. Test-set evaluation is a follow-up PR (separate entrypoint, no fold loop, possibly slide-level aggregation). Keeping test out of this PR keeps the held-out set untouched while we tune the probe.

## 0. Current state (what already exists)

- `ml/train.py` — Lightning entrypoint, `mode={fit,test}`, instantiates `data` / `model` / `trainer` from Hydra.
- `ml/data/embeddings_datamodule.py` — `EmbeddingsDataModule` that loads `train_dir` / `test_dir` parquet via `datasets.load_dataset`, filters by `fold`, maps `label` → idx.
- `ml/models/linear_probe.py` — `nn.Linear` head, CE loss, accuracy + macro-F1 (`torchmetrics`).
- `configs/ml/linear_probe.yaml` — wires the above; assumes a single parquet dir per split with `embedding`, `label`, `fold`, `tissue_prop` already joined.
- `configs/class_mapping/collapse_alterations_to_other.yaml` — **the mapping we will use for training**: 7 classes (Nerve, Blood, Connective-Tissue, Fat, Epithelium, Muscle, Other). Inflammation/necrosis/neoplastic alterations are collapsed into `Other`. `standard.yaml` is the alternate 9-class mapping; not used here.

## 1. Gaps to close before this works end-to-end

### 1.1 Embeddings and labels are not in the same parquet
`embed.py` writes `train/tiles/*.parquet` with columns `slide_id, x, y, embedding`. `kfold_split.py` writes one `kfold_tiles.parquet` with columns `slide_id, x, y, label, tissue_prop, fold` (+ `roi_coverage_*`). The datamodule today assumes everything is in one file. → must **join on `(slide_id, x, y)`**.

**Recommendation:** do the join lazily in `setup()` using `pyarrow` / `duckdb` over the parquet files (no extra preprocessing script, no extra MLflow run). The labels parquet is small enough to fit in RAM; embeddings stay memory-mapped.

### 1.2 Inputs come from MLflow artifacts, not local disk
Current config hardcodes `${project_path}/embeddings/${embeddings_run_id}/...`. The other scripts (`embed.py`, `kfold_split.py`) consistently use `mlflow.artifacts.download_artifacts(run_id=..., artifact_path=...)`. → datamodule should accept `embeddings_run_id` + `kfold_run_id` and download to a cache dir on `prepare_data()` (single-process hook).

### 1.3 Raw labels in kfold parquet are not the 9 canonical classes
`kfold_split.py` writes `label = roi_coverage_<NAME>` argmax → values like `"EPITHELIUM-BB"`, `"NEOPLASTIC-MALIGNANT"`, or `"background"`. The probe expects canonical names (`Epithelium`, `Neoplastic`, …). → apply `class_mapping` (raw → canonical) inside the datamodule. Tiles whose raw label isn't covered by the mapping (today: `"background"`) need a policy — see §1.4.

### 1.4 Background and coverage-threshold filtering
**Background**: `collapse_alterations_to_other.yaml` has no Background class. `filter_tiles.py` already drops tiles with zero tissue coverage and zero annotation coverage upstream — so by the time we reach the kfold parquet, `"background"` rows can still appear (a tile can have tissue but no annotation overlap, or vice-versa, depending on the filter logic). Drop any rows whose raw label isn't in the mapping. Config knob: `drop_unmapped: true` (default true).

**Coverage thresholds (live in the datamodule, not a separate PR/script)**: the upstream filter is the coarse cleaning step (any tissue, any annotation). For training-time experimentation, expose two filters as datamodule knobs and apply them after the join, before the train/val split:

- `tissue_prop_min: float = 0.0` — drop tiles whose total annotation coverage `tissue_prop` is below the threshold. This already exists as a field on the datamodule; keep it.
- `class_coverage_min: float = 0.0` — drop tiles whose **dominant class coverage** (i.e. the `roi_coverage_*` value backing the assigned label, after collapsing per the class mapping) is below the threshold. Forces the label to be "confident" — useful when many tiles are mosaics.

Both are pure row masks on the labels DataFrame, cheap, and get logged as MLflow params with the run, so threshold sweeps show up cleanly. Rationale for not making this a separate preprocessing PR: these thresholds are experimental knobs you'll sweep alongside LR / weight decay; locking them into a parquet artifact would force a re-preprocessing run for every variant. The fundamental cleaning (any-tissue, any-annotation) stays where it belongs in `filter_tiles.py`.

To support `class_coverage_min` after class collapsing, the datamodule needs the per-class collapsed coverage. Compute it in `setup()`: for each canonical class C, sum the `roi_coverage_<raw>` columns whose raw label maps to C. Then the dominant-class coverage for a tile is `max_C(collapsed_coverage_C)`. The kfold parquet already carries `roi_coverage_*` columns, so no new artifact is needed.

### 1.5 Config bugs
- `configs/ml/linear_probe.yaml:30` uses `class_mapping.class_names`, which doesn't exist in `standard.yaml`. Either (a) add a `class_names` list to the class-mapping yaml derived from the dict keys, or (b) change the reference. Pick (a) — most readable.
- `${len:...}` resolver — verify it's registered (rationai.mlkit likely does, but confirm by running once).

### 1.6 K-fold orchestration
User wants **all folds in one MLflow run**. Today `train.py` runs a single fold (`val_fold` param). → wrap fit in a loop over folds, log per-fold metrics under `fold_{i}/...` and write aggregate (`val/acc_mean`, `val/acc_std`, `val/f1_macro_mean`, …) at the end.

### 1.7 `trainer.test()` after fit
`train.py:21` calls `trainer.test(...)` after `trainer.fit(...)`. Remove for this PR (no test in this PR). Re-introduce in the test-PR, in a separate `mode=test` path that doesn't loop folds.

---

## 2. Concrete step-by-step plan

### Step 1 — Fix `configs/class_mapping/collapse_alterations_to_other.yaml`
Add a derived `class_names` list (so configs that reference `class_mapping.class_names` work) and switch `linear_probe.yaml` to default to this mapping:
```yaml
class_names:
  - Nerve
  - Blood
  - Connective-Tissue
  - Fat
  - Epithelium
  - Muscle
  - Other
```
Apply the same change to `standard.yaml` for consistency, but `linear_probe.yaml`'s `defaults` should point at `collapse_alterations_to_other`. Keep `class_mapping` (canonical→raw list) and `class_indices` as they are.

### Step 2 — Rewrite `EmbeddingsDataModule`
Responsibilities, in order:

1. **`prepare_data()`** (single-process):
   - Download embeddings artifact: `mlflow.artifacts.download_artifacts(run_id=embeddings_run_id, artifact_path="train")`. Cache path on `self`.
   - Download kfold artifact: `mlflow.artifacts.download_artifacts(run_id=kfold_run_id, artifact_path="<kfold_artifact_path>/kfold_tiles.parquet")`.
2. **`setup(stage)`**:
   - Read kfold parquet into pandas (small: ~few M rows × handful of cols). **Keep** `roi_coverage_*` columns until thresholds are applied.
   - Build a `raw → canonical` lookup from the config's `class_mapping` (dict of canonical → list[raw]) and apply it to the `label` column.
   - Drop rows whose raw label isn't in the mapping (handles `"background"` and any stragglers) — gated by `drop_unmapped: true`.
   - Compute per-tile **collapsed coverage**: for each canonical class C, sum `roi_coverage_<raw>` over its raw members. Add a `dominant_coverage` column = the collapsed coverage of the assigned canonical label.
   - Apply `tissue_prop_min` and `class_coverage_min` row masks. Log row-count deltas at each step (initial → after raw-label drop → after `tissue_prop_min` → after `class_coverage_min`) as MLflow metrics so threshold sweeps are interpretable.
   - Drop `roi_coverage_*` columns once thresholds are done.
   - Load embeddings as an Arrow table: `pyarrow.dataset.dataset(emb_dir, format="parquet").to_table(columns=["slide_id","x","y","embedding"])`. Memory-mapped, zero-copy.
   - **Join** on `(slide_id, x, y)` via `pyarrow.Table.join(labels_table, keys=["slide_id","x","y"], join_type="inner")`. The two parquets share this key by construction (both downstream of `filter_tiles/train_tiles.parquet`, neither remaps coords), so the inner-join is effectively 1:1. Use `pyarrow.Table.join` rather than `pandas.merge` — the embedding column is heavy (~2560 × 4B × N) and we want to avoid copies. Wrap the joined Arrow table back into `datasets.Dataset(arrow_table=...)`.
   - **Verify the join**: log `n_embeddings`, `n_labels`, `n_joined` as MLflow metrics. If `n_joined < n_labels`, log a warning with the gap — it means the embed run dropped tiles (e.g., upstream API failures past retries) and you'll want to know.
   - Map `label` → `y` (int) using `class_indices` (or `class_names.index(label)`).
   - For the configured `val_fold`, split into `train_set` (`fold != val_fold`) and `val_set` (`fold == val_fold`). `with_format("torch", columns=["embedding", "y"])`.
3. **`train_dataloader` / `val_dataloader`**: as today. Drop `test_dataloader` / `test_dir` arg in this PR (or leave the arg optional `test_dir: Optional[str] = None` with a `NotImplementedError` if requested — cleaner to just remove until the test PR adds it).
4. Make `val_fold` a settable attribute (not just hparam) so the train script can rebuild the data split per fold without reloading the parquet:
   - Cache the joined `full_dataset` on the datamodule.
   - Expose a `set_val_fold(fold: int)` that re-derives `train_set` / `val_set` by filtering — this avoids re-downloading and re-joining N times.

### Step 3 — K-fold loop in `ml/train.py`
Refactor `main()` for `mode == "fit"`:
```python
datamodule = instantiate(config.data)
datamodule.prepare_data()
datamodule.setup("fit")  # builds full_dataset once

per_fold_metrics: list[dict] = []
for fold in range(config.n_folds):
    pl.seed_everything(config.seed + fold)  # fresh init per fold
    datamodule.set_val_fold(fold)
    model = instantiate(config.model)
    trainer = instantiate(config.trainer, logger=logger)
    trainer.fit(model, datamodule=datamodule)
    # collect last-epoch val metrics
    per_fold_metrics.append({k: float(v) for k, v in trainer.callback_metrics.items()
                             if k.startswith("val/")})
    # log per-fold
    for k, v in per_fold_metrics[-1].items():
        mlflow.log_metric(f"fold_{fold}/{k}", v)

# aggregate
import numpy as np
keys = per_fold_metrics[0].keys()
for k in keys:
    vals = np.array([m[k] for m in per_fold_metrics])
    mlflow.log_metric(f"{k}_mean", vals.mean())
    mlflow.log_metric(f"{k}_std", vals.std())
```
Note: `trainer.test()` removed. Keep `mode == "test"` as `raise NotImplementedError("Test mode arrives in the test-set PR")` for now — clearer than silently breaking.

### Step 4 — Update `configs/ml/linear_probe.yaml`
- Replace `embeddings_run_id` with two run-id fields:
  ```yaml
  embeddings_run_id: ???   # e.g. f05076dcd5e64cb2839efe5fb20a22ae
  kfold_run_id: ???        # e.g. 2e81b0597b614ba8b675e3b34528c1df
  ```
- Add `n_folds: 5` (or wire to read from kfold artifact metadata if available).
- Drop `test_dir` from `data:` block.
- Update `data:` to:
  ```yaml
  data:
    _target_: ml.data.embeddings_datamodule.EmbeddingsDataModule
    embeddings_run_id: ${embeddings_run_id}
    kfold_run_id: ${kfold_run_id}
    kfold_artifact_path: kfold_split/kfold_tiles.parquet  # confirm against logged path
    class_mapping: ${class_mapping.class_mapping}
    class_indices: ${class_indices}
    drop_unmapped: true
    tissue_prop_min: 0.0      # threshold sweep knob
    class_coverage_min: 0.0   # threshold sweep knob
    batch_size: 1024
    num_workers: 4
  ```
- The `defaults` block should select `collapse_alterations_to_other` for `class_mapping`.
- Remove the `val_fold` field at the global level — folds are now driven by the loop.

### Step 5 — Strengthen `LinearProbe`
Small additions, low risk:
- Per-class F1 logged at `validation_epoch_end`: `F1Score(..., average=None)` → log each class as `val/f1_<class_name>`.
- Confusion matrix (`torchmetrics.ConfusionMatrix`) computed at end of validation, logged as an MLflow figure or table per fold.
- Optional embedding L2-normalization toggle (`normalize_input: bool`) — Virchow2 outputs are typically not L2-normalized at the CLS-token stage; making this a flag is one line and a common probe variant to try.
- Optional class weights for CE (defer wiring, but leave a `class_weights: Optional[list[float]] = None` parameter).

### Step 6 — Logging hygiene
- Log artifacts: the resolved class list, the join coverage stats (`#tiles in embeddings`, `#tiles in kfold`, `#joined`, `#dropped_background`, `#dropped_no_label`).
- Log a one-row summary table per fold: `n_train`, `n_val`, label distribution.
- Set `metadata.run_name` to include both run-ids: `"Linear probe (embed=${embeddings_run_id[:8]}, kfold=${kfold_run_id[:8]})"`.

### Step 7 — Smoke run
End-to-end against the existing artifacts:
```
embeddings_run_id=f05076dcd5e64cb2839efe5fb20a22ae
kfold_run_id=2e81b0597b614ba8b675e3b34528c1df
embed_dim=<virchow2 dim, confirm — likely 2560 for Virchow2>
n_folds=5  # confirm against the kfold run's params
```
Run for 1–2 epochs first to confirm wiring, then full `max_epochs=30`.

---

## 3. Resolved decisions

1. **Virchow2 embedding dimension** — 2560.
2. **Kfold artifact path** — `kfold_split/kfold_tiles.parquet`.
3. **n_folds** — 5.
4. **Validation cadence** — fit one fold, then move on (sequential).
5. **Reproducibility of `set_val_fold`** — confirmed: re-instantiate model + seed-per-fold (`seed + fold`).

---

## 4. Out of scope for this PR (next PR)

- Test-set evaluation (single pass, no folds, possibly with slide-level aggregation).
- Fine-tuning beyond the linear head.
- Class-weighted CE / focal loss / soft-label CE on `roi_coverage` proportions.
- Model selection across folds (best ckpt per fold, ensemble at test time).
- Multi-GPU / DDP — single GPU is plenty for a linear probe on cached embeddings.
