# H-Optimus-0 vs Virchow2 + Graz inference — tracker

Last updated: 2026-11-23

## Goal

1. Add **H-Optimus-0** (`bioptimus/H-optimus-0`, Apache-2.0, commercially usable) as a
   foundation model and compare it with **Virchow2** (CC-BY-NC-ND, cannot be deployed
   commercially) on our held-out test set.
2. Prepare a **standalone repo/CLI** that the supervisor runs in Graz on **15 slides with
   GT**: inference + evaluation for both models.
3. Later (not now): if H-Optimus-0 is good, Graz deploys it to **HALO** (Indica Labs) for
   routine use by pathologists.

## Status

Need: **must** = required for the Graz test next week · **should** = needed for a
trustworthy comparison / thesis, not blocking Graz · **nice** = optional.

| # | Task | Need | Status | Notes |
|---|------|------|--------|-------|
| 1 | Local GPU embedding backend (`backend: local`) | must | ✅ done | `preprocessing/foundation_models.py`, `preprocessing/embeddings.py`, commit `b226ffe` on `feature/hoptimus-graz-inference` |
| 2 | HF token + gated access (Virchow2, H-Optimus-0) | must | ✅ done | token in `/mnt/projects/tissue_classification/hf_cache/token` (`HF_HOME`), both repos `auth_check` OK |
| 3 | Parity: local Virchow2 vs model service | must | ✅ PASS | run `49421ae4584147a7bd1f5804037f7a27` vs `6dbdc6e5…`; 4917 tiles, cosine mean 0.999999 / min 0.999989, rel. L2 mean 0.12 %. Parity script/config removed after the check (in git history, commit `b226ffe`) |
| 4 | H-Optimus-0 embeddings (train + test) | must | ✅ done | `scripts/submit_embeddings.py` with `gpu="A40"`, `HF_HOME`, 16 CPU (one-off submit script removed, in git history). Run ID: `ec9904fdcaa5494babfe0bfa4f348d52`. Rows match `filter_tiles` exactly (train 1 570 445, test 481 806), no NaN/inf, dim 1536. Runtime 7.6 h (train 95 tiles/s, test 44 tiles/s) |
| 5 | Add `embedding_run_ids.hoptimus0` to `configs/data/dataset.yaml` | must | ✅ done | `hoptimus0: ec9904fd…` |
| 6 | H0 ML configs, LBFGS (train/final/test) | must | ✅ done | `configs/experiment/ml/*hoptimus0_lbfgs*`; wd 1e-3, test → final `9b7a1269…` |
| 6b | H0 ML configs, AdamW (train/final/test) | nice | final ✅ / test todo | final `4a5ffaf1…` (wd 1e-3 reused from LBFGS, no AdamW sweep); test `+experiment=ml/test_linear_hoptimus0_adamw` only for a complete table |
| 7 | Group k-fold sweep for H0 (LBFGS) | must | ✅ done | best wd 1e-3 (mean val F1-macro 0.876 ± 0.047); same grid as Virchow2: `val_fold=0..4`, `weight_decay=0,1e-5,1e-4,1e-3,1e-2` (+1e-1 if best is at the edge); `+experiment=ml/train_linear_hoptimus0_lbfgs_group_kfold` |
| 8 | Final + test for H0 (LBFGS) | must | ✅ done | test `10d443de…` F1-macro 0.896; final LBFGS `9b7a1269c65c4e9780fcad655088be14`, AdamW `4a5ffaf1b4c74d6b872620d1f3f3bf6d` (both wd 1e-3; AdamW reuses the LBFGS wd); submit with `+experiment=ml/final_linear_hoptimus0_lbfgs`, then test with `final_train_run_id=<run>` |
| 9 | Comparison Virchow2 vs H0 on our test set | must | ✅ done | V2 test `d604fb19…` (head `a0ec1995`) 0.893 vs H0 0.896; paired per-slide acc H0−V2 −0.005, 95 % CI [−0.022, +0.010] → tie (see log). Tile-level paired bootstrap would need test-mode prediction parquet (not logged yet) |
| 9b | + ProvGigaPath in the comparison | nice | partial | May PGP checkpoints: 0.818 / 0.829; Nov finals `03e0b84b` / `f0ea2068` untested |
| 10 | Export linear heads (ckpt → safetensors + json) | must | ✅ done | `python -m inference.export_head`; Virchow2 head = LBFGS final run `a0ec1995d1b1472785d30ccbe15e84ef` (same as test/predict configs); H0 head = LBFGS final run `9b7a1269c65c4e9780fcad655088be14` (wd 1e-3); must run where MLflow is reachable |
| 11 | Graz CLI: inference (slides → mask → tiles → embed → classify → maps) | must | ✅ implemented | `python -m inference predict`, see `inference/README.md`; smoke-tested end-to-end on a synthetic pyramid (dummy model/head, stubbed tissue mask) — not yet on a real slide |
| 11b | Graz CLI: evaluation vs GT + HTML report | must | ✅ implemented | `inference evaluate` / `inference labels`; GT via shared parser (`preprocessing/annotations.py`) + shared training label rule (`preprocessing/_labels.assign_tile_labels`, verified identical to old loader logic); metrics, merged Epithelium+Other variant, slide bootstrap, GT/model panels, HTML report |
| 11c | Graz CLI: GeoJSON export of predictions | should | todo | for QuPath review; HALO later |
| 12 | GT class mapping Graz → our classes | must | provisional | `inference/label_maps/graz.yaml`; TODO confirm Lymphatic, Abnormal-Epithelium, Blood-Plasma, Tubular-/Villous-glands |
| 13 | Dry run of Graz CLI on our test slides | must | todo | e-INFRA / MUSICA |
| 14 | Torch build that runs on V100 (sm_70) | must | todo | verify; likely CUDA 12.6 wheels |
| 15 | Send Graz questions (nvidia-smi, internet, slide format, exhaustive GT) | must | todo | draft in chat |
| 16 | Merge `chore/update-final-checkpoints` before/with this branch | should | todo | this branch is based on it (4 commits not in `master`) |

## Known limitation: effective resolution (decided 2026-11-23)

`closest_level(0.5)` picks level 0 on 0.25 mpp SVS with a 1/4/16 pyramid, so the
"0.5 mpp" pipeline effectively tiles at **0.25 mpp** for TCGA: 93.6 % of train tiles,
96.1 % of test tiles (MUG 0.485, MOU 0.486, FNBrno 0.44 are the rest).
Graz SVS (levels 0.25 / 1.0) are read the same way → consistent with the trained heads.

**Decision:** keep as is (Virchow2 and ProvGigaPath perform well). Revisit only if
H-Optimus-0 (pretrained at 0.5 mpp) underperforms. **Checked 2026-11-24: it does not** (test tie; per-slide H0−V2 TCGA +0.002, non-TCGA −0.013), so closed — first check: V2 vs H0 test metrics
split by resolution group (TCGA@0.25 vs rest@~0.5).

## Fair-comparison invariants

Only the foundation model changes. Same `filter_tiles_run_id` tiles, same
`stratified_group_kfold_run_id` folds, same test split, same tile-label thresholds,
same sweep grid.

## Graz environment (known so far)

- Local server in Graz (not MUSICA), **1× V100 32 GB**.
- Unknown: OS / CUDA driver version, internet access (PyPI, gitlab.ics.muni.cz, HuggingFace),
  container support, slide format/scanner.
- GT annotations: **GeoJSON**, more detailed classes than ours → need a mapping.
- HALO: postponed; first test on the 15 slides.

### V100-specific risks

- V100 = compute capability 7.0 (Volta). Recent PyTorch CUDA 12.8+ wheels may have dropped
  Volta kernels → probably need a CUDA 12.6 build of torch on that machine. **Verify.**
- No bf16 on V100 → keep fp16 autocast (already the default on CUDA).
- 32 GB is enough for ViT-g inference in fp16 (weights ~2.2 GB); batch size may need tuning.

## Graz CLI — design (agreed)

```
manifest.csv ─► tissue-mask ─► tile ─► embed[model] ─► classify[model] ─► maps + evaluate ─► compare
```

- Input: manifest CSV `slide_path, annotation_path, slide_id`.
- No Ray / MLflow / kube_jobs; one process, one GPU, DataLoader over openslide.
- Embeddings cached per slide, resumable; offline weights via `HF_HOME` + `HF_HUB_OFFLINE=1`.
- Reuse `preprocessing/foundation_models.py` (verified by parity test).
- Tissue mask: same filter as training (`rationai-masks` @ branch `feature/modular-tissue-mask`, which has
  the Vips filters; Otsu + closing 50 + opening 5 at 2 mpp). Branch pin is intentional.
- GT → tile labels must follow training rules (per-class coverage thresholds, `tissue_prop_min: 0.2`);
  unmapped Graz labels are ignored, never silently counted as Other.
- Outputs per slide: class-map TIFF, tiles parquet (`x, y, probs, pred, gt`), PNG overlay
  (thumbnail + class map, GT / Virchow2 / H0 side by side), GeoJSON.
  Plus an HTML report with metrics (per-class F1, macro F1, confusion matrix, per-slide) and `run.json`.
- No xOpat in Graz → PNG + HTML for quick review, GeoJSON for QuPath (and later HALO).

## Open questions

- [ ] Graz: OS / NVIDIA driver / CUDA version on the V100 server (`nvidia-smi`).
- [ ] Graz: internet access from the server (PyPI, GitLab, HuggingFace)? Docker/Apptainer?
- [ ] Graz: slide format + scanner, is mpp metadata correct?
- [ ] Graz: full list of GT label names and the agreed mapping to our 7 classes.
- [ ] Graz: are annotations exhaustive or only ROIs?
- [ ] Graz: can we get one slide + GT to test the CLI before next week?
- [x] H0 embedding throughput on A40 → total runtime: 7.6 h for 2.05 M tiles (~75 tiles/s overall).
- [ ] Graz: path to the two sample GeoJSON annotation files (received, not yet in the repo).

## Log

- 2026-11-24: Code-review cleanup: shared `ml/prediction_maps.py` (class-vote map) and `preprocessing/_tissue_filter.py`, tiling constants in `inference/slides.py` (head JSON validated against them, hidden `--mpp/--tile-size/--stride` removed), `evaluate()` split; submit scripts back to master templates. Verified on synthetic slides: inference outputs identical before/after.
- 2026-11-23: Supervisor request (H-Optimus-0 vs Virchow2, 15 Graz slides, later HALO).
- 2026-11-23: H-Optimus-0 not on model service → local GPU backend implemented and pushed.
- 2026-11-23: Parity test PASS (cosine ≈ 1.0).
- 2026-11-24: H0 head exported (`inference/heads/h-optimus-0.*`, run `9b7a1269…`); Graz can now run `predict --models virchow2 h-optimus-0`.
- 2026-11-24: V2 LBFGS test of the Graz head `a0ec1995` → run `d604fb19…`: F1-macro 0.893, acc-macro 0.955, slide-acc mean 0.950. **Fair H0 vs V2 (same embeddings protocol, same tiles):** F1-macro 0.896 vs 0.893; paired per-slide tile accuracy H0−V2 −0.005, 95 % CI [−0.022, +0.010], H0 better on 12/35 slides, worse on 18 (TCGA +0.002, non-TCGA −0.013) → tie. H0 better on Connective-Tissue (0.805 vs 0.746), worse on Blood (0.949 vs 0.972).
- 2026-11-24: H0 LBFGS test `10d443de…`: F1-macro 0.896, acc-macro 0.958, slide-acc mean 0.946. Interim V2 counterpart `8dcd1d41…` (0.893) used the **May** checkpoint `0e2230c7` (embeddings `c325e3a5`, old k-fold), not `a0ec1995`. Paired per-slide tile accuracy H0−V2: +0.000, 95 % CI [−0.020, +0.018] (35 slides; TCGA +0.008, non-TCGA −0.011). **The Nov finals (`a0ec1995`, `2a52372d`, `03e0b84b`, `f0ea2068`) have no test runs yet**, including the Graz Virchow2 head. Test mode logs no tile predictions, so a tile-level paired bootstrap is not possible yet.
- 2026-11-24: H0 final runs done: LBFGS `9b7a1269…`, AdamW `4a5ffaf1…`; ids set in the test configs.
- 2026-11-24: H0 LBFGS sweep done (25/25). Mean val F1-macro by wd: 0=0.835, 1e-5=0.856, 1e-4=0.869, **1e-3=0.876**, 1e-2=0.872 → final wd 1e-3. Virchow2 (same embeddings `6dbdc6e5`) best 0.877 @ 1e-4; paired per-fold diff H0−V2 = −0.0007 (tie). ProvGigaPath 0.833. Note: V2 final `a0ec1995` uses wd 1e-2, picked from the old sweep on embeddings `c325e3a5`; the current V2 sweep prefers 1e-4.
- 2026-11-23: H-Optimus-0 embeddings finished (run `ec9904fd…`); added to `dataset.yaml`, LBFGS sweep ready to submit.
- 2026-11-23: H-Optimus-0 embedding job submitted.
- 2026-11-23: Graz: GPU is V100 32 GB; sample GT GeoJSONs received, classes more detailed than ours.
- 2026-11-23: Graz sample GT analysed: QuPath GeoJSON (`classification.name`), layered (coarse gland/no-gland + detailed); 40–67 % of annotated area maps to our classes, usable tiles 77–96 % Other; Fat exists in other slides (label name TBD).
- 2026-11-23: `inference embed` added so Graz can compute embeddings (both models) before heads exist.
- 2026-11-23: Inference CLI (`inference/`) implemented: tissue mask, tiles, embeddings (cached), head, class-map TIFF, overlay PNG, summary.csv, run.json.
