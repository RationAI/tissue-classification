# Tissue classification — standalone inference

Runs the tissue classifier (frozen foundation model + linear head) on whole-slide
images. No cluster tooling, MLflow or Ray is needed: one machine, one GPU.

```
slides ─► tissue mask ─► 224 px tiles @ 0.5 mpp ─► foundation model ─► linear head ─► maps
```

Classes: Nerve, Blood, Connective-Tissue, Fat, Epithelium, Muscle, Other.

Available models (heads in `inference/heads/`):

| model | foundation model | head trained on |
|---|---|---|
| `virchow2` | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) | ✅ final LBFGS head (`heads/virchow2.json`) |
| `h-optimus-0` | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) | coming soon (embeddings can already be computed) |

## 1. Setup

Requirements:

- Linux, NVIDIA GPU (tested target: V100 32 GB), Python 3.12
- System libraries: **OpenSlide** and **libvips built with OpenSlide support**
  (Debian/Ubuntu: `apt install openslide-tools libvips-dev`; conda:
  `conda install -c conda-forge openslide libvips`)
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
git clone -b feature/hoptimus-graz-inference https://github.com/RationAI/tissue-classification.git
cd tissue-classification
uv sync
```

**V100 / older GPUs:** check that the installed PyTorch has kernels for the GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); print(torch.ones(2, device='cuda') @ torch.ones(2, device='cuda'))"
```

If `sm_70` is missing from the list or the last command fails with *no kernel image is
available*, install a CUDA 12.6 build of the same torch version:

```bash
uv pip install --reinstall torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
# then always run with: uv run --no-sync ...
```

## 2. Hugging Face access (model weights)

Both foundation models are gated. With your own Hugging Face account:

1. Accept the terms on the model page(s) you will run (links above).
2. Create a **read** token at https://huggingface.co/settings/tokens.
3. `uv run hf auth login` and paste the token.

Weights are downloaded once (~2.5 GB Virchow2, ~4.5 GB H-optimus-0) to
`~/.cache/huggingface` (override with `HF_HOME`). Without internet on the GPU machine:
download them on a machine with access, copy the `HF_HOME` directory over and set
`HF_HUB_OFFLINE=1`.

## 3. Run

Two steps. The expensive one (embeddings) does not need the classification heads, so it
can run before the heads are released:

```bash
# 3a. tissue masks, tiles and embeddings (GPU, the long part)
uv run python -m inference embed \
    --slides /path/to/slides/ \
    --out results/graz \
    --models virchow2 h-optimus-0

# 3b. classification + maps (seconds per slide; reuses everything cached by 3a)
uv run python -m inference predict \
    --slides /path/to/slides/ \
    --out results/graz \
    --models virchow2
```

Always use the same `--slides` and `--out` for both steps — `predict` picks up the
cached tiles and embeddings from `--out`. Running `predict` directly also works; it
embeds whatever is not cached yet.

- `--slides`: slide files and/or directories (`.svs .tif .tiff .ndpi .mrxs .scn ...`).
  Slides must have microns-per-pixel metadata.
- `--models virchow2 h-optimus-0`: several models in one run; tissue masks and tiles are
  computed once and shared. Each model needs its Hugging Face access (section 2).
- `--batch-size` (default 128) — lower it if the GPU runs out of memory.
- `--num-workers` (default 8) — processes reading tiles; raise it if the GPU is idle.
- Interrupted? Re-run the same command: finished slides are skipped (cached).
- Quick test first: point `--slides` at a single slide file and check the outputs.
- Long runs: start inside `tmux`/`screen` or with `nohup ... > embed.log 2>&1 &`.

## 4. Outputs

```
results/graz/
    run_embed.json, run_predict.json environment, versions, per-slide info, timings, errors
    summary_predict.csv              per slide and model: tile count, predicted class fractions
    <slide>/
        tissue_mask.png              tissue mask used to select tiles (QC)
        tiles.parquet                tile coordinates (on the 0.5 mpp level) + tissue coverage
        embeddings_<model>.npy       cached embeddings
        predictions_<model>.parquet  x, y, prob_<class>..., pred, pred_name
        pred_<model>.tiff            class map (BigTIFF at 0.5 mpp, values = round(255*(i+1)/7))
        overlay_<model>.png          thumbnail next to the colour-coded prediction
```

Start with `overlay_<model>.png` for a quick visual check and `run_*.json` if a slide failed.

## Notes

- Tiles: 224 px at the pyramid level closest to 0.5 mpp, stride 112; overlapping tile
  predictions are fused by summed probabilities — identical to how the model was trained
  and evaluated.
- A warning is printed if the closest level is far from 0.5 mpp (e.g. missing pyramid
  levels); predictions on such slides are not reliable.
