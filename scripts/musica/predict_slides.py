"""Standalone whole-slide prediction for a handful of unlabeled slides.

Runs the full inference chain for each slide in a single process, without the
MLflow run chaining of the main pipeline:

    tissue mask -> grid tiles -> tissue coverage filter -> Virchow2 embeddings
    -> final linear classifier -> prediction TIFFs + PNG overlay

Every step mirrors the corresponding pipeline stage so the outputs match what
``predict_linear_virchow2_lbfgs_tissue_tiles`` would produce:

- tissue mask: ``preprocessing/tissue_masks.py`` (closest level to 2.0 MPP)
- tiling: ``preprocessing/tiling.py`` via ``ratiopath`` (closest level to
  0.5 MPP, no resampling, 224 px tiles, 112 px stride)
- coverage: ``preprocessing/coverage_stats.py`` (``tile_tissue_coverage > 0``)
- embeddings: RationAI model-service Virchow2 deployment
  (``[class token, mean(patch tokens)]``, fp16 autocast)
- maps: ``ml/callbacks/tiff_prediction_map_writer.py``

Example:
    python scripts/musica/predict_slides.py \
        --slides-glob '/data/.../prostate_mug/*.svs' --num-slides 10 \
        --checkpoint $DATA/checkpoints/linear_virchow2_lbfgs_0e2230c7.ckpt \
        --output-dir $DATA/predictions/prostate_mug_demo
"""

import argparse
import glob
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import pandas as pd
import pyvips
import torch
from PIL import Image
from rationai.masks import tissue_mask, write_big_tiff
from rationai.masks.vips_filters import (
    VipsClosing,
    VipsCompose,
    VipsGrayScaleFilter,
    VipsOpening,
    VipsOtsu,
)
from ratiopath.openslide import OpenSlide
from ratiopath.tiling import grid_tiles
from torch.utils.data import DataLoader, Dataset

from ml.callbacks import TiffPredictionMapWriter
from preprocessing.coverage_stats import sat_coverage


# Must match the training pipeline configs.
TILE_MPP = 0.5  # configs/experiment/preprocessing/tiling_0_5mpp.yaml
TILE_SIZE = 224
TILE_STRIDE = 112
MASK_MPP = 2.0  # configs/experiment/preprocessing/tissue_masks_2mpp.yaml
MASK_DISK_FACTOR = 50
TISSUE_MIN = 0.0  # predict_linear_virchow2_lbfgs_tissue_tiles.yaml
VIRCHOW2_REPO = "paige-ai/Virchow2"
NUM_REGISTER_TOKENS = 4

# Overlay colours per class index (matches checkpoint ``class_indices`` order).
CLASS_COLORS = [
    (255, 215, 0),  # Nerve
    (220, 20, 60),  # Blood
    (65, 105, 225),  # Connective-Tissue
    (255, 250, 205),  # Fat
    (148, 0, 211),  # Epithelium
    (255, 127, 80),  # Muscle
    (128, 128, 128),  # Other
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slides", nargs="*", default=[], help="Explicit slide paths.")
    parser.add_argument("--slides-glob", help="Glob of slides to sample from.")
    parser.add_argument("--num-slides", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def select_slides(args: argparse.Namespace) -> list[Path]:
    slides = [Path(p) for p in args.slides]
    if args.slides_glob:
        candidates = sorted(glob.glob(args.slides_glob))
        random.Random(args.seed).shuffle(candidates)
        slides += [Path(p) for p in candidates[: args.num_slides]]
    if not slides:
        raise SystemExit("No slides selected; pass --slides or --slides-glob.")
    return slides


def compute_tissue_mask(slide_path: Path, output_path: Path) -> np.ndarray:
    """Tissue mask exactly as ``preprocessing/tissue_masks.py`` builds it."""
    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(MASK_MPP)
        mpp_x, mpp_y = slide.slide_resolution(level)

    image = pyvips.Image.openslideload(str(slide_path), level=level)
    custom_filter = VipsCompose(
        [
            VipsGrayScaleFilter(),
            VipsOtsu(),
            VipsClosing(disc_factor=MASK_DISK_FACTOR),
            VipsOpening(disc_factor=5),
        ]
    )
    mask, _ = tissue_mask(image, mpp=(mpp_x, mpp_y), filter=custom_filter)
    mask = mask.copy_memory()  # evaluate the lazy pipeline once for TIFF + numpy
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_big_tiff(mask, path=output_path, mpp_x=mpp_x, mpp_y=mpp_y)

    mask_np = mask.numpy()
    return mask_np[..., 0] if mask_np.ndim > 2 else mask_np


def tissue_tiles(
    mask: np.ndarray, extent: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Grid tiles with ``tile_tissue_coverage > TISSUE_MIN``.

    Mirrors ``coverage_stats.process_slide``: the tile -> mask scale is the ratio
    of pixel dimensions, not of configured MPPs.
    """
    coords = np.array(
        list(
            grid_tiles(
                slide_extent=extent,
                tile_extent=(TILE_SIZE, TILE_SIZE),
                stride=(TILE_STRIDE, TILE_STRIDE),
            )
        )
    )
    mask_h, mask_w = mask.shape
    scale_x = mask_w / extent[0]
    scale_y = mask_h / extent[1]
    tm_extent_x = max(1, round(TILE_SIZE * scale_x))
    tm_extent_y = max(1, round(TILE_SIZE * scale_y))
    xs = np.round(coords[:, 0] * scale_x).astype(int)
    ys = np.round(coords[:, 1] * scale_y).astype(int)

    sat = np.zeros((mask_h + 1, mask_w + 1), dtype=np.int64)
    np.greater(mask, 0, out=sat[1:, 1:], casting="unsafe")
    np.cumsum(sat, axis=0, out=sat)
    np.cumsum(sat, axis=1, out=sat)
    coverage = sat_coverage(sat, mask_h, mask_w, ys, xs, tm_extent_x, tm_extent_y)

    keep = coverage > TISSUE_MIN
    return coords[keep], coverage[keep]


class SlideTiles(Dataset[torch.Tensor]):
    """Reads tiles like ``ratiopath.tiling.read_slide_tiles`` (white background)."""

    def __init__(
        self, slide_path: Path, coords: np.ndarray, level: int, transform: Any
    ) -> None:
        self.slide_path = slide_path
        self.coords = coords
        self.level = level
        self.transform = transform
        self._slide: OpenSlide | None = None

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._slide is None:
            self._slide = OpenSlide(self.slide_path)
        x, y = self.coords[idx]
        try:
            tile = self._slide.read_tile(
                int(x), int(y), TILE_SIZE, TILE_SIZE, self.level
            )
        except openslide.OpenSlideError as exc:
            # A corrupt JPEG tile puts the OpenSlide handle into a permanent
            # error state; reopen it and substitute background for this tile.
            print(f"[{self.slide_path.stem}] unreadable tile x={x} y={y}: {exc}")
            self._slide = OpenSlide(self.slide_path)
            tile = np.full((TILE_SIZE, TILE_SIZE, 3), 255, dtype=np.uint8)
        return self.transform(Image.fromarray(tile))


def load_virchow2(device: torch.device) -> tuple[torch.nn.Module, Any]:
    """Load Virchow2 exactly like the RationAI model-service deployment."""
    import timm
    from timm.data.config import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers.mlp import SwiGLUPacked

    model = timm.create_model(
        f"hf-hub:{VIRCHOW2_REPO}",
        pretrained=True,
        num_classes=0,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.to(device).eval()
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    return model, transform


@torch.inference_mode()
def embed(
    model: torch.nn.Module, loader: DataLoader[torch.Tensor], device: torch.device
) -> np.ndarray:
    """Return ``[class token, mean(patch tokens)]`` per tile (2 x 1280 = 2560)."""
    chunks = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            tokens = model(batch.to(device, non_blocking=True))
        class_token = tokens[:, 0]
        patch_mean = tokens[:, 1 + NUM_REGISTER_TOKENS :].mean(dim=1)
        pooled = torch.cat([class_token, patch_mean], dim=1)
        chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(chunks)


def load_classifier(
    checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Linear, list[str]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weight = ckpt["state_dict"]["decode_head.weight"]
    bias = ckpt["state_dict"]["decode_head.bias"]
    head = torch.nn.Linear(weight.shape[1], weight.shape[0])
    head.load_state_dict({"weight": weight, "bias": bias})
    class_indices: dict[str, int] = ckpt["hyper_parameters"]["class_indices"]
    class_names = sorted(class_indices, key=class_indices.__getitem__)
    return head.to(device).eval(), class_names


def write_overlay(
    slide_path: Path,
    coords: np.ndarray,
    probs: np.ndarray,
    extent: tuple[int, int],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Thumbnail with stride-resolution class map blended on top, plus legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    grid_w = -(-extent[0] // TILE_STRIDE)
    grid_h = -(-extent[1] // TILE_STRIDE)
    cells = TILE_SIZE // TILE_STRIDE
    acc = np.zeros((len(class_names), grid_h, grid_w), dtype=np.float32)
    for (x, y), prob in zip(coords // TILE_STRIDE, probs, strict=True):
        acc[:, y : y + cells, x : x + cells] += prob[:, None, None]
    covered = acc.sum(axis=0) > 0
    colors = np.array(CLASS_COLORS, dtype=np.uint8)
    rgb = colors[acc.argmax(axis=0)]

    with OpenSlide(slide_path) as slide:
        thumb = slide.get_thumbnail((2048, 2048)).convert("RGB")
    overlay = Image.fromarray(rgb).resize(thumb.size, Image.Resampling.NEAREST)
    alpha = Image.fromarray((covered * 115).astype(np.uint8)).resize(
        thumb.size, Image.Resampling.NEAREST
    )
    blended = Image.composite(overlay, thumb, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10 * thumb.size[1] / thumb.size[0] + 1))
    axes[0].imshow(thumb)
    axes[0].set_title("Slide")
    axes[1].imshow(blended)
    axes[1].set_title("Predicted tissue classes")
    for ax in axes:
        ax.axis("off")
    fig.legend(
        handles=[
            Patch(color=(c[0] / 255, c[1] / 255, c[2] / 255), label=n)
            for c, n in zip(CLASS_COLORS, class_names, strict=True)
        ],
        loc="lower center",
        ncol=len(class_names),
    )
    fig.suptitle(slide_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def predict_slide(
    slide_path: Path,
    model: torch.nn.Module,
    transform: Any,
    head: torch.nn.Linear,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    t0 = time.monotonic()
    out = args.output_dir
    stem = slide_path.stem

    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(TILE_MPP)
        mpp = slide.slide_resolution(level)
        extent = slide.level_dimensions[level]

    mask = compute_tissue_mask(slide_path, out / "tissue_masks" / f"{stem}.tiff")
    coords, coverage = tissue_tiles(mask, extent)
    del mask
    print(
        f"[{stem}] level={level} mpp={mpp[0]:.4f} extent={extent} "
        f"tissue_tiles={len(coords)} ({time.monotonic() - t0:.0f}s)",
        flush=True,
    )

    loader = DataLoader(
        SlideTiles(slide_path, coords, level, transform),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    embeddings = embed(model, loader, device)
    print(f"[{stem}] embedded ({time.monotonic() - t0:.0f}s)", flush=True)

    with torch.inference_mode():
        logits = head(torch.from_numpy(embeddings).to(device))
        probs = logits.softmax(dim=1).cpu().numpy()

    predictions = pd.DataFrame(
        {
            "slide_id": stem,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "tile_tissue_coverage": coverage,
            "pred": probs.argmax(axis=1),
        }
    )
    for i, name in enumerate(class_names):
        predictions[f"prob_{name}"] = probs[:, i]
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out / "predictions" / f"{stem}.parquet", index=False)

    slide_row = {
        "path": str(slide_path),
        "extent_x": extent[0],
        "extent_y": extent[1],
        "tile_extent_x": TILE_SIZE,
        "tile_extent_y": TILE_SIZE,
        "stride_x": TILE_STRIDE,
        "stride_y": TILE_STRIDE,
        "mpp_x": mpp[0],
        "mpp_y": mpp[1],
    }
    TiffPredictionMapWriter(slides_uri="")._write_slide_maps(
        slide_row,
        pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "probs": list(probs)}),
        out,
        class_names=class_names,
    )
    write_overlay(
        slide_path, coords, probs, extent, class_names, out / "overlays" / f"{stem}.png"
    )
    print(f"[{stem}] done ({time.monotonic() - t0:.0f}s)", flush=True)


def main() -> None:
    args = parse_args()
    slides = select_slides(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} slides={len(slides)}", flush=True)

    model, transform = load_virchow2(device)
    head, class_names = load_classifier(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series([str(s) for s in slides], name="path").to_csv(
        args.output_dir / "slides.csv", index=False
    )

    for i, slide_path in enumerate(slides, start=1):
        print(f"=== {i}/{len(slides)} {slide_path.name}", flush=True)
        predict_slide(slide_path, model, transform, head, class_names, args, device)


if __name__ == "__main__":
    main()
