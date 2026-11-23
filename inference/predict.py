"""Per-slide prediction: tissue tiles -> embeddings -> linear head -> maps.

Output layout (one directory per slide)::

    <out>/<slide_stem>/
        tiles.parquet                 x, y (level coords), tissue_coverage
        embeddings_<model>.npy        cached; reused on re-runs (resume)
        predictions_<model>.parquet   x, y, prob_<class>..., pred, pred_name
        pred_<model>.tiff             class map, BigTIFF at the 0.5 mpp level
        overlay_<model>.png           thumbnail | thumbnail + class map, with legend
        tissue_mask.png               QC of the tissue mask
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from inference.slides import (
    SlideGeometry,
    slide_geometry,
    tile_grid,
    tissue_coverage,
    tissue_mask,
)


if TYPE_CHECKING:
    from pathlib import Path

    from ratiopath.openslide import OpenSlide

    from inference.heads import LinearHead
    from preprocessing.foundation_models import FoundationModel


# Fixed colours per class name so maps are comparable across models and slides.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Nerve": (255, 215, 0),
    "Blood": (220, 20, 60),
    "Connective-Tissue": (255, 160, 200),
    "Fat": (135, 206, 250),
    "Epithelium": (148, 0, 211),
    "Muscle": (255, 120, 0),
    "Other": (60, 179, 113),
}
_FALLBACK_COLORS = [(31, 119, 180), (44, 160, 44), (214, 39, 40), (127, 127, 127)]


class _TileDataset(Dataset[np.ndarray]):
    """Reads tiles exactly like training (``OpenSlide.read_tile`` on the level)."""

    def __init__(self, geometry: SlideGeometry, coords: np.ndarray) -> None:
        self.geometry = geometry
        self.coords = coords
        self._slide: OpenSlide | None = None

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> np.ndarray:
        from ratiopath.openslide import OpenSlide

        if self._slide is None:  # opened lazily, once per DataLoader worker
            self._slide = OpenSlide(self.geometry.path)
        x, y = self.coords[index]
        size = self.geometry.tile_size
        return self._slide.read_tile(int(x), int(y), size, size, self.geometry.level)


def prepare_slide(
    path: Path,
    slide_dir: Path,
    mpp: float,
    tile_size: int,
    stride: int,
    tissue_min: float,
) -> tuple[SlideGeometry, pd.DataFrame]:
    """Geometry + tissue tiles for a slide; cached in ``tiles.parquet``."""
    geometry = slide_geometry(path, mpp=mpp, tile_size=tile_size, stride=stride)
    tiles_path = slide_dir / "tiles.parquet"
    if tiles_path.exists():
        return geometry, pd.read_parquet(tiles_path)

    mask = tissue_mask(path)
    _save_mask_png(mask, slide_dir / "tissue_mask.png")
    coords = tile_grid(geometry)
    coverage = tissue_coverage(mask, geometry, coords)
    keep = coverage > tissue_min
    tiles = pd.DataFrame(
        {
            "x": coords[keep, 0],
            "y": coords[keep, 1],
            "tissue_coverage": coverage[keep].astype(np.float32),
        }
    )
    tiles.to_parquet(tiles_path, index=False)
    return geometry, tiles


def embed_slide(
    model: FoundationModel,
    geometry: SlideGeometry,
    tiles: pd.DataFrame,
    cache_path: Path,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == len(tiles):
            return cached
        print(
            f"  stale cache {cache_path.name} ({len(cached)} != {len(tiles)}), recomputing"
        )

    coords = tiles[["x", "y"]].to_numpy()
    loader = DataLoader(
        _TileDataset(geometry, coords),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=np.stack,
        persistent_workers=False,
    )
    out = np.empty((len(coords), model.spec.embedding_dim), dtype=np.float32)
    done = 0
    start = time.monotonic()
    for batch in loader:
        out[done : done + len(batch)] = model.embed(batch)
        done += len(batch)
        rate = done / max(time.monotonic() - start, 1e-6)
        print(
            f"\r  {model.spec.name}: {done}/{len(coords)} tiles ({rate:.0f}/s)",
            end="",
            flush=True,
        )
    print(flush=True)

    tmp = cache_path.with_suffix(".tmp.npy")
    np.save(tmp, out)
    tmp.replace(cache_path)
    return out


def write_predictions(
    head: LinearHead,
    embeddings: np.ndarray,
    geometry: SlideGeometry,
    tiles: pd.DataFrame,
    slide_dir: Path,
    thumbnail_size: int,
) -> pd.DataFrame:
    probs = head.probabilities(embeddings)
    pred = probs.argmax(axis=1)
    frame = pd.DataFrame({"x": tiles["x"].to_numpy(), "y": tiles["y"].to_numpy()})
    for i, name in enumerate(head.class_names):
        frame[f"prob_{name}"] = probs[:, i]
    frame["pred"] = pred
    frame["pred_name"] = np.asarray(head.class_names)[pred]
    frame.to_parquet(slide_dir / f"predictions_{head.model}.parquet", index=False)

    if len(frame):
        class_idx, covered, cell = _class_grid(probs, frame, geometry)
        _write_class_tiff(
            class_idx,
            covered,
            cell,
            geometry,
            len(head.class_names),
            slide_dir / f"pred_{head.model}.tiff",
        )
        _write_overlay(
            class_idx,
            covered,
            geometry,
            head,
            thumbnail_size,
            slide_dir / f"overlay_{head.model}.png",
        )
    return frame


def _class_grid(
    probs: np.ndarray, frame: pd.DataFrame, geometry: SlideGeometry
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Fuse overlapping tiles into a per-cell class grid (same rule as training maps)."""
    from ml.callbacks.tiff_prediction_map_writer import _ClassVoteAssembler

    assembler = _ClassVoteAssembler(
        geometry.extent_x,
        geometry.extent_y,
        geometry.tile_size,
        geometry.tile_size,
        geometry.stride,
        geometry.stride,
        n_classes=probs.shape[1],
    )
    assembler.update(
        torch.from_numpy(probs.astype(np.float32)),
        torch.from_numpy(frame["x"].to_numpy()),
        torch.from_numpy(frame["y"].to_numpy()),
    )
    class_idx, covered = assembler.class_indices()
    return class_idx, covered, (assembler.cdx, assembler.cdy)


def _write_class_tiff(
    class_idx: np.ndarray,
    covered: np.ndarray,
    cell: tuple[int, int],
    geometry: SlideGeometry,
    n_classes: int,
    path: Path,
) -> None:
    """BigTIFF in the value encoding of the training prediction maps."""
    from ml.callbacks.tiff_prediction_map_writer import _emit_mask, _spread_lut

    labels = _spread_lut(n_classes)[class_idx].astype(np.uint8)
    labels[~covered] = 0
    _emit_mask(
        np.ascontiguousarray(labels),
        cell[0],
        cell[1],
        (geometry.extent_x, geometry.extent_y),
        0,
        path,
        (geometry.mpp_x, geometry.mpp_y),
    )


def class_colors(class_names: list[str]) -> np.ndarray:
    return np.array(
        [
            CLASS_COLORS.get(name, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
            for i, name in enumerate(class_names)
        ],
        dtype=np.uint8,
    )


def _write_overlay(
    class_idx: np.ndarray,
    covered: np.ndarray,
    geometry: SlideGeometry,
    head: LinearHead,
    thumbnail_size: int,
    path: Path,
    alpha: float = 0.45,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    from ratiopath.openslide import OpenSlide

    with OpenSlide(geometry.path) as slide:
        thumb = slide.get_thumbnail((thumbnail_size, thumbnail_size)).convert("RGB")
    width, height = thumb.size

    # The class grid spans the level extent in stride-sized cells; stretching it
    # to the thumbnail is off by < 1 cell (112 px at 0.5 mpp), invisible here.
    colors = class_colors(head.class_names)
    rgb = colors[class_idx]
    grid_img = Image.fromarray(rgb).resize((width, height), Image.Resampling.NEAREST)
    cov_img = Image.fromarray(covered.astype(np.uint8) * 255).resize(
        (width, height), Image.Resampling.NEAREST
    )
    base = np.asarray(thumb, dtype=np.float32)
    over = np.asarray(grid_img, dtype=np.float32)
    weight = (np.asarray(cov_img, dtype=np.float32) / 255.0)[..., None] * alpha
    blended = (base * (1 - weight) + over * weight).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7 * height / width + 1.2), dpi=150)
    axes[0].imshow(thumb)
    axes[0].set_title(geometry.path.name, fontsize=9)
    axes[1].imshow(blended)
    axes[1].set_title(f"{head.model} prediction", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.legend(
        handles=[
            Patch(color=(c[0] / 255, c[1] / 255, c[2] / 255), label=n)
            for n, c in zip(head.class_names, colors, strict=True)
        ],
        loc="lower center",
        ncol=len(head.class_names),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _save_mask_png(mask: np.ndarray, path: Path, max_size: int = 2048) -> None:
    from PIL import Image

    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    img.thumbnail((max_size, max_size))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
