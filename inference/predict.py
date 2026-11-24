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
from ml.prediction_maps import ClassVoteAssembler


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

Panel = tuple[str, "np.ndarray | None", "np.ndarray | None"]
"""(title, class_idx grid or None for the plain thumbnail, covered grid or None)."""


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
    path: Path, slide_dir: Path, tissue_min: float
) -> tuple[SlideGeometry, pd.DataFrame]:
    """Geometry + tissue tiles for a slide; cached in ``tiles.parquet``."""
    geometry = slide_geometry(path)
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
        class_map = assemble_class_map(
            probs, frame["x"].to_numpy(), frame["y"].to_numpy(), geometry
        )
        # Same value encoding as the training prediction maps.
        class_map.write_tiff(
            (geometry.extent_x, geometry.extent_y),
            slide_dir / f"pred_{head.model}.tiff",
            (geometry.mpp_x, geometry.mpp_y),
        )
        class_idx, covered = class_map.class_indices()
        write_panels(
            geometry,
            [
                (geometry.path.name, None, None),
                (f"{head.model} prediction", class_idx, covered),
            ],
            head.class_names,
            thumbnail_size,
            slide_dir / f"overlay_{head.model}.png",
        )
    return frame


def assemble_class_map(
    probs: np.ndarray, xs: np.ndarray, ys: np.ndarray, geometry: SlideGeometry
) -> ClassVoteAssembler:
    """Fuse overlapping tiles into a per-cell class grid (same rule as training maps)."""
    assembler = ClassVoteAssembler(
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
        torch.from_numpy(xs),
        torch.from_numpy(ys),
    )
    return assembler


def class_colors(class_names: list[str]) -> np.ndarray:
    return np.array([CLASS_COLORS[name] for name in class_names], dtype=np.uint8)


def write_panels(
    geometry: SlideGeometry,
    panels: list[Panel],
    class_names: list[str],
    thumbnail_size: int,
    path: Path,
    alpha: float = 0.45,
) -> None:
    """Side-by-side thumbnail panels with colour-coded class grids and a legend."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    from ratiopath.openslide import OpenSlide

    with OpenSlide(geometry.path) as slide:
        thumb = slide.get_thumbnail((thumbnail_size, thumbnail_size)).convert("RGB")
    width, height = thumb.size
    base = np.asarray(thumb, dtype=np.float32)
    colors = class_colors(class_names)

    def blend(class_idx: np.ndarray, covered: np.ndarray) -> np.ndarray:
        # The class grid spans the level extent in stride-sized cells; stretching
        # it to the thumbnail is off by < 1 cell, invisible at this scale.
        grid = Image.fromarray(colors[class_idx]).resize(
            (width, height), Image.Resampling.NEAREST
        )
        cov = Image.fromarray(covered.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.NEAREST
        )
        weight = (np.asarray(cov, dtype=np.float32) / 255.0)[..., None] * alpha
        over = np.asarray(grid, dtype=np.float32)
        return (base * (1 - weight) + over * weight).astype(np.uint8)

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(7 * n, 7 * height / width + 1.2), dpi=150, squeeze=False
    )
    for ax, (title, class_idx, covered) in zip(axes[0], panels, strict=True):
        if class_idx is None or covered is None:
            ax.imshow(thumb)
        else:
            ax.imshow(blend(class_idx, covered))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.legend(
        handles=[
            Patch(color=(c[0] / 255, c[1] / 255, c[2] / 255), label=name)
            for name, c in zip(class_names, colors, strict=True)
        ],
        loc="lower center",
        ncol=len(class_names),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_mask_png(mask: np.ndarray, path: Path, max_size: int = 2048) -> None:
    from PIL import Image

    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    img.thumbnail((max_size, max_size))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
