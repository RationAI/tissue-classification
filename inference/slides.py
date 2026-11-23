"""Slide geometry, tissue mask and tile grid — mirrors the training preprocessing.

- level selection: ``OpenSlide.closest_level(mpp)`` (as ``ratiopath.ray.read_slides``)
- tile grid: ``ratiopath.tiling.grid_tiles`` on the selected level (as ``tiling.py``)
- tissue mask: same filter as ``tissue_masks.py`` (grayscale, Otsu, closing 50,
  opening 5) at the level closest to 2 mpp
- tissue coverage: fraction of the tile footprint covered by the mask, with the
  tile -> mask scale taken from image dimensions (as ``coverage_stats.py``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ratiopath.openslide import OpenSlide
from ratiopath.tiling import grid_tiles


if TYPE_CHECKING:
    from pathlib import Path


SLIDE_EXTENSIONS = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
    ".svslide",
)


# Effective tile resolutions in the training data (tiling run fdf7550a...):
# TCGA 0.228-0.253, FNBrno 0.44, MUG 0.485, MOU 0.486 mpp.
TRAINING_MPP_RANGE = (0.228, 0.486)


@dataclass(frozen=True)
class SlideGeometry:
    path: Path
    level: int
    extent_x: int
    extent_y: int
    mpp_x: float
    mpp_y: float
    downsample: float
    tile_size: int
    stride: int


def find_slides(inputs: list[Path]) -> list[Path]:
    """Expand files and directories (non-recursive) into a sorted list of slides."""
    slides: list[Path] = []
    for item in inputs:
        if item.is_dir():
            slides.extend(
                p
                for p in sorted(item.iterdir())
                if p.is_file() and p.suffix.lower() in SLIDE_EXTENSIONS
            )
        elif item.is_file():
            slides.append(item)
        else:
            raise FileNotFoundError(item)
    if not slides:
        raise FileNotFoundError(f"no slides found in {[str(p) for p in inputs]}")
    return slides


def slide_geometry(
    path: Path, mpp: float, tile_size: int, stride: int
) -> SlideGeometry:
    with OpenSlide(path) as slide:
        try:
            level = slide.closest_level(mpp)
            mpp_x, mpp_y = slide.slide_resolution(level)
        except KeyError as exc:
            raise ValueError(
                f"{path.name}: slide has no microns-per-pixel metadata "
                "(openslide.mpp-x / openslide.mpp-y); cannot select the 0.5 mpp level"
            ) from exc
        extent_x, extent_y = slide.level_dimensions[level]
        downsample = float(slide.level_downsamples[level])

    # Tiles are read natively on the closest level, exactly like the training
    # tiling. On 0.25 mpp SVS with a 1/4/16 pyramid that level is 0.25 mpp, which
    # is what most training slides (TCGA, ~94 % of tiles) used, so only warn
    # outside the resolution range the training data covered.
    lo, hi = TRAINING_MPP_RANGE
    if not lo * 0.9 <= mpp_x <= hi * 1.1:
        print(
            f"[warn] {path.name}: tiles read at {mpp_x:.3f} mpp (level {level}), "
            f"outside the training range {lo}-{hi} mpp; predictions may be unreliable",
            flush=True,
        )
    return SlideGeometry(
        path=path,
        level=level,
        extent_x=int(extent_x),
        extent_y=int(extent_y),
        mpp_x=float(mpp_x),
        mpp_y=float(mpp_y),
        downsample=downsample,
        tile_size=tile_size,
        stride=stride,
    )


def tissue_mask(path: Path, mpp: float = 2.0) -> np.ndarray:
    """Binary tissue mask (uint8 0/255) at the level closest to ``mpp``."""
    import pyvips
    from rationai.masks import tissue_mask as rationai_tissue_mask
    from rationai.masks.vips_filters import (
        VipsClosing,
        VipsCompose,
        VipsGrayScaleFilter,
        VipsOpening,
        VipsOtsu,
    )

    with OpenSlide(path) as slide:
        level = slide.closest_level(mpp)
        mpp_x, mpp_y = slide.slide_resolution(level)

    image = pyvips.Image.openslideload(str(path), level=level)
    custom_filter = VipsCompose(
        [
            VipsGrayScaleFilter(),
            VipsOtsu(),
            VipsClosing(disc_factor=50),
            VipsOpening(disc_factor=5),
        ]
    )
    mask, _ = rationai_tissue_mask(image, mpp=(mpp_x, mpp_y), filter=custom_filter)
    array = np.asarray(mask.numpy())
    if array.ndim > 2:
        array = array[..., 0]
    return array


def tile_grid(geometry: SlideGeometry) -> np.ndarray:
    """(N, 2) tile top-left coordinates on the selected level."""
    coords = list(
        grid_tiles(
            slide_extent=(geometry.extent_x, geometry.extent_y),
            tile_extent=(geometry.tile_size, geometry.tile_size),
            stride=(geometry.stride, geometry.stride),
        )
    )
    if not coords:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(coords, dtype=np.int64)


def tissue_coverage(
    mask: np.ndarray, geometry: SlideGeometry, coords: np.ndarray
) -> np.ndarray:
    """Fraction of each tile's footprint covered by tissue (``tile_tissue_coverage``)."""
    mask_h, mask_w = mask.shape
    scale_x = mask_w / geometry.extent_x
    scale_y = mask_h / geometry.extent_y
    ext_x = max(1, round(geometry.tile_size * scale_x))
    ext_y = max(1, round(geometry.tile_size * scale_y))

    xs = np.round(coords[:, 0] * scale_x).astype(int)
    ys = np.round(coords[:, 1] * scale_y).astype(int)

    sat = np.zeros((mask_h + 1, mask_w + 1), dtype=np.int64)
    np.greater(mask, 0, out=sat[1:, 1:], casting="unsafe")
    np.cumsum(sat, axis=0, out=sat)
    np.cumsum(sat, axis=1, out=sat)

    y0, x0 = np.clip(ys, 0, mask_h), np.clip(xs, 0, mask_w)
    y1, x1 = np.clip(ys + ext_y, 0, mask_h), np.clip(xs + ext_x, 0, mask_w)
    sums = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
    return sums / (ext_x * ext_y)
