"""Class prediction maps from overlapping tile probabilities.

Shared by the Lightning ``TiffPredictionMapWriter`` callback and the standalone
``inference`` CLI; kept free of Lightning / MLflow imports.
"""

from math import gcd
from pathlib import Path

import numpy as np
import torch


class ClassVoteAssembler:
    """Confidence-weighted per-class accumulator over the full tile footprint.

    Mirrors ``HeatmapAssembler``'s GCD-compressed grid + ``(x, y)`` -> ROI
    mapping, but accumulates the per-class softmax of every tile that covers a
    cell instead of summing a categorical index. Overlapping strided tiles are
    fused by ``argmax`` over the summed probabilities, so heavy stride overlap
    (e.g. tile 224 / stride 112) is used instead of discarded. ``count == 0``
    marks never-covered cells as background.
    """

    def __init__(
        self,
        extent_x: int,
        extent_y: int,
        tile_x: int,
        tile_y: int,
        stride_x: int,
        stride_y: int,
        n_classes: int,
    ) -> None:
        self.cdx = gcd(stride_x, tile_x)
        self.cdy = gcd(stride_y, tile_y)
        self._sx = stride_x // self.cdx
        self._sy = stride_y // self.cdy
        self._tx = tile_x // self.cdx
        self._ty = tile_y // self.cdy
        self._gx = extent_x // self.cdx
        self._gy = extent_y // self.cdy
        self._n_classes = n_classes
        self._acc = torch.zeros(n_classes, self._gy, self._gx, dtype=torch.float32)
        self._count = torch.zeros(self._gy, self._gx, dtype=torch.int32)

    def update(self, probs: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> None:
        for prob, x, y in zip(probs, xs, ys, strict=False):
            cx = int(x.item()) // self.cdx
            cy = int(y.item()) // self.cdy
            x0, y0 = cx * self._sx, cy * self._sy
            x1 = min(x0 + self._tx, self._gx)
            y1 = min(y0 + self._ty, self._gy)
            if x1 <= x0 or y1 <= y0:
                continue
            self._acc[:, y0:y1, x0:x1] += prob[:, None, None]
            self._count[y0:y1, x0:x1] += 1

    def class_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell argmax class index and a mask of cells covered by any tile."""
        return self._acc.argmax(0).numpy(), self._count.numpy() > 0

    def labels(self, background_value: int = 0) -> np.ndarray:
        """Encode prediction labels like ``remap_annotation_masks``.

        Class ``i`` (0-based) maps to
        ``round(255 * (i + 1) / n_classes)``. Never-covered pixels map to
        ``background_value``.

        The reporting tool expects GT and prediction masks in the same
        evenly-spread value space, so this must mirror that LUT exactly.
        """
        idx, covered = self.class_indices()
        out = spread_lut(self._n_classes)[idx].astype(np.uint8)
        out[~covered] = uint8_scalar(background_value)
        return np.ascontiguousarray(out)

    def write_tiff(
        self,
        extent: tuple[int, int],
        path: Path,
        mpp: tuple[float, float],
        background_value: int = 0,
    ) -> None:
        """Write :meth:`labels` as a BigTIFF class map at the WSI ``extent``."""
        emit_mask(
            self.labels(background_value),
            self.cdx,
            self.cdy,
            extent,
            int(uint8_scalar(background_value)),
            path,
            mpp,
        )


def emit_mask(
    grid: np.ndarray,
    cdx: int,
    cdy: int,
    extent: tuple[int, int],
    background_value: int,
    path: Path,
    mpp: tuple[float, float],
) -> None:
    """Upscale the GCD-compressed grid back to WSI extent and write a BigTIFF.

    NEAREST resize by the GCD factor (tile footprint placed at its true
    position, no recenter), background-padded to the full extent.
    """
    import pyvips
    from rationai.masks import write_big_tiff

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = pyvips.Image.new_from_array(grid).cast(pyvips.BandFormat.UCHAR)
    mask = mask.resize(cdx, vscale=cdy, kernel=pyvips.enums.Kernel.NEAREST)
    mask = mask.embed(
        0,
        0,
        extent[0],
        extent[1],
        extend=pyvips.enums.Extend.BACKGROUND,
        background=[background_value],
    )
    write_big_tiff(mask, path, mpp[0], mpp[1])


def spread_lut(n_classes: int) -> np.ndarray:
    """0-based class index -> evenly-spread uint8 value.

    Mirrors ``preprocessing/remap_annotation_masks.py`` exactly so prediction
    masks share the GT value space the reporting tool expects:
    class ``i`` -> ``round(255 * (i + 1) / n_classes)``. Index 0 of the
    returned array is unused by callers (background is set separately).
    """
    return np.array(
        [round(255 * (i + 1) / n_classes) for i in range(n_classes)],
        dtype=np.uint8,
    )


def uint8_scalar(value: int) -> int:
    return np.asarray(value).clip(0, np.iinfo(np.uint8).max).astype(np.uint8).item()
