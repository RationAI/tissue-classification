from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


def rewrite_tiff(
    src_path: Path,
    dst_path: Path,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Re-write a grayscale TIFF pyramid, optionally transforming pixel data.

    Preserves all pyramid levels, compression, tiling, predictor, and
    resolution metadata.

    Args:
        src_path: Source TIFF path.
        dst_path: Destination TIFF path.
        transform: Optional pixel-level transform applied to each page's data.
    """
    with tifffile.TiffFile(str(src_path)) as tif:  # noqa: SIM117
        with tifffile.TiffWriter(str(dst_path), bigtiff=tif.is_bigtiff) as writer:
            for page_idx, page in enumerate(tif.pages):
                assert isinstance(page, tifffile.TiffPage)
                data = page.asarray()
                if data.ndim == 3:
                    data = data[..., 0]

                if transform is not None:
                    data = transform(data)

                write_kwargs: dict[str, Any] = {
                    "photometric": "minisblack",
                    "compression": page.compression,
                }

                if page.is_tiled:
                    write_kwargs["tile"] = (page.tilelength, page.tilewidth)

                pred_tag = page.tags.get(317)
                if pred_tag and pred_tag.value > 1:
                    write_kwargs["predictor"] = pred_tag.value

                xres_tag = page.tags.get(282)
                yres_tag = page.tags.get(283)
                res_unit_tag = page.tags.get(296)
                if xres_tag and yres_tag:
                    write_kwargs["resolution"] = (xres_tag.value, yres_tag.value)
                    if res_unit_tag:
                        write_kwargs["resolutionunit"] = res_unit_tag.value

                if page_idx > 0:
                    write_kwargs["subfiletype"] = 1  # REDUCEDIMAGE

                writer.write(data, **write_kwargs)
