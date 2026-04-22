from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks import process_items, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def _draw_tile_outlines(
    tiles: pd.DataFrame,
    tile_extent: tuple[int, int],
    size: tuple[int, int],
    outline_width: int = 4,
) -> np.ndarray:
    tw, th = tile_extent
    width, height = size
    ow = outline_width
    mask = np.zeros((height, width), dtype=np.uint8)

    xs = tiles["x"].to_numpy()
    ys = tiles["y"].to_numpy()

    # Precompute shared index spans — O(n × tile_size) each
    h_cols = (xs[:, None] + np.arange(tw)[None, :]).ravel()  # (n × tw,)
    v_rows = (ys[:, None] + np.arange(th)[None, :]).ravel()  # (n × th,)

    h_cols_in = (h_cols >= 0) & (h_cols < width)
    v_rows_in = (v_rows >= 0) & (v_rows < height)

    for dy in range(ow):
        for ry in (ys + dy, ys + th - ow + dy):
            rows = np.repeat(ry, tw)
            ok = h_cols_in & (rows >= 0) & (rows < height)
            mask[rows[ok], h_cols[ok]] = 255

    for dx in range(ow):
        for cx in (xs + dx, xs + tw - ow + dx):
            cols = np.repeat(cx, th)
            ok = v_rows_in & (cols >= 0) & (cols < width)
            mask[v_rows[ok], cols[ok]] = 255

    return mask


@ray.remote(num_cpus=1, memory=3 * 1024**3)
def process_slide(
    item: tuple[dict, pd.DataFrame],
    output_dir: str,
    downsample: int,
) -> None:
    slide, slide_tiles = item
    slide_path = Path(slide["path"])

    scaled_tiles = slide_tiles.assign(
        x=slide_tiles["x"] // downsample,
        y=slide_tiles["y"] // downsample,
    )

    mask = _draw_tile_outlines(
        scaled_tiles,
        tile_extent=(
            slide["tile_extent_x"] // downsample,
            slide["tile_extent_y"] // downsample,
        ),
        size=(
            slide["extent_x"] // downsample,
            slide["extent_y"] // downsample,
        ),
    )

    height, width = mask.shape

    write_big_tiff(
        image=pyvips.Image.new_from_memory(
            data=mask.tobytes(), width=width, height=height, bands=1, format="uchar"
        ),
        path=Path(output_dir, slide_path.with_suffix(".tiff").name),
        mpp_x=slide["mpp_x"] * downsample,
        mpp_y=slide["mpp_y"] * downsample,
    )


@with_cli_args(["+preprocessing=tile_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_run_id = config.dataset.mlflow_artifacts.tiling_run_id

    slides = pd.read_parquet(
        mlflow.artifacts.download_artifacts(
            run_id=tiling_run_id,
            artifact_path=config.slides_artifact_path,
        )
    )
    tiles = pd.read_parquet(
        mlflow.artifacts.download_artifacts(
            run_id=tiling_run_id,
            artifact_path=config.tiles_artifact_path,
        ),
        columns=["slide_id", "x", "y"],
    )

    tiles_by_slide = {
        slide_id: group.drop(columns="slide_id")
        for slide_id, group in tiles.groupby("slide_id")
    }

    items = [
        (slide.to_dict(), tiles_by_slide.get(slide["id"], pd.DataFrame(columns=["x", "y"])))
        for _, slide in slides.iterrows()
    ]

    with TemporaryDirectory() as output_dir:
        process_items(
            items,
            process_item=process_slide,
            fn_kwargs={
                "output_dir": output_dir,
                "downsample": config.downsample,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
