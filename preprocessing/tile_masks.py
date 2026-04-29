from math import gcd
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks import process_items, tile_mask, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(num_cpus=1, memory=3 * 1024**3)
def process_slide(
    item: tuple[dict[str, Any], pd.DataFrame],
    output_dir: str,
) -> None:
    slide, slide_tiles = item
    filename = f"{Path(slide['path']).stem}.tiff"

    d = gcd(int(slide["stride_x"]), int(slide["tile_extent_x"]))
    mask = tile_mask(
        slide_tiles.assign(x=slide_tiles["x"] // d, y=slide_tiles["y"] // d),
        tile_extent=(slide["tile_extent_x"] // d, slide["tile_extent_y"] // d),
        size=(slide["extent_x"] // d, slide["extent_y"] // d),
    )
    write_big_tiff(
        pyvips.Image.new_from_array(np.array(mask)),
        Path(output_dir, "outlines", filename),
        mpp_x=slide["mpp_x"] * d,
        mpp_y=slide["mpp_y"] * d,
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

    items: list[tuple[dict[str, Any], pd.DataFrame]] = [
        (
            {str(k): v for k, v in slide.to_dict().items()},
            tiles_by_slide.get(
                slide["id"], pd.DataFrame(columns=["x", "y"])
            ),
        )
        for _, slide in slides.iterrows()
    ]

    with TemporaryDirectory() as output_dir:
        Path(output_dir, "outlines").mkdir()
        process_items(
            items,
            process_item=process_slide,
            fn_kwargs={
                "output_dir": output_dir,
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
