from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
import torch
from omegaconf import DictConfig
from rationai.masks import process_items, tile_mask, write_big_tiff
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(num_cpus=1, memory=3 * 1024**3)
def process_slide(
    item: tuple[dict[str, Any], pd.DataFrame],
    output_dir: str,
    tile_percentage_cols: list[str],
) -> None:
    slide, slide_tiles = item
    filename = f"{Path(slide['path']).stem}.tiff"

    for percentage_col in tile_percentage_cols:
        builder = ScalarMaskBuilder(
            save_dir=Path(output_dir, percentage_col),
            filename=filename,
            extent_x=slide["extent_x"],
            extent_y=slide["extent_y"],
            mpp_x=slide["mpp_x"],
            mpp_y=slide["mpp_y"],
            extent_tile=slide["tile_extent_x"],
            stride=slide["stride_x"],
        )
        builder.update(
            data=torch.tensor(slide_tiles[percentage_col].values).unsqueeze(1),
            xs=torch.tensor(slide_tiles["x"].values),
            ys=torch.tensor(slide_tiles["y"].values),
        )
        builder.save()

    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide["tile_extent_x"], slide["tile_extent_y"]),
        size=(slide["extent_x"], slide["extent_y"]),
    )
    write_big_tiff(
        pyvips.Image.new_from_array(np.array(mask)),
        Path(output_dir, "outlines", filename),
        mpp_x=slide["mpp_x"],
        mpp_y=slide["mpp_y"],
    )


@with_cli_args(["+preprocessing=tile_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_run_id = config.dataset.mlflow_artifacts.tiling_run_id
    tile_percentage_cols: list[str] = list(config.tile_percentage_cols)

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
        columns=["slide_id", "x", "y", *tile_percentage_cols],
    )

    tiles_by_slide = {
        slide_id: group.drop(columns="slide_id")
        for slide_id, group in tiles.groupby("slide_id")
    }

    items: list[tuple[dict[str, Any], pd.DataFrame]] = [
        (
            {str(k): v for k, v in slide.to_dict().items()},
            tiles_by_slide.get(slide["id"], pd.DataFrame(columns=["x", "y", *tile_percentage_cols])),
        )
        for _, slide in slides.iterrows()
    ]

    with TemporaryDirectory() as output_dir:
        process_items(
            items,
            process_item=process_slide,
            fn_kwargs={
                "output_dir": output_dir,
                "tile_percentage_cols": tile_percentage_cols,
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
