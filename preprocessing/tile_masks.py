from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import mlflow
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
    downsample: int,
) -> None:
    slide, slide_tiles = item

    scaled_tiles = slide_tiles.assign(
        x=slide_tiles["x"] // downsample,
        y=slide_tiles["y"] // downsample,
    )

    mask = tile_mask(
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

    width, height = mask.size

    write_big_tiff(
        image=pyvips.Image.new_from_memory(
            data=mask.tobytes(), width=width, height=height, bands=1, format="uchar"
        ),
        path=Path(output_dir, f"{Path(slide['path']).stem}.tiff"),
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

    items: list[tuple[dict[str, Any], pd.DataFrame]] = [
        (
            {str(k): v for k, v in slide.to_dict().items()},
            tiles_by_slide.get(slide["id"], pd.DataFrame(columns=["x", "y"])),
        )
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
