from pathlib import Path
from tempfile import TemporaryDirectory

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
    item: tuple[dict, pd.DataFrame],
    output_dir: str,
) -> None:
    slide, slide_tiles = item
    slide_path = Path(slide["path"])

    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide["tile_extent_x"], slide["tile_extent_y"]),
        size=(slide["extent_x"], slide["extent_y"]),
    )

    width, height = mask.size
    mask_bytes = mask.tobytes()
    del mask

    write_big_tiff(
        image=pyvips.Image.new_from_memory(
            data=mask_bytes, width=width, height=height, bands=1, format="uchar"
        ),
        path=Path(output_dir, slide_path.with_suffix(".tiff").name),
        mpp_x=slide["mpp_x"],
        mpp_y=slide["mpp_y"],
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
        )
    )

    items = [
        (slide.to_dict(), tiles[tiles["slide_id"] == slide["id"]][["x", "y"]])
        for _, slide in slides.iterrows()
    ]

    with TemporaryDirectory() as output_dir:
        process_items(
            items,
            process_item=process_slide,
            fn_kwargs={"output_dir": output_dir},
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
