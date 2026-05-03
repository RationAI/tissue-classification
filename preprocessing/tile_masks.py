from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import mlflow
import pandas as pd
import pyvips
import torch
from omegaconf import DictConfig
from rationai.masks import tile_mask, write_big_tiff
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm


def process_slide(
    slide: dict[str, Any],
    output_dir: str,
    tile_percentage_cols: list[str],
    tiles_path: str,
) -> None:
    slide_tiles = pd.read_parquet(
        tiles_path,
        columns=["x", "y", *tile_percentage_cols],
        filters=[("slide_id", "==", slide["id"])],
    )
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
    width, height = mask.size
    mask_bytes = mask.tobytes()
    del mask

    write_big_tiff(
        image=pyvips.Image.new_from_memory(
            data=mask_bytes,
            width=width,
            height=height,
            bands=1,
            format="uchar",
        ),
        path=Path(output_dir, "outlines", filename),
        mpp_x=slide["mpp_x"],
        mpp_y=slide["mpp_y"],
    )


@with_cli_args(["+preprocessing=tile_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_run_id = config.dataset.mlflow_artifacts.tiling_run_id
    tile_percentage_cols: list[str] = list(config.tile_percentage_cols)

    slides_path = mlflow.artifacts.download_artifacts(
        run_id=tiling_run_id,
        artifact_path=config.slides_artifact_path,
    )
    tiles_path = mlflow.artifacts.download_artifacts(
        run_id=tiling_run_id,
        artifact_path=config.tiles_artifact_path,
    )
    slides = pd.read_parquet(slides_path)

    items = slides.to_dict(orient="records")

    with TemporaryDirectory() as output_dir:
        Path(output_dir, "outlines").mkdir()
        for slide in tqdm(items):
            process_slide(
                slide,
                output_dir=output_dir,
                tile_percentage_cols=tile_percentage_cols,
                tiles_path=tiles_path,
            )

        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
