from pathlib import Path
from typing import Any

import hydra
import mlflow.artifacts
import pandas as pd
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, tile_overlay_overlap
from ratiopath.tiling.utils import row_hash
from ray.data.expressions import col
from shapely import Polygon
from shapely.geometry import box


def add_mask_paths(row: dict[str, Any], mask_folder: Path) -> dict[str, Any]:
    row["mask_path"] = str(mask_folder / f"{Path(row["path"]).stem}.tiff")
    return row


def create_tissue_roi(tile_extent: int) -> Polygon:
    offset = tile_extent // 4
    size = tile_extent // 2
    return box(offset, offset, offset + size, offset + size)


def tile(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["id"],
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
            "mask_path": row["mask_path"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
        )
    ]


def process_mask_overlap(row: dict[str, Any]) -> dict[str, Any]:
    overlap = row["mask_overlap"]

    bg_prop = overlap.get("255", 0.0)
    if bg_prop is None:
        bg_prop = 1.0

    row["tissue_prop"] = 1.0 - bg_prop

    best_class = 0
    best_prop = 0.0
    for cls_str, prop in overlap.items():
        if cls_str != "255" and prop is not None and prop > best_prop:
            best_prop = prop
            best_class = int(cls_str)

    row["label"] = best_class
    return row


def select(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slide_id": row["slide_id"],
        "x": row["tile_x"],
        "y": row["tile_y"],
        "tissue_prop": row["tissue_prop"],
        "label": row["label"],
    }


def tiling(
    df: pd.DataFrame,
    mask_folder: Path,
    tile_extent: int,
    stride: int,
    mpp: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slides = read_slides(df["wsi_path"].tolist(), tile_extent=tile_extent, stride=stride, mpp=mpp).map(
        row_hash, num_cpus=0.1, memory=128 * 1024**2
    )

    tiles = (
        slides.map(
            add_mask_paths,
            fn_args=(mask_folder,),
            num_cpus=0.1,
            memory=128 * 1024**2,
        )
        .flat_map(tile, num_cpus=0.2, memory=128 * 1024**2)
        .repartition(target_num_rows_per_block=4096)
        .with_column(
            "mask_overlap",
            tile_overlay_overlap(
                create_tissue_roi(tile_extent),
                col("mask_path"),
                col("tile_x"),
                col("tile_y"),
                col("mpp_x"),
                col("mpp_y"),
            ),
            num_cpus=1,
            memory=256 * 1024**2,
        )
        .map(process_mask_overlap, num_cpus=0.1, memory=128 * 1024**2)
        .map(select, num_cpus=0.1, memory=128 * 1024**2)
    )

    return slides.to_pandas(), tiles.to_pandas()


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    mask_folder = Path(
        mlflow.artifacts.download_artifacts(
            run_id=config.dataset.mlflow_artifacts.annotation_masks_run_id,
            artifact_path=str(Path(config.dataset.mlflow_artifacts.annotation_masks_filename).parent),
        )
    )

    split_artifacts = {
        "train": config.dataset.mlflow_artifacts.train_split_filename,
        "test": config.dataset.mlflow_artifacts.test_split_filename,
    }

    for split_name, split_artifact in split_artifacts.items():
        split_df = pd.read_csv(
            mlflow.artifacts.download_artifacts(
                run_id=config.dataset.mlflow_artifacts.train_test_split_run_id,
                artifact_path=split_artifact,
            ),
        )

        df_slides, df_tiles = tiling(
            df=split_df,
            mask_folder=mask_folder,
            tile_extent=config.tile_size,
            stride=config.stride,
            mpp=config.mpp,
        )

        save_mlflow_dataset(df_slides, df_tiles, f"{split_name}_split")


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
