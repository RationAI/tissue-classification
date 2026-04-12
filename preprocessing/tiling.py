import re
from pathlib import Path
from typing import Any

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.parsers import Darwin7JSONParser, GeoJSONParser
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles
from ratiopath.tiling.utils import row_hash
from shapely import Polygon, make_valid
from shapely.geometry.base import BaseGeometry

BACKGROUND_VALUE = 255


def get_validated_polygons(
    parser: GeoJSONParser | Darwin7JSONParser, original_names: list[str]
) -> list[BaseGeometry]:
    if not original_names:
        return []

    regex_pattern = f"(?i)^({'|'.join(re.escape(name) for name in original_names)})$"

    if isinstance(parser, GeoJSONParser):
        raw_polygons = parser.get_polygons(meta_category_value=regex_pattern)
    else:
        raw_polygons = parser.get_polygons(name=regex_pattern)

    valid_polygons = []
    for poly in raw_polygons:
        if poly is None or poly.is_empty:
            continue

        validated = make_valid(poly) if not poly.is_valid else poly

        if isinstance(validated, Polygon):
            valid_polygons.append(validated)
        elif hasattr(validated, "geoms"):
            for geom in validated.geoms:
                if isinstance(geom, Polygon):
                    valid_polygons.append(geom)

    return valid_polygons


def extract_mapped_regions(
    annotation_path: Path, class_mapping: dict[str, list[str]]
) -> dict[str, list[BaseGeometry]] | None:
    path_str = str(annotation_path)
    if path_str.endswith((".geo.json", ".geojson")):
        parser = GeoJSONParser(annotation_path)
    elif path_str.endswith(".json"):
        parser = Darwin7JSONParser(annotation_path)
    else:
        return None

    return {
        cls_name: get_validated_polygons(parser, original_names)
        for cls_name, original_names in class_mapping.items()
    }


def rasterize_slide_mask(
    extracted_regions: dict[str, list[BaseGeometry]],
    mask_size: tuple[int, int],
    downsample: float,
    class_indices: dict[str, int],
) -> np.ndarray:
    """Draw all class polygons into a single uint8 mask at slide-level resolution.

    Background pixels are BACKGROUND_VALUE; class pixels carry class_indices[name];
    polygon holes are re-filled with BACKGROUND_VALUE.
    """
    mask = Image.new("L", size=mask_size, color=BACKGROUND_VALUE)
    drawer = ImageDraw.Draw(mask)
    inv = 1.0 / downsample
    for cls_name, polygons in extracted_regions.items():
        idx = class_indices[cls_name]
        for poly in polygons:
            if poly.is_empty:
                continue
            drawer.polygon(
                [(x * inv, y * inv) for x, y in poly.exterior.coords], fill=idx
            )
            for interior in poly.interiors:
                drawer.polygon(
                    [(x * inv, y * inv) for x, y in interior.coords],
                    fill=BACKGROUND_VALUE,
                )
    return np.asarray(mask)


def add_annotation_path(
    row: dict[str, Any], path_to_annotation: dict[str, str]
) -> dict[str, Any]:
    row["annotation_path"] = path_to_annotation.get(row["path"], "")
    return row


def tile_with_coverage(
    row: dict[str, Any],
    class_mapping: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Generate tiles and compute per-class annotation coverage for a slide.

    Parses the annotation file associated with the slide and returns one row per
    tile with two coverage values per class:
    - tile_coverage: fraction of the whole tile area covered by the class
    - roi_coverage: fraction of the central half-size ROI covered by the class
    """
    annotation_path = Path(row["annotation_path"])
    extracted_regions = extract_mapped_regions(annotation_path, class_mapping)
    if extracted_regions is None:
        return []

    tile_w, tile_h = row["tile_extent_x"], row["tile_extent_y"]
    coordinates = list(
        grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(tile_w, tile_h),
            stride=(row["stride_x"], row["stride_y"]),
        )
    )
    if not coordinates:
        return []

    class_indices = {name: i for i, name in enumerate(class_mapping)}
    slide_mask = rasterize_slide_mask(
        extracted_regions,
        (row["extent_x"], row["extent_y"]),
        row["downsample"],
        class_indices,
    )

    qx, qy = tile_w // 4, tile_h // 4
    roi_w, roi_h = tile_w // 2, tile_h // 2

    results = []
    for x, y in coordinates:
        crop = slide_mask[y : y + tile_h, x : x + tile_w]
        roi = crop[qy : qy + roi_h, qx : qx + roi_w]
        out: dict[str, Any] = {
            "tile_x": int(x),
            "tile_y": int(y),
            "path": row["path"],
            "slide_id": row["id"],
            "level": row["level"],
            "tile_extent_x": tile_w,
            "tile_extent_y": tile_h,
            "mpp_x": row["mpp_x"],
            "mpp_y": row["mpp_y"],
        }
        tile_area = crop.size
        roi_area = roi.size
        for cls_name, idx in class_indices.items():
            out[f"tile_coverage_{cls_name}"] = (
                float(np.count_nonzero(crop == idx)) / tile_area if tile_area else 0.0
            )
            out[f"roi_coverage_{cls_name}"] = (
                float(np.count_nonzero(roi == idx)) / roi_area if roi_area else 0.0
            )
        results.append(out)
    return results


def select(row: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    return {
        "slide_id": row["slide_id"],
        "x": row["tile_x"],
        "y": row["tile_y"],
        **{
            f"tile_coverage_{cls_name}": row[f"tile_coverage_{cls_name}"]
            for cls_name in class_names
        },
        **{
            f"roi_coverage_{cls_name}": row[f"roi_coverage_{cls_name}"]
            for cls_name in class_names
        },
    }


def tiling(
    df: pd.DataFrame,
    class_mapping: dict[str, list[str]],
    tile_extent: int,
    stride: int,
    mpp: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the tiling pipeline for a set of slides.

    Returns a (slides, tiles) tuple of DataFrames. The slides DataFrame contains
    slide-level metadata with unique IDs. The tiles DataFrame contains one row per tile
    with its coordinates and two sets of per-class annotation coverage values:
    - tile_coverage_{class}: fraction of the whole tile area covered by the class
    - roi_coverage_{class}: fraction of the central half-size ROI (tile_extent/2 x
      tile_extent/2) covered by the class

    All tiles are retained regardless of coverage — filtering by threshold is done
    downstream once the distribution is known.
    """
    path_to_annotation = dict(zip(df["wsi_path"], df["annotation_path"], strict=True))
    class_names = list(class_mapping.keys())

    slides = read_slides(
        df["wsi_path"].tolist(), tile_extent=tile_extent, stride=stride, mpp=mpp
    ).map(row_hash, num_cpus=0.1, memory=128 * 1024**2)

    tiles = (
        slides.map(
            add_annotation_path,
            fn_args=(path_to_annotation,),
            num_cpus=0.1,
            memory=128 * 1024**2,
        )
        .flat_map(
            tile_with_coverage,
            fn_args=(class_mapping,),
            num_cpus=1,
            memory=2 * 1024**3,
        )
        .repartition(target_num_rows_per_block=4096)
        .map(
            select,
            fn_args=(class_names,),
            num_cpus=0.1,
            memory=128 * 1024**2,
        )
    )

    return slides.to_pandas(), tiles.to_pandas()


@with_cli_args(["+preprocessing=tiling"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
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
            class_mapping=OmegaConf.to_container(config.class_mapping, resolve=True),
            tile_extent=config.tile_size,
            stride=config.stride,
            mpp=config.mpp,
        )

        save_mlflow_dataset(df_slides, df_tiles, f"{split_name}_split")


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
