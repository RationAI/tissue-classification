"""Perineural invasion (PNI) candidates from tile-level tissue predictions.

A tile is flagged when it is predicted Nerve and an Epithelium tile lies within
``radius`` grid steps, or the other way round: tumour glands in prostate are
epithelial, so invasion shows up as the two classes meeting. Flagged tiles are
joined into 8-connected regions on the stride grid, and regions smaller than
``min_tiles`` are dropped. Regions are ranked per slide by size, which on the
annotated slides was the feature that best separated PNI from other contacts.

Per slide the stage logs:
- ``masks/<slide file name>``: uint8 BigTIFF in the tiling level's pixel space
  (same as the prediction maps), 255 inside a candidate region and 0 elsewhere
- ``regions/<slide stem>.geojson``: one level-0 polygon per region, rank and tile
  counts in the properties, importable into QuPath
- ``regions.csv``: all regions of all slides

and, for slides with annotations, how many PNI and Nerve polygons a region touches.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy import ndimage
from shapely import STRtree, box, unary_union
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from pni.annotations import NERVE, PNI, load_qupath_annotations


def flag_tiles(
    tiles: pd.DataFrame,
    stride: int,
    nerve_index: int,
    epithelium_index: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid of tiles where Nerve and Epithelium predictions meet.

    Args:
        tiles: Tiles of one slide with level coordinates ``x``, ``y`` and the
            predicted class index ``pred``.
        stride: Tiling stride in level pixels; tile ``(x, y)`` sits at grid cell
            ``(x // stride, y // stride)``.
        nerve_index: Class index of Nerve.
        epithelium_index: Class index of Epithelium.
        radius: Chebyshev distance in grid steps within which the other class
            must occur.

    Returns:
        ``(flag, nerve, epithelium)`` boolean grids indexed ``[gy, gx]``.
    """
    gx = (tiles["x"].to_numpy() // stride).astype(np.int64)
    gy = (tiles["y"].to_numpy() // stride).astype(np.int64)
    pred = tiles["pred"].to_numpy()
    shape = (int(gy.max()) + 1, int(gx.max()) + 1)

    nerve = np.zeros(shape, dtype=bool)
    epithelium = np.zeros(shape, dtype=bool)
    nerve[gy, gx] = pred == nerve_index
    epithelium[gy, gx] = pred == epithelium_index

    window = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    flag = (nerve & ndimage.binary_dilation(epithelium, window)) | (
        epithelium & ndimage.binary_dilation(nerve, window)
    )
    return flag, nerve, epithelium


def find_regions(
    flag: np.ndarray,
    nerve: np.ndarray,
    epithelium: np.ndarray,
    min_tiles: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Join flagged tiles into 8-connected regions and rank them by size.

    Returns:
        ``(labels, regions)``: ``labels`` holds the region rank (1 = largest) per
        grid cell, 0 outside kept regions; ``regions`` has one row per kept region
        with ``rank``, ``n_tiles``, ``n_nerve``, ``n_epithelium`` and the grid
        cells' ``gx`` / ``gy`` arrays.
    """
    components, _ = ndimage.label(flag, structure=np.ones((3, 3), dtype=bool))
    rows = []
    for index, window in enumerate(ndimage.find_objects(components), start=1):
        if window is None:
            continue
        ys, xs = np.nonzero(components[window] == index)
        ys = ys + window[0].start
        xs = xs + window[1].start
        if len(xs) < min_tiles:
            continue
        rows.append(
            {
                "component": index,
                "n_tiles": len(xs),
                "n_nerve": int(nerve[ys, xs].sum()),
                "n_epithelium": int(epithelium[ys, xs].sum()),
                "gx": xs,
                "gy": ys,
            }
        )

    regions = pd.DataFrame(
        rows,
        columns=["component", "n_tiles", "n_nerve", "n_epithelium", "gx", "gy"],
    )
    regions = regions.sort_values(
        ["n_tiles", "component"], ascending=[False, True], ignore_index=True
    )
    regions.insert(0, "rank", np.arange(1, len(regions) + 1))

    labels = np.zeros(flag.shape, dtype=np.int32)
    for row in regions.to_dict("records"):
        labels[row["gy"], row["gx"]] = row["rank"]
    return labels, regions.drop(columns="component")


def region_polygon(
    gx: np.ndarray,
    gy: np.ndarray,
    stride: int,
    tile_extent: int,
    downsample: float,
) -> BaseGeometry:
    """Level-0 union of the full footprints of a region's tiles."""
    return unary_union(
        [
            box(
                x * stride * downsample,
                y * stride * downsample,
                (x * stride + tile_extent) * downsample,
                (y * stride + tile_extent) * downsample,
            )
            for x, y in zip(gx, gy, strict=True)
        ]
    )


def footprint_grid(labels: np.ndarray, stride: int, tile_extent: int) -> np.ndarray:
    """Binary stride-cell grid covered by the full footprint of any region tile.

    A tile at cell ``(gx, gy)`` spans ``tile_extent / stride`` cells from there.
    """
    if tile_extent % stride:
        raise ValueError(
            f"tile_extent {tile_extent} must be a multiple of stride {stride}"
        )
    span = tile_extent // stride
    occupied = labels > 0
    grid = np.zeros(
        (occupied.shape[0] + span - 1, occupied.shape[1] + span - 1), dtype=bool
    )
    for dy in range(span):
        for dx in range(span):
            grid[dy : dy + occupied.shape[0], dx : dx + occupied.shape[1]] |= occupied
    return grid


def evaluate(
    polygons: list[BaseGeometry],
    annotations: dict[str, list[Any]],
) -> dict[str, float]:
    """Count annotated PNI / Nerve polygons touched by any candidate region."""
    tree = STRtree(polygons) if polygons else None

    def touched(targets: list[Any]) -> int:
        if tree is None:
            return 0
        return sum(bool(len(tree.query(t, predicate="intersects"))) for t in targets)

    pni, nerve = annotations.get(PNI, []), annotations.get(NERVE, [])
    pni_tree = STRtree(pni) if pni else None
    on_pni = (
        sum(bool(len(pni_tree.query(p, predicate="intersects"))) for p in polygons)
        if pni_tree is not None
        else 0
    )
    return {
        "pni_polygons": len(pni),
        "pni_hit": touched(pni),
        "nerve_polygons": len(nerve),
        "nerve_hit": touched(nerve),
        "regions_on_pni": on_pni,
    }


def write_mask(
    grid: np.ndarray,
    stride: int,
    slide: pd.Series,
    path: Path,
) -> None:
    """Write a stride-cell grid as a 0/255 BigTIFF aligned with the tiling level."""
    from ml.callbacks.tiff_prediction_map_writer import _emit_mask

    # Footprints of edge tiles can reach past the slide; keep cells that start inside.
    rows = -(-int(slide["extent_y"]) // stride)
    cols = -(-int(slide["extent_x"]) // stride)
    grid = grid[:rows, :cols]
    _emit_mask(
        np.ascontiguousarray(np.where(grid, 255, 0).astype(np.uint8)),
        stride,
        stride,
        (int(slide["extent_x"]), int(slide["extent_y"])),
        0,
        path,
        (float(slide["mpp_x"]), float(slide["mpp_y"])),
    )


@with_cli_args(["+preprocessing=pni_detection"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    artifacts = config.dataset.mlflow_artifacts
    predictions = pd.read_parquet(
        download_artifacts(
            run_id=artifacts.prediction_run_id,
            artifact_path="predictions/predictions.parquet",
        ),
        columns=["slide_id", "x", "y", "pred"],
    )
    slides = pd.read_parquet(
        download_artifacts(
            run_id=artifacts.tiling_run_id, artifact_path="test_split/slides.parquet"
        )
    )
    annotation_paths = dict(config.dataset.get("pni_annotations") or {})
    nerve_index = int(config.class_indices[config.nerve_class])
    epithelium_index = int(config.class_indices[config.epithelium_class])

    all_regions = []
    totals: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir)
        (out / "regions").mkdir()
        for _, slide in slides.iterrows():
            name = Path(str(slide["path"])).name
            stride = int(slide["stride_x"])
            tile_extent = int(slide["tile_extent_x"])
            downsample = float(slide["downsample"])
            tiles = predictions[predictions["slide_id"] == slide["id"]]

            flag, nerve, epithelium = flag_tiles(
                tiles, stride, nerve_index, epithelium_index, config.radius
            )
            labels, regions = find_regions(flag, nerve, epithelium, config.min_tiles)
            records = regions.to_dict("records")
            polygons = [
                region_polygon(r["gx"], r["gy"], stride, tile_extent, downsample)
                for r in records
            ]

            write_mask(
                footprint_grid(labels, stride, tile_extent),
                stride,
                slide,
                out / "masks" / name,
            )
            features = [
                {
                    "type": "Feature",
                    "geometry": mapping(polygon),
                    "properties": {
                        "objectType": "annotation",
                        "classification": {"name": "PNI candidate"},
                        "rank": int(r["rank"]),
                        "n_tiles": int(r["n_tiles"]),
                        "n_nerve": int(r["n_nerve"]),
                        "n_epithelium": int(r["n_epithelium"]),
                    },
                }
                for r, polygon in zip(records, polygons, strict=True)
            ]
            stem = name.removesuffix(".tiff").removesuffix(".tif")
            with open(out / "regions" / f"{stem}.geojson", "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": features}, f)

            all_regions.append(
                regions.drop(columns=["gx", "gy"]).assign(
                    slide=name,
                    min_x=[p.bounds[0] for p in polygons],
                    min_y=[p.bounds[1] for p in polygons],
                    max_x=[p.bounds[2] for p in polygons],
                    max_y=[p.bounds[3] for p in polygons],
                )
            )
            mlflow.log_metric(f"regions_{stem}", len(regions))

            if name in annotation_paths:
                scores = evaluate(
                    polygons, load_qupath_annotations(annotation_paths[name])
                )
                scores["regions"] = len(regions)
                for key, value in scores.items():
                    mlflow.log_metric(f"{key}_{stem}", value)
                    totals[key] = totals.get(key, 0) + value

        pd.concat(all_regions, ignore_index=True).to_csv(
            out / "regions.csv", index=False
        )
        logger.log_artifacts(tmp_dir)

    if totals.get("pni_polygons"):
        mlflow.log_metric("pni_hit_rate", totals["pni_hit"] / totals["pni_polygons"])
    if totals.get("nerve_polygons"):
        mlflow.log_metric(
            "nerve_hit_rate", totals["nerve_hit"] / totals["nerve_polygons"]
        )
    if totals.get("regions"):
        mlflow.log_metric(
            "annotated_regions_on_pni_rate",
            totals["regions_on_pni"] / totals["regions"],
        )


if __name__ == "__main__":
    main()
