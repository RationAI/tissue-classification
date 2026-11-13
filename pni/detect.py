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
- ``nerve_masks/<slide file name>`` and ``nerve_regions/<slide stem>.geojson``:
  the nerve of each region: cells of the prediction run's tissue map that are
  Nerve and lie under the region's Nerve tiles. The pathologist's PNI outlines
  enclose the invaded nerve, so this is the shape shown for review. It is taken
  from the same fused map the report shows as the tissue layer, so it sits on
  that map's Nerve cells exactly; the full regions (both sides of the contact,
  whole tile footprints) stay for comparison. The nerve mask encodes the
  candidate's display tier: by position among the slide's shown candidates
  (largest region first), the first ``display_tiers[0]`` are 255, those up to
  ``display_tiers[1]`` 170 and the rest 85, so the report can tell the
  candidates most worth a look from the rest
- ``regions.csv``: all regions of all slides
- ``stats/*.csv``: summary tables from ``pni.stats``
- ``annotations/<slide file name>``: on annotated slides, the QuPath Nerve (128)
  and PNI (255) polygons rasterised at ``annotation_mpp``, for display next to
  the candidates

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

from pni import stats
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


def cell_polygon(
    gx: np.ndarray,
    gy: np.ndarray,
    stride: int,
    downsample: float,
) -> BaseGeometry:
    """Level-0 union of stride grid cells."""
    return unary_union(
        [
            box(
                x * stride * downsample,
                y * stride * downsample,
                (x + 1) * stride * downsample,
                (y + 1) * stride * downsample,
            )
            for x, y in zip(gx, gy, strict=True)
        ]
    )


def nerve_cells(
    labels: np.ndarray,
    nerve: np.ndarray,
    map_nerve: np.ndarray,
    stride: int,
    tile_extent: int,
) -> np.ndarray:
    """Region rank per stride cell of the nerve shape, 0 elsewhere.

    A cell belongs to a region's nerve when the fused tissue map calls it Nerve
    and the footprint of one of the region's Nerve tiles covers it. Per-tile
    argmax and the fused map can disagree (the map sums the probabilities of all
    tiles over a cell), so a Nerve tile whose cells the map gives to another class
    adds nothing.

    Args:
        labels: Region rank per tile grid cell, from ``find_regions``.
        nerve: Tiles predicted Nerve.
        map_nerve: Cells the fused tissue map calls Nerve, on the same grid.
        stride: Tiling stride in level pixels.
        tile_extent: Tile size in level pixels, a multiple of ``stride``.
    """
    span = tile_extent // stride
    ranks = np.where(nerve, labels, 0)
    shape = (ranks.shape[0] + span - 1, ranks.shape[1] + span - 1)
    out = np.zeros(shape, dtype=np.int32)
    for dy in range(span):
        for dx in range(span):
            window = out[dy : dy + ranks.shape[0], dx : dx + ranks.shape[1]]
            # Overlapping footprints of two regions: keep the better (lower) rank.
            take = (ranks > 0) & ((window == 0) | (ranks < window))
            window[take] = ranks[take]
    rows = min(shape[0], map_nerve.shape[0])
    cols = min(shape[1], map_nerve.shape[1])
    out = out[:rows, :cols]
    out[~map_nerve[:rows, :cols]] = 0
    return out


def read_map_nerve(path: str, nerve_value: int, stride: int) -> np.ndarray:
    """Cells of a tissue map TIFF that hold ``nerve_value``.

    The map is constant over each stride cell, so one pixel per cell suffices.
    """
    import pyvips

    image = pyvips.Image.new_from_file(path, access="sequential")
    cells = image.extract_band(0).subsample(stride, stride)
    data = np.frombuffer(cells.write_to_memory(), dtype=np.uint8)
    return data.reshape(cells.height, cells.width) == nerve_value


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


#: Annotation mask values, spread over uint8 like the prediction maps so one
#: colormap convention reads both: background 0, then 255 * (i + 1) / n.
ANNOTATION_VALUES = {NERVE: 128, PNI: 255}


def write_annotation_mask(
    annotations: dict[str, list[Any]],
    slide_path: str,
    mpp: float,
    path: Path,
) -> None:
    """Rasterise QuPath Nerve / PNI polygons at the slide level closest to `mpp`.

    Drawn like ``preprocessing.annotation_masks``: exteriors filled with the class
    value, holes cleared, written as a BigTIFF carrying that level's resolution.
    """
    import pyvips
    from PIL import Image, ImageDraw
    from rationai.masks import write_big_tiff
    from ratiopath.openslide import OpenSlide

    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        mpp_x, mpp_y = slide.slide_resolution(level)
        base_width, base_height = slide.level_dimensions[0]
        width, height = slide.level_dimensions[level]
    scale_x, scale_y = width / base_width, height / base_height

    mask = Image.new("L", size=(width, height), color=0)
    draw = ImageDraw.Draw(mask)
    for class_name, value in ANNOTATION_VALUES.items():
        for polygon in annotations.get(class_name, []):
            draw.polygon(
                [(x * scale_x, y * scale_y) for x, y in polygon.exterior.coords],
                fill=value,
            )
            for interior in polygon.interiors:
                draw.polygon(
                    [(x * scale_x, y * scale_y) for x, y in interior.coords], fill=0
                )
    del draw

    path.parent.mkdir(parents=True, exist_ok=True)
    write_big_tiff(
        image=pyvips.Image.new_from_memory(
            data=mask.tobytes(), width=width, height=height, bands=1, format="uchar"
        ),
        path=path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )


#: Nerve mask value of each display tier, best first; background is 0.
TIER_VALUES = (255, 170, 85)


def display_tiers(n: int, bounds: list[int]) -> np.ndarray:
    """Tier index (0 = best) of the first ``n`` candidates in display order."""
    return np.searchsorted(np.asarray(bounds), np.arange(n), side="right")


def write_mask(
    grid: np.ndarray,
    cell: int,
    slide: pd.Series,
    path: Path,
) -> None:
    """Write a grid of ``cell``-pixel cells as a uint8 BigTIFF on the tiling level.

    Boolean grids are written as 0/255, integer grids as they are.
    """
    from ml.callbacks.tiff_prediction_map_writer import _emit_mask

    # Footprints of edge tiles can reach past the slide; keep cells that start inside.
    rows = -(-int(slide["extent_y"]) // cell)
    cols = -(-int(slide["extent_x"]) // cell)
    grid = grid[:rows, :cols]
    _emit_mask(
        np.ascontiguousarray(
            np.where(grid, 255, 0).astype(np.uint8)
            if grid.dtype == bool
            else grid.astype(np.uint8)
        ),
        cell,
        cell,
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
    class_names = sorted(config.class_indices, key=lambda c: config.class_indices[c])
    # Value of Nerve in the prediction maps: round(255 * (i + 1) / n), as written
    # by ml.callbacks.tiff_prediction_map_writer.
    from ml.callbacks.tiff_prediction_map_writer import _spread_lut

    nerve_value = int(_spread_lut(len(class_names))[nerve_index])

    all_regions = []
    tables: dict[str, list[dict[str, Any]]] = {
        "class_distribution": [],
        "class_in_annotations": [],
        "detection": [],
        "ranking": [],
    }
    shapes = {"region": "regions", "nerve": "nerve_regions"}
    totals: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir)
        for directory in shapes.values():
            (out / directory).mkdir()
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
            # Nerve of each region, on the fused tissue map's cells. A region has
            # none when it lacks Nerve tiles (its nerve partner is within the
            # radius but not 8-connected) or when the map gives their cells to
            # other classes; such regions are left out of the nerve views.
            map_path = download_artifacts(
                run_id=artifacts.prediction_run_id,
                artifact_path=f"prediction_maps_tiff/pred/{name}",
                dst_path=tmp_dir + "/maps",
            )
            cells = nerve_cells(
                labels,
                nerve,
                read_map_nerve(map_path, nerve_value, stride),
                stride,
                tile_extent,
            )
            Path(map_path).unlink()
            cell_y, cell_x = np.nonzero(cells)
            cell_rank = cells[cell_y, cell_x]
            nerve_records, nerve_polygons = [], []
            for r in records:
                own = cell_rank == r["rank"]
                if own.any():
                    nerve_records.append({**r, "n_nerve_cells": int(own.sum())})
                    nerve_polygons.append(
                        cell_polygon(cell_x[own], cell_y[own], stride, downsample)
                    )
            nerve_ranks = [r["rank"] for r in nerve_records]
            tiers = display_tiers(len(nerve_records), list(config.display_tiers))
            tier_value = np.zeros(int(cells.max()) + 1, dtype=np.uint8)
            for r, tier in zip(nerve_records, tiers, strict=True):
                r["tier"] = int(tier) + 1
                tier_value[r["rank"]] = TIER_VALUES[tier]
            nerve_regions = regions[regions["rank"].isin(nerve_ranks)].reset_index(
                drop=True
            )

            write_mask(
                footprint_grid(labels, stride, tile_extent),
                stride,
                slide,
                out / "masks" / name,
            )
            write_mask(tier_value[cells], stride, slide, out / "nerve_masks" / name)
            stem = name.removesuffix(".tiff").removesuffix(".tif")
            for shape, shape_records, shape_polygons in (
                ("region", records, polygons),
                ("nerve", nerve_records, nerve_polygons),
            ):
                features = [
                    {
                        "type": "Feature",
                        "geometry": mapping(polygon),
                        "properties": {
                            "objectType": "annotation",
                            "classification": {
                                "name": f"PNI candidate tier {r['tier']}"
                                if "tier" in r
                                else "PNI candidate"
                            },
                            "rank": int(r["rank"]),
                            "n_tiles": int(r["n_tiles"]),
                            "n_nerve": int(r["n_nerve"]),
                            "n_epithelium": int(r["n_epithelium"]),
                        },
                    }
                    for r, polygon in zip(shape_records, shape_polygons, strict=True)
                ]
                path = out / shapes[shape] / f"{stem}.geojson"
                with open(path, "w", encoding="utf-8") as f:
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
            mlflow.log_metric(f"nerve_regions_{stem}", len(nerve_regions))
            # Share of the Nerve tiles' footprint that the fused map calls Nerve.
            footprint = nerve_cells(
                labels, nerve, np.ones(cells.shape, dtype=bool), stride, tile_extent
            )
            mlflow.log_metric(
                f"nerve_map_agreement_{stem}",
                (cells > 0).sum() / max(int((footprint > 0).sum()), 1),
            )
            tables["class_distribution"] += stats.class_distribution(
                tiles, class_names, name
            )

            annotations = None
            if name in annotation_paths:
                annotations = load_qupath_annotations(annotation_paths[name])
                tables["class_in_annotations"] += stats.class_in_annotations(
                    tiles,
                    {str(k): v for k, v in slide.items()},
                    annotations,
                    class_names,
                    name,
                )
                for shape, shape_regions, shape_polygons in (
                    ("region", regions, polygons),
                    ("nerve", nerve_regions, nerve_polygons),
                ):
                    tables["ranking"] += [
                        {"shape": shape, **row}
                        for row in stats.ranking(
                            shape_regions,
                            shape_polygons,
                            annotations,
                            config.ranking_ks,
                            name,
                        )
                    ]
                write_annotation_mask(
                    annotations,
                    str(slide["path"]),
                    config.annotation_mpp,
                    out / "annotations" / name,
                )
                for prefix, shape_regions, shape_polygons in (
                    ("", regions, polygons),
                    ("nerve_shape_", nerve_regions, nerve_polygons),
                ):
                    scores = evaluate(shape_polygons, annotations)
                    scores["regions"] = len(shape_regions)
                    for key, value in scores.items():
                        mlflow.log_metric(f"{prefix}{key}_{stem}", value)
                        totals[prefix + key] = totals.get(prefix + key, 0) + value

            for shape, shape_regions, shape_polygons in (
                ("region", regions, polygons),
                ("nerve", nerve_regions, nerve_polygons),
            ):
                tables["detection"].append(
                    {
                        "shape": shape,
                        **stats.detection_summary(
                            shape_regions, shape_polygons, annotations, name
                        ),
                    }
                )

        pd.concat(all_regions, ignore_index=True).to_csv(
            out / "regions.csv", index=False
        )
        (out / "stats").mkdir()
        for table, rows in tables.items():
            pd.DataFrame(rows).to_csv(out / "stats" / f"{table}.csv", index=False)
        logger.log_artifacts(tmp_dir)

    for prefix in ("", "nerve_shape_"):
        if totals.get(f"{prefix}pni_polygons"):
            mlflow.log_metric(
                f"{prefix}pni_hit_rate",
                totals[f"{prefix}pni_hit"] / totals[f"{prefix}pni_polygons"],
            )
        if totals.get(f"{prefix}nerve_polygons"):
            mlflow.log_metric(
                f"{prefix}nerve_hit_rate",
                totals[f"{prefix}nerve_hit"] / totals[f"{prefix}nerve_polygons"],
            )
        if totals.get(f"{prefix}regions"):
            mlflow.log_metric(
                f"{prefix}annotated_regions_on_pni_rate",
                totals[f"{prefix}regions_on_pni"] / totals[f"{prefix}regions"],
            )


if __name__ == "__main__":
    main()
