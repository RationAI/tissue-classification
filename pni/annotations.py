"""Loader for QuPath perineural invasion (PNI) annotations used as evaluation targets.

QuPath GeoJSON differs from the xOpat exports the training pipeline parses with
``ratiopath.parsers.GeoJSONParser``: the class name sits on every feature under
``properties.classification.name`` instead of in a ``presetID``-linked definition
feature, and QuPath's plain export is a bare feature array that geopandas cannot
read. This module therefore parses the JSON directly and accepts both the
FeatureCollection and the bare-array form.

Geometries are kept at level 0, like every annotation in this repository, so tile
coverage is computed with the same ``ratiopath.tiling.tile_annotations`` call that
``preprocessing.tiling`` uses.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ratiopath.tiling import tile_annotations
from shapely import Polygon, make_valid
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


NERVE = "Nerve"
PNI = "Perineural invasion"
UNCLASSIFIED = "Unclassified"


def _polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Validate a geometry and flatten it into non-empty polygons."""
    if geometry.is_empty:
        return []
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if isinstance(geometry, Polygon):
        return [geometry] if not geometry.is_empty else []
    return [
        polygon
        for part in getattr(geometry, "geoms", [])
        for polygon in _polygons(part)
    ]


def _features(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        return data["features"]
    if isinstance(data, dict) and data.get("type") == "Feature":
        return [data]
    raise ValueError("expected a GeoJSON FeatureCollection, Feature or feature array")


def load_qupath_annotations(path: str | Path) -> dict[str, list[Polygon]]:
    """Load a QuPath GeoJSON export as level-0 polygons grouped by class name.

    Features without a classification are grouped under ``UNCLASSIFIED``. Points,
    lines and other non-areal geometries are dropped.
    """
    with open(path, encoding="utf-8") as file:
        features = _features(json.load(file))

    by_class: dict[str, list[Polygon]] = defaultdict(list)
    for feature in features:
        if not feature.get("geometry"):
            continue
        properties = feature.get("properties") or {}
        classification = properties.get("classification") or {}
        class_name = classification.get("name") or UNCLASSIFIED
        by_class[class_name].extend(_polygons(shape(feature["geometry"])))

    return dict(by_class)


def load_dataset_annotations(
    annotation_paths: Mapping[str, str],
) -> dict[str, dict[str, list[Polygon]]]:
    """Load every annotation file of a dataset, keyed by slide file name.

    ``annotation_paths`` is the ``dataset.pni_annotations`` config mapping.
    """
    return {
        slide_name: load_qupath_annotations(path)
        for slide_name, path in annotation_paths.items()
    }


def tile_coverage(
    tiles: pd.DataFrame,
    slide: Mapping[str, Any],
    annotations: Mapping[str, Iterable[Polygon]],
) -> pd.DataFrame:
    """Per-tile coverage of each annotation class for one slide.

    Args:
        tiles: Tiles of one slide with level coordinates in columns ``x`` and ``y``.
        slide: The slide's row from the tiling ``slides.parquet`` (needs
            ``tile_extent_x``, ``tile_extent_y`` and ``downsample``).
        annotations: Level-0 polygons by class name, e.g. one entry of
            ``load_dataset_annotations``.

    Returns:
        ``tiles`` with ``tile_coverage_<class>`` (fraction of the whole tile) and
        ``roi_coverage_<class>`` (fraction of the central half-size ROI) columns,
        matching the column convention of ``preprocessing.tiling``.
    """
    tile_w, tile_h = int(slide["tile_extent_x"]), int(slide["tile_extent_y"])
    downsample = float(slide["downsample"])
    tile_polygon = Polygon([(0, 0), (tile_w, 0), (tile_w, tile_h), (0, tile_h)])
    cx, cy = tile_w / 2, tile_h / 2
    roi_polygon = Polygon(
        [
            (cx - tile_w / 4, cy - tile_h / 4),
            (cx + tile_w / 4, cy - tile_h / 4),
            (cx + tile_w / 4, cy + tile_h / 4),
            (cx - tile_w / 4, cy + tile_h / 4),
        ]
    )
    coordinates = tiles[["x", "y"]].to_numpy(dtype=np.int64)

    result = tiles.copy()
    for class_name, polygons in annotations.items():
        polygons = list(polygons)
        for prefix, roi in (("tile", tile_polygon), ("roi", roi_polygon)):
            column = f"{prefix}_coverage_{class_name}"
            if not polygons or len(coordinates) == 0:
                result[column] = 0.0
                continue
            result[column] = [
                intersection.area / roi.area
                for intersection in tile_annotations(
                    polygons, roi, coordinates, downsample
                )
            ]
    return result
