"""Shared annotation parsing: label names -> validated polygons.

Used by the training tiling / annotation-mask steps and by the standalone
evaluation. Supported formats:

- GeoJSON from xOpat: label in ``properties.meta.category.value``
  (``label_key="meta_category_value"``, the training data)
- GeoJSON from QuPath: label in ``properties.classification.name``
  (``label_key="classification_name"``, e.g. the Graz annotations)
- Darwin V7 JSON: label in ``name``
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ratiopath.parsers import Darwin7JSONParser, GeoJSONParser
from shapely import Polygon, make_valid


if TYPE_CHECKING:
    from pathlib import Path

    from shapely.geometry.base import BaseGeometry


XOPAT_LABEL_KEY = "meta_category_value"
QUPATH_LABEL_KEY = "classification_name"
# Nested ``properties`` path of each label key in a GeoJSON feature.
_LABEL_KEY_PATHS = {
    XOPAT_LABEL_KEY: ("meta", "category", "value"),
    QUPATH_LABEL_KEY: ("classification", "name"),
}


def open_annotations(path: Path) -> GeoJSONParser | Darwin7JSONParser | None:
    """Parser for a GeoJSON or Darwin V7 JSON file; None for unsupported files."""
    name = str(path)
    if name.endswith((".geo.json", ".geojson")):
        return GeoJSONParser(path)
    if name.endswith(".json"):
        return Darwin7JSONParser(path)
    return None


def label_pattern(names: list[str]) -> str:
    """Case-insensitive exact-match regex for any of ``names``."""
    return f"(?i)^({'|'.join(re.escape(name) for name in names)})$"


def get_validated_polygons(
    parser: GeoJSONParser | Darwin7JSONParser,
    original_names: list[str],
    label_key: str = XOPAT_LABEL_KEY,
) -> list[BaseGeometry]:
    """Valid polygons whose label is one of ``original_names`` (case-insensitive).

    Empty geometries are dropped, invalid ones repaired with ``make_valid``, and
    multi-part results flattened into individual polygons.
    """
    if not original_names:
        return []

    pattern = label_pattern(original_names)
    if isinstance(parser, GeoJSONParser):
        raw_polygons = parser.get_polygons(**{label_key: pattern})
    else:
        raw_polygons = parser.get_polygons(name=pattern)

    valid_polygons: list[BaseGeometry] = []
    for poly in raw_polygons:
        if poly is None or poly.is_empty:
            continue

        validated = make_valid(poly) if not poly.is_valid else poly

        if isinstance(validated, Polygon):
            valid_polygons.append(validated)
        elif hasattr(validated, "geoms"):
            valid_polygons.extend(g for g in validated.geoms if isinstance(g, Polygon))

    return valid_polygons


def geojson_feature_label(
    properties: dict[str, Any] | None, label_key: str
) -> str | None:
    """Label of a GeoJSON feature under ``label_key`` (None if it has none)."""
    value: Any = properties or {}
    for key in _LABEL_KEY_PATHS[label_key]:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return str(value) if value else None
