"""Summary tables for perineural invasion (PNI) detection.

Every function returns rows (lists of dicts) for one slide, so callers concatenate
them over slides and write one CSV per table:

- ``class_distribution``: predicted tissue classes over all tissue tiles.
- ``class_in_annotations``: predicted classes of tiles inside the pathologist's
  Nerve and PNI outlines and outside both, the check that the tissue model sees
  nerve and epithelium where the outlines say they are.
- ``region_labels`` / ``detection_summary``: what each candidate region touches,
  and per slide how many annotated outlines the candidates reach.
- ``ranking``: precision and PNI recall of the ``k`` largest regions, since size
  rank is how the candidates are meant to be read.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from pni.annotations import NERVE, PNI, tile_coverage


OUTSIDE = "outside annotations"


def class_distribution(
    tiles: pd.DataFrame, class_names: Sequence[str], slide: str
) -> list[dict[str, Any]]:
    counts = np.bincount(tiles["pred"].to_numpy(), minlength=len(class_names))
    total = int(counts.sum())
    return [
        {
            "slide": slide,
            "class": name,
            "tiles": int(count),
            "share": count / total if total else 0.0,
        }
        for name, count in zip(class_names, counts, strict=True)
    ]


def class_in_annotations(
    tiles: pd.DataFrame,
    slide_row: Mapping[str, Any],
    annotations: Mapping[str, Sequence[Any]],
    class_names: Sequence[str],
    slide: str,
    min_roi: float = 0.5,
) -> list[dict[str, Any]]:
    """Predicted classes of tiles whose central ROI lies mostly in an outline."""
    covered = tile_coverage(
        tiles, slide_row, {k: annotations.get(k, []) for k in (NERVE, PNI)}
    )
    groups = {
        NERVE: covered[f"roi_coverage_{NERVE}"] > min_roi,
        PNI: covered[f"roi_coverage_{PNI}"] > min_roi,
        OUTSIDE: (covered[f"tile_coverage_{NERVE}"] == 0)
        & (covered[f"tile_coverage_{PNI}"] == 0),
    }
    rows = []
    for group, selected in groups.items():
        for row in class_distribution(covered[selected], class_names, slide):
            rows.append({"inside": group, **row})
    return rows


def region_labels(
    polygons: Sequence[BaseGeometry], annotations: Mapping[str, Sequence[Any]]
) -> list[str]:
    """``PNI`` if a region touches a PNI outline, else ``Nerve``, else ``other``."""
    trees = {
        name: STRtree(list(annotations[name]))
        for name in (PNI, NERVE)
        if annotations.get(name)
    }
    labels = []
    for polygon in polygons:
        label = "other"
        for name in (PNI, NERVE):
            if name in trees and len(trees[name].query(polygon, "intersects")):
                label = "PNI" if name == PNI else "Nerve"
                break
        labels.append(label)
    return labels


def _touched(polygons: Sequence[BaseGeometry], targets: Sequence[Any]) -> int:
    if not polygons or not targets:
        return 0
    tree = STRtree(list(polygons))
    return sum(bool(len(tree.query(t, "intersects"))) for t in targets)


def detection_summary(
    regions: pd.DataFrame,
    polygons: Sequence[BaseGeometry],
    annotations: Mapping[str, Sequence[Any]] | None,
    slide: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "slide": slide,
        "regions": len(regions),
        "tiles_in_regions": int(regions["n_tiles"].sum()),
        "median_region_tiles": float(regions["n_tiles"].median())
        if len(regions)
        else 0.0,
        "max_region_tiles": int(regions["n_tiles"].max()) if len(regions) else 0,
    }
    if annotations is None:
        return row
    labels = region_labels(polygons, annotations)
    pni, nerve = annotations.get(PNI, []), annotations.get(NERVE, [])
    pni_hit, nerve_hit = _touched(polygons, pni), _touched(polygons, nerve)
    row.update(
        {
            "pni_outlines": len(pni),
            "pni_touched": pni_hit,
            "pni_recall": pni_hit / len(pni) if pni else np.nan,
            "nerve_outlines": len(nerve),
            "nerve_touched": nerve_hit,
            "nerve_touched_rate": nerve_hit / len(nerve) if nerve else np.nan,
            "regions_on_pni": labels.count("PNI"),
            "regions_on_nerve": labels.count("Nerve"),
            "regions_elsewhere": labels.count("other"),
            "region_precision": labels.count("PNI") / len(labels) if labels else np.nan,
        }
    )
    return row


def ranking(
    regions: pd.DataFrame,
    polygons: Sequence[BaseGeometry],
    annotations: Mapping[str, Sequence[Any]],
    ks: Sequence[int],
    slide: str,
) -> list[dict[str, Any]]:
    """Precision and PNI recall of the ``k`` largest regions (``rank <= k``).

    ``regions`` and ``polygons`` must be in rank order, as ``find_regions`` and
    the detection stage produce them.
    """
    labels = region_labels(polygons, annotations)
    pni = annotations.get(PNI, [])
    rows = []
    for k in ks:
        top = min(k, len(polygons))
        hits = _touched(polygons[:top], pni)
        rows.append(
            {
                "slide": slide,
                "k": k,
                "regions": top,
                "regions_on_pni": labels[:top].count("PNI"),
                "precision": labels[:top].count("PNI") / top if top else np.nan,
                "pni_touched": hits,
                "pni_recall": hits / len(pni) if pni else np.nan,
            }
        )
    return rows
