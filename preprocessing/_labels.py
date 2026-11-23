"""Shared helpers for deriving tile labels from roi_coverage_* columns."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def compute_label_and_tissue_prop(
    roi_data: Mapping[str, Any],
    roi_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (label, tissue_prop) from roi_coverage_* columns.

    label = argmax across roi_cols (with ``roi_coverage_`` prefix stripped),
    falling back to ``"background"`` whenever all coverages are zero.
    tissue_prop = sum across roi_cols.
    """
    roi_df = pd.DataFrame({col: roi_data[col] for col in roi_cols})
    tp = roi_df.sum(axis=1).to_numpy()
    lbl = roi_df.idxmax(axis=1).str.removeprefix("roi_coverage_").to_numpy()
    lbl[tp == 0] = "background"
    return lbl, tp


ROI_PREFIX = "roi_coverage_"
REASON_OK = "ok"
REASON_LOW_COVERAGE = "low_coverage"
REASON_MULTI_CLASS = "multi_class"
REASON_BELOW_THRESHOLD = "below_threshold"


def assign_tile_labels(
    roi: pd.DataFrame,
    thresholds: Mapping[str, float],
    tissue_prop_min: float,
) -> pd.DataFrame:
    """Training tile-labelling rule on ``roi_coverage_<class>`` columns.

    A tile gets the label of its dominant class only if
    1. the summed annotated ROI coverage is >= ``tissue_prop_min``,
    2. its ROI touches at most one class, and
    3. the dominant class coverage is >= that class's threshold.

    Returns a frame aligned with ``roi`` with ``label`` (None when dropped) and
    ``reason`` (``ok`` / ``low_coverage`` / ``multi_class`` / ``below_threshold``,
    checked in that order).
    """
    roi_cols = [c for c in roi.columns if c.startswith(ROI_PREFIX)]
    if not roi_cols:
        raise ValueError(f"no {ROI_PREFIX}* columns; cannot label")
    names = np.array([c.removeprefix(ROI_PREFIX) for c in roi_cols], dtype=object)
    missing = set(names) - set(thresholds)
    if missing:
        raise ValueError(
            f"thresholds missing entries for classes present in data: {sorted(missing)}"
        )

    cov = roi[roi_cols].to_numpy(dtype=np.float64)
    tissue_prop = cov.sum(axis=1)
    nonzero = (cov > 0).sum(axis=1)
    dominant = cov.argmax(axis=1)
    dominant_value = cov.max(axis=1)
    threshold = np.array([thresholds[n] for n in names], dtype=np.float64)[dominant]

    reason = np.full(len(cov), REASON_OK, dtype=object)
    reason[dominant_value < threshold] = REASON_BELOW_THRESHOLD
    reason[nonzero > 1] = REASON_MULTI_CLASS
    reason[tissue_prop < tissue_prop_min] = REASON_LOW_COVERAGE
    label = np.full(len(cov), None, dtype=object)
    ok = reason == REASON_OK
    label[ok] = names[dominant[ok]]
    return pd.DataFrame({"label": label, "reason": reason}, index=roi.index)
