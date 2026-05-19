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
