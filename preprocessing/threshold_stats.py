import tempfile
from pathlib import Path
from typing import cast

import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


SCALAR_QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
QC_COLUMNS = ("roi_residual_coverage", "roi_folding_coverage", "roi_blur_coverage")
TISSUE_COLUMN = "roi_tissue_coverage"
BACKGROUND_LABEL = "background"


def load_table(run_id: str, artifact_path: str, columns: list[str]) -> pa.Table:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    return pq.read_table(local_path, columns=columns)


def quantile_metric_key(quantile: float) -> str:
    return f"q{round(quantile * 1000):04d}"


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    qs = np.quantile(values, SCALAR_QUANTILES)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        **{
            quantile_metric_key(q): float(v)
            for q, v in zip(SCALAR_QUANTILES, qs, strict=True)
        },
    }


def survival_curve(values: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, fraction_remaining) sampled at empirical quantiles.

    Empirical-quantile sampling concentrates points where data is dense, giving
    a more useful curve for threshold selection than uniform [0, 1] sampling.
    """
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    sample_qs = np.linspace(0.0, 1.0, n_points)
    thresholds = np.quantile(sorted_vals, sample_qs)
    counts_remaining = n - np.searchsorted(sorted_vals, thresholds, side="left")
    return thresholds, counts_remaining / n


def compute_dominant_class(table: pa.Table, class_names: list[str]) -> np.ndarray:
    matrix = np.stack(
        [
            table.column(f"roi_coverage_{cls}").to_numpy(zero_copy_only=False)
            for cls in class_names
        ],
        axis=1,
    )
    has_class = matrix.sum(axis=1) > 0
    dominant_idx = matrix.argmax(axis=1)
    labels = np.array([*class_names, BACKGROUND_LABEL], dtype=object)
    return labels[np.where(has_class, dominant_idx, len(class_names))]


def join_inputs(
    tiling_run_id: str,
    tiling_artifact: str,
    tissue_run_id: str,
    tissue_artifact: str,
    qc_run_id: str,
    qc_artifact: str,
    class_names: list[str],
) -> pa.Table:
    tiling_columns = ["slide_id", "x", "y"] + [
        f"roi_coverage_{cls}" for cls in class_names
    ]
    tissue_columns = ["slide_id", "x", "y", TISSUE_COLUMN]
    qc_columns = ["slide_id", "x", "y", *QC_COLUMNS]

    tiling = load_table(tiling_run_id, tiling_artifact, tiling_columns)
    tissue = load_table(tissue_run_id, tissue_artifact, tissue_columns)
    qc = load_table(qc_run_id, qc_artifact, qc_columns)

    keys = ["slide_id", "x", "y"]
    joined = tiling.join(tissue, keys=keys, join_type="inner")
    joined = joined.join(qc, keys=keys, join_type="inner")
    return joined


def plot_survival(
    title: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (thresholds, fractions) in curves.items():
        ax.plot(
            thresholds, fractions, label=label, linewidth=1.5 if label == "all" else 1.0
        )
    ax.set_xlabel("threshold")
    ax.set_ylabel("fraction of tiles with coverage >= threshold")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def log_scalar_stats(prefix: str, values: np.ndarray) -> None:
    for key, value in scalar_stats(values).items():
        mlflow.log_metric(f"{prefix}_{key}", value)


def analyze(
    table: pa.Table,
    class_names: list[str],
    n_curve_points: int,
    output_dir: Path,
) -> None:
    dominant = compute_dominant_class(table, class_names)
    strata = [*class_names, BACKGROUND_LABEL]

    mlflow.log_metric("tile_count", len(table))
    for label in strata:
        mlflow.log_metric(f"tile_count_{label}", int((dominant == label).sum()))

    # Class coverage columns: scalars (global only) + one combined survival figure.
    class_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls in class_names:
        values = table.column(f"roi_coverage_{cls}").to_numpy(zero_copy_only=False)
        log_scalar_stats(f"roi_coverage_{cls}", values)
        class_curves[cls] = survival_curve(values, n_curve_points)
    plot_survival(
        "ROI class coverage — survival curves",
        class_curves,
        output_dir / "survival_class_coverage.png",
    )

    # Tissue + QC columns: scalars (global + per dominant class) + per-column figure.
    for col in (TISSUE_COLUMN, *QC_COLUMNS):
        values = table.column(col).to_numpy(zero_copy_only=False)
        log_scalar_stats(col, values)

        curves = {"all": survival_curve(values, n_curve_points)}
        for label in strata:
            mask = dominant == label
            if not mask.any():
                continue
            stratum_values = values[mask]
            log_scalar_stats(f"{col}_class_{label}", stratum_values)
            curves[label] = survival_curve(stratum_values, n_curve_points)

        plot_survival(
            f"{col} — survival curves by dominant class",
            curves,
            output_dir / f"survival_{col}.png",
        )


@with_cli_args(["+preprocessing=threshold_stats"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    class_mapping = cast(
        "dict[str, list[str]]",
        OmegaConf.to_container(config.class_mapping, resolve=True),
    )
    class_names = list(class_mapping.keys())

    table = join_inputs(
        tiling_run_id=config.dataset.mlflow_artifacts.tiling_run_id,
        tiling_artifact=config.tiling_tiles_artifact,
        tissue_run_id=config.dataset.mlflow_artifacts.tissue_stats_run_id,
        tissue_artifact=config.tissue_tiles_artifact,
        qc_run_id=config.dataset.mlflow_artifacts.qc_stats_run_id,
        qc_artifact=config.qc_tiles_artifact,
        class_names=class_names,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        analyze(table, class_names, config.survival_curve_points, output_dir)
        mlflow.log_artifacts(tmp_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    main()
