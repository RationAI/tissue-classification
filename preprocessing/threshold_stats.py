import tempfile
import time
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
TISSUE_COLUMN = "roi_tissue_coverage"
BACKGROUND_LABEL = "background"


def load_table(run_id: str, artifact_path: str, columns: list[str]) -> pa.Table:
    print(f"[load_table] downloading run={run_id[:8]} path={artifact_path}", flush=True)
    t0 = time.perf_counter()
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    size_mb = Path(local_path).stat().st_size / 1024**2
    print(
        f"[load_table] got {size_mb:.1f} MB in {time.perf_counter() - t0:.1f}s "
        f"-> {local_path}",
        flush=True,
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


HISTOGRAM_BIN_EDGES = np.linspace(0.0, 1.0, 21)


def threshold_sweep(values: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, count_strictly_above) for thresholds uniformly sampled in [0, 1]."""
    sorted_vals = np.sort(values)
    thresholds = np.linspace(0.0, 1.0, n_points)
    # side="right": count elements strictly > threshold, so tiles with coverage==1.0
    # are not counted at threshold==1.0, matching the axis label.
    counts_above = len(sorted_vals) - np.searchsorted(
        sorted_vals, thresholds, side="right"
    )
    return thresholds, counts_above


def histogram_counts(values: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=HISTOGRAM_BIN_EDGES)
    return counts


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


def load_filter_tiles(
    run_id: str,
    artifact: str,
    class_names: list[str],
) -> pa.Table:
    columns = ["slide_id", "x", "y", TISSUE_COLUMN] + [
        f"roi_coverage_{cls}" for cls in class_names
    ]
    return load_table(run_id, artifact, columns)



def plot_class_coverage_sweep(
    title: str,
    thresholds: np.ndarray,
    counts: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, counts, linewidth=1.5)
    ax.set_xlabel("threshold")
    ax.set_ylabel("count of tiles with coverage > threshold")
    ax.set_title(title)
    nonzero = np.where(counts > 0)[0]
    x_max = float(thresholds[nonzero[-1]]) if len(nonzero) else 1.0
    ax.set_xlim(0, min(x_max + 0.02, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_histogram(
    title: str,
    series: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_centers = (HISTOGRAM_BIN_EDGES[:-1] + HISTOGRAM_BIN_EDGES[1:]) / 2
    bin_width = HISTOGRAM_BIN_EDGES[1] - HISTOGRAM_BIN_EDGES[0]

    if len(series) == 1:
        label, counts = next(iter(series.items()))
        ax.bar(bin_centers, counts, width=bin_width * 0.9, label=label)
    else:
        # Stacked bars per dominant class.
        bottom = np.zeros(len(bin_centers))
        for label, counts in series.items():
            ax.bar(
                bin_centers, counts, width=bin_width * 0.9, bottom=bottom, label=label
            )
            bottom = bottom + counts

    ax.set_xlabel("coverage bucket")
    ax.set_ylabel("tile count (log)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def log_scalar_stats(metric_prefix: str, values: np.ndarray) -> None:
    for key, value in scalar_stats(values).items():
        mlflow.log_metric(f"{metric_prefix}_{key}", value)


def analyze(
    split: str,
    table: pa.Table,
    class_names: list[str],
    n_curve_points: int,
    output_dir: Path,
) -> None:
    print(f"[analyze {split}] computing dominant class...", flush=True)
    t0 = time.perf_counter()
    dominant = compute_dominant_class(table, class_names)
    print(
        f"[analyze {split}] dominant class done in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    strata = [*class_names, BACKGROUND_LABEL]

    mlflow.log_metric(f"{split}_tile_count", len(table))
    for label in strata:
        mlflow.log_metric(f"{split}_tile_count_{label}", int((dominant == label).sum()))

    # Class coverage columns: scalars (global) + one combined sweep + one combined histogram.
    class_sweeps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    class_hists: dict[str, np.ndarray] = {}
    for cls in class_names:
        t0 = time.perf_counter()
        values = table.column(f"roi_coverage_{cls}").to_numpy(zero_copy_only=False)
        log_scalar_stats(f"{split}_roi_coverage_{cls}", values)
        class_sweeps[cls] = threshold_sweep(values, n_curve_points)
        class_hists[cls] = histogram_counts(values)
        print(
            f"[analyze {split}] roi_coverage_{cls} done in "
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    for cls, (thresholds, counts) in class_sweeps.items():
        plot_class_coverage_sweep(
            f"{split} — ROI coverage: {cls}",
            thresholds,
            counts,
            output_dir / f"sweep_class_coverage_{cls}.png",
        )
    plot_histogram(
        f"{split} — ROI class coverage histogram",
        class_hists,
        output_dir / "histogram_class_coverage.png",
    )

    annotated_mask = dominant != BACKGROUND_LABEL

    # Tissue column: per-class scalars + per-class sweep + binary and per-class histograms.
    col = TISSUE_COLUMN
    t0 = time.perf_counter()
    values = table.column(col).to_numpy(zero_copy_only=False)
    log_scalar_stats(f"{split}_{col}", values)
    tissue_hists: dict[str, np.ndarray] = {}
    tissue_binary_hists: dict[str, np.ndarray] = {}
    for label in strata:
        mask = dominant == label
        if not mask.any():
            continue
        stratum_values = values[mask]
        log_scalar_stats(f"{split}_{col}_class_{label}", stratum_values)
        tissue_hists[label] = histogram_counts(stratum_values)
        thresholds, counts = threshold_sweep(stratum_values, n_curve_points)
        plot_class_coverage_sweep(
            f"{split} — tissue coverage: {label}",
            thresholds,
            counts,
            output_dir / f"sweep_{col}_{label}.png",
        )
    for label, mask in [
        ("annotated", annotated_mask),
        (BACKGROUND_LABEL, ~annotated_mask),
    ]:
        if not mask.any():
            continue
        tissue_binary_hists[label] = histogram_counts(values[mask])
    plot_histogram(
        f"{split} — {col} histogram by dominant class",
        tissue_hists,
        output_dir / f"histogram_{col}.png",
    )
    plot_histogram(
        f"{split} — {col} histogram (annotated vs background)",
        tissue_binary_hists,
        output_dir / f"histogram_binary_{col}.png",
    )
    print(
        f"[analyze {split}] {col} done in {time.perf_counter() - t0:.1f}s", flush=True
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split in ("train", "test"):
            table = load_filter_tiles(
                run_id=config.dataset.mlflow_artifacts.filter_tiles_run_id,
                artifact=config.filter_tiles_artifact_template.format(split=split),
                class_names=class_names,
            )

            split_dir = Path(tmp_dir) / split
            split_dir.mkdir()
            analyze(split, table, class_names, config.survival_curve_points, split_dir)

        mlflow.log_artifacts(tmp_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    main()
