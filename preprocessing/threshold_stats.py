import tempfile
import time
from pathlib import Path
from typing import cast

import duckdb
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
    print(
        f"[load_table] downloading run={run_id[:8]} path={artifact_path}", flush=True
    )
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


HISTOGRAM_BIN_EDGES = np.linspace(0.0, 1.0, 11)


def threshold_sweep(values: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, count_above) for thresholds uniformly sampled in [0, 1]."""
    sorted_vals = np.sort(values)
    thresholds = np.linspace(0.0, 1.0, n_points)
    counts_above = len(sorted_vals) - np.searchsorted(
        sorted_vals, thresholds, side="left"
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
    print(
        f"[join_inputs] loaded rows: tiling={tiling.num_rows} "
        f"tissue={tissue.num_rows} qc={qc.num_rows}",
        flush=True,
    )

    # pyarrow's Table.join silently produces incorrect results on string-keyed
    # joins at 10M+ rows (we observed 49M output from a 79M-row inner join that
    # should have matched fully). DuckDB returns the correct row count.
    t0 = time.perf_counter()
    con = duckdb.connect()
    con.register("tiling_t", tiling)
    con.register("tissue_t", tissue)
    con.register("qc_t", qc)
    joined = con.execute(
        """
        SELECT *
        FROM tiling_t
        INNER JOIN tissue_t USING (slide_id, x, y)
        INNER JOIN qc_t USING (slide_id, x, y)
        """
    ).to_arrow_table()
    print(
        f"[join_inputs] duckdb 3-way join: {joined.num_rows} rows in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return joined


def plot_threshold_sweep(
    title: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (thresholds, counts) in curves.items():
        ax.plot(
            thresholds, counts, label=label, linewidth=1.5 if label == "all" else 1.0
        )
    ax.set_xlabel("threshold")
    ax.set_ylabel("count of tiles with coverage > threshold (log)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
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
        mlflow.log_metric(
            f"{split}_tile_count_{label}", int((dominant == label).sum())
        )

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
    plot_threshold_sweep(
        f"{split} — ROI class coverage threshold sweep",
        class_sweeps,
        output_dir / "sweep_class_coverage.png",
    )
    plot_histogram(
        f"{split} — ROI class coverage histogram",
        class_hists,
        output_dir / "histogram_class_coverage.png",
    )

    annotated_mask = dominant != BACKGROUND_LABEL

    # Tissue + QC columns: scalars (global + per dominant class) + per-column figures.
    for col in (TISSUE_COLUMN, *QC_COLUMNS):
        t0 = time.perf_counter()
        values = table.column(col).to_numpy(zero_copy_only=False)
        log_scalar_stats(f"{split}_{col}", values)

        sweeps = {"all": threshold_sweep(values, n_curve_points)}
        hists: dict[str, np.ndarray] = {}
        for label in strata:
            mask = dominant == label
            if not mask.any():
                continue
            stratum_values = values[mask]
            log_scalar_stats(f"{split}_{col}_class_{label}", stratum_values)
            sweeps[label] = threshold_sweep(stratum_values, n_curve_points)
            hists[label] = histogram_counts(stratum_values)

        binary_sweeps = {"all": sweeps["all"]}
        binary_hists: dict[str, np.ndarray] = {}
        for label, mask in [
            ("annotated", annotated_mask),
            (BACKGROUND_LABEL, ~annotated_mask),
        ]:
            if not mask.any():
                continue
            stratum_values = values[mask]
            binary_sweeps[label] = threshold_sweep(stratum_values, n_curve_points)
            binary_hists[label] = histogram_counts(stratum_values)

        plot_threshold_sweep(
            f"{split} — {col} threshold sweep by dominant class",
            sweeps,
            output_dir / f"sweep_{col}.png",
        )
        plot_threshold_sweep(
            f"{split} — {col} threshold sweep (annotated vs background)",
            binary_sweeps,
            output_dir / f"sweep_binary_{col}.png",
        )
        plot_histogram(
            f"{split} — {col} histogram (annotated vs background)",
            binary_hists,
            output_dir / f"histogram_binary_{col}.png",
        )
        plot_histogram(
            f"{split} — {col} histogram by dominant class",
            hists,
            output_dir / f"histogram_{col}.png",
        )
        print(
            f"[analyze {split}] {col} done in {time.perf_counter() - t0:.1f}s",
            flush=True,
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
            table = join_inputs(
                tiling_run_id=config.dataset.mlflow_artifacts.tiling_run_id,
                tiling_artifact=config.tiling_tiles_artifact_template.format(
                    split=split
                ),
                tissue_run_id=config.dataset.mlflow_artifacts.tissue_stats_run_id,
                tissue_artifact=config.tissue_tiles_artifact_template.format(
                    split=split
                ),
                qc_run_id=config.dataset.mlflow_artifacts.qc_stats_run_id,
                qc_artifact=config.qc_tiles_artifact_template.format(split=split),
                class_names=class_names,
            )

            split_dir = Path(tmp_dir) / split
            split_dir.mkdir()
            analyze(split, table, class_names, config.survival_curve_points, split_dir)

        mlflow.log_artifacts(tmp_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    main()
