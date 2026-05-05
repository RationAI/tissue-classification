import tempfile
import time
from pathlib import Path

import hydra
import mlflow
import mlflow.artifacts
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def filter_split(
    tiling_run_id: str,
    tissue_stats_run_id: str,
    tissue_stats_artifact_path: str,
    tissue_column: str,
    split_name: str,
    output_path: Path,
) -> dict[str, int]:
    """Drop tiles with no annotation coverage and no tissue coverage.

    Uses PyArrow predicate pushdown so the full tiles parquet is never loaded into
    memory — only rows passing the annotation filter are materialised. The tissue
    coverage table is then joined in-memory to drop tiles outside tissue and to
    carry through the per-tile coverage values into the output.
    """
    tiles_local = mlflow.artifacts.download_artifacts(
        run_id=tiling_run_id, artifact_path=f"{split_name}_split/tiles.parquet"
    )
    tiles_ds = pads.dataset(tiles_local, format="parquet")
    original_count = tiles_ds.count_rows()

    ann_cols = [f.name for f in tiles_ds.schema if f.name.startswith("tile_coverage_")]
    if not ann_cols:
        raise RuntimeError(
            "No tile_coverage_* columns found in tiles parquet. "
            "Check that tiling used a class mapping with annotations."
        )
    ann_filter = pads.field(ann_cols[0]) > 0
    for c in ann_cols[1:]:
        ann_filter = ann_filter | (pads.field(c) > 0)

    tiles_table = tiles_ds.to_table(filter=ann_filter)
    ann_count = len(tiles_table)
    print(
        f"[{split_name}] annotation filter: "
        f"{original_count} → {ann_count} ({ann_count / original_count:.1%} kept)",
        flush=True,
    )
    if ann_count == 0:
        raise RuntimeError(
            f"All {original_count} tiles dropped by annotation filter for {split_name}. "
            "Check the tiling run's class mapping and annotation sources."
        )

    tissue_local = mlflow.artifacts.download_artifacts(
        run_id=tissue_stats_run_id,
        artifact_path=f"{tissue_stats_artifact_path}/{split_name}_tiles.parquet",
    )
    tissue_ds = pads.dataset(tissue_local, format="parquet")
    tissue_cols = [
        f.name
        for f in tissue_ds.schema
        if f.name in {"slide_id", "x", "y"} or f.name.endswith("_tissue_coverage")
    ]
    t = time.monotonic()
    print(f"[{split_name}] reading tissue stats: columns={tissue_cols}", flush=True)
    tissue_table = tissue_ds.to_table(
        columns=tissue_cols,
        filter=pads.field(tissue_column) > 0,
    )
    print(
        f"[{split_name}] tissue read: {len(tissue_table)} rows "
        f"in {time.monotonic() - t:.1f}s",
        flush=True,
    )

    t = time.monotonic()
    print(f"[{split_name}] joining via pandas…", flush=True)
    tiles_df = tiles_table.to_pandas()
    tissue_df = tissue_table.to_pandas()
    del tiles_table, tissue_table
    filtered_df = tiles_df.merge(tissue_df, on=["slide_id", "x", "y"], how="inner")
    del tiles_df, tissue_df
    filtered = pa.Table.from_pandas(filtered_df, preserve_index=False)
    final_count = len(filtered)
    del filtered_df
    print(
        f"[{split_name}] tissue join: "
        f"{ann_count} → {final_count} ({final_count / ann_count:.1%} kept) "
        f"in {time.monotonic() - t:.1f}s",
        flush=True,
    )

    t = time.monotonic()
    pq.write_table(filtered, str(output_path))
    print(
        f"[{split_name}] wrote {output_path.name} in {time.monotonic() - t:.1f}s",
        flush=True,
    )
    return {
        "original_count": original_count,
        "after_annotation": ann_count,
        "after_tissue": final_count,
    }


@with_cli_args(["+preprocessing=filter_tiles"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_run_id = config.dataset.mlflow_artifacts.tiling_run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split_name in ("train", "test"):
            output_path = Path(tmp_dir) / f"{split_name}_tiles.parquet"
            stats = filter_split(
                tiling_run_id=tiling_run_id,
                tissue_stats_run_id=config.tissue_stats_run_id,
                tissue_stats_artifact_path=config.tissue_stats_artifact_path,
                tissue_column=config.tissue_coverage_column,
                split_name=split_name,
                output_path=output_path,
            )
            for key, value in stats.items():
                mlflow.log_metric(f"{split_name}_{key}", value)

        mlflow.log_artifacts(tmp_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    main()
