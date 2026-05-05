import tempfile
from pathlib import Path

import hydra
import mlflow
import mlflow.artifacts
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
        f"{original_count} → {ann_count} ({ann_count / original_count:.1%} kept)"
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
    tissue_table = pads.dataset(tissue_local, format="parquet").to_table(
        filter=pads.field(tissue_column) > 0,
    )

    filtered = tiles_table.join(
        tissue_table, keys=["slide_id", "x", "y"], join_type="inner"
    )
    final_count = len(filtered)
    print(
        f"[{split_name}] tissue filter: "
        f"{ann_count} → {final_count} ({final_count / ann_count:.1%} kept)"
    )

    pq.write_table(filtered, str(output_path))
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
