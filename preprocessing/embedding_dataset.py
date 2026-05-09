"""Build an embedding training dataset by joining tile metadata with embeddings.

Joins precomputed tile embeddings with k-fold metadata (train) / filter_tiles
metadata (test), applies tissue + per-class ROI thresholds before the join, and
emits a training-ready Parquet dataset (per-split ``slides.parquet`` +
``tiles.parquet``) ready for ``rationai.mlkit.data.datasets.SlidesTilesLoader``.
"""

import shutil
import tempfile
import time
from pathlib import Path

import hydra
import mlflow
import mlflow.artifacts
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing._labels import compute_label_and_tissue_prop


def apply_thresholds(
    df: pd.DataFrame,
    tissue_prop_min: float,
    thresholds: dict[str, float],
    roi_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """Filter df by tissue_prop_min then by per-dominant-class roi threshold.

    Returns ``(filtered_df, after_tissue_count)`` so the caller can log both
    intermediate counts.
    """
    df = df[df["tissue_prop"] >= tissue_prop_min]
    after_tissue = len(df)
    if df.empty:
        return df, after_tissue

    roi_only = df[roi_cols]
    dominant = roi_only.idxmax(axis=1).str.removeprefix("roi_coverage_")
    dominant_value = roi_only.max(axis=1).to_numpy()
    threshold_per_row = dominant.map(thresholds).to_numpy()
    keep = dominant_value >= threshold_per_row
    return df[keep].copy(), after_tissue


def join_embeddings(
    tiles_table: pa.Table,
    embedding_run_id: str,
    embedding_split: str,
) -> tuple[pa.Table, int]:
    """Join filtered tile metadata with embeddings on (slide_id, x, y) using Arrow join.

    Stays entirely in Arrow to avoid the slow fixed-size-list to_pandas() conversion
    on the embedding column.
    """
    t0 = time.time()
    emb_dir = mlflow.artifacts.download_artifacts(
        run_id=embedding_run_id, artifact_path=f"{embedding_split}/tiles"
    )
    print(f"[timing] download embeddings: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    emb_ds = pads.dataset(emb_dir, format="parquet")
    print(f"[timing] embedding dataset has {emb_ds.count_rows()} rows", flush=True)
    emb_table = emb_ds.to_table(columns=["slide_id", "x", "y", "embedding"])
    print(f"[timing] to_table: {time.time() - t0:.1f}s", flush=True)

    # Arrow Acero join doesn't support list<double> in non-key fields, so join on
    # keys only using a row-index column, then pull embeddings via take().
    # Cast embedding to large_list first: 1.1M rows * 768 doubles overflows int32
    # list offsets when chunks are concatenated by take().
    t0 = time.time()
    emb_col = emb_table.column("embedding")
    if pa.types.is_list(emb_col.type):
        emb_col = emb_col.cast(pa.large_list(emb_col.type.value_type))
    elif pa.types.is_fixed_size_list(emb_col.type):
        pass  # fixed_size_list has no offsets, no overflow risk
    else:
        emb_col = emb_col.cast(pa.large_list(pa.float64()))

    emb_idx = pa.array(range(emb_table.num_rows), type=pa.int32())
    emb_keys = emb_table.drop(["embedding"]).append_column("_emb_idx", emb_idx)
    del emb_table, emb_idx

    joined_keys = tiles_table.join(emb_keys, keys=["slide_id", "x", "y"], join_type="inner")
    del emb_keys
    embeddings = emb_col.take(joined_keys.column("_emb_idx"))
    del emb_col

    joined = joined_keys.drop(["_emb_idx"]).append_column("embedding", embeddings)
    del joined_keys, embeddings
    print(f"[timing] arrow join: {time.time() - t0:.1f}s  rows={joined.num_rows}", flush=True)

    dropped_no_embedding = tiles_table.num_rows - joined.num_rows
    return joined, dropped_no_embedding


def process_split(
    split_name: str,
    src_run_id: str,
    src_artifact_path: str,
    embedding_run_id: str,
    tissue_prop_min: float,
    thresholds: dict[str, float],
    output_split_dir: Path,
    derive: bool,
) -> dict[str, int]:
    print(f"[{split_name}] downloading src tiles...", flush=True)
    t0 = time.time()
    src_local = mlflow.artifacts.download_artifacts(
        run_id=src_run_id, artifact_path=src_artifact_path
    )
    df = pads.dataset(src_local, format="parquet").to_table().to_pandas()
    input_count = len(df)
    print(f"[{split_name}] src tiles loaded: {input_count} rows  {time.time() - t0:.1f}s", flush=True)

    roi_cols = [c for c in df.columns if c.startswith("roi_coverage_")]
    if not roi_cols:
        raise RuntimeError(
            f"No roi_coverage_* columns in {src_artifact_path}. "
            "Cannot apply class thresholds."
        )

    classes_in_data = {c.removeprefix("roi_coverage_") for c in roi_cols}
    missing = classes_in_data - set(thresholds.keys())
    if missing:
        raise ValueError(
            f"thresholds is missing entries for roi_coverage_* classes present "
            f"in data: {sorted(missing)}"
        )

    if derive:
        lbl, tp = compute_label_and_tissue_prop(df, roi_cols)
        df["label"] = lbl
        df["tissue_prop"] = tp

    df, after_tissue_filter = apply_thresholds(
        df, tissue_prop_min, thresholds, roi_cols
    )
    after_class_threshold = len(df)
    if after_class_threshold == 0:
        raise RuntimeError(
            f"All {input_count} tiles dropped by thresholds for split '{split_name}'."
        )

    drop_cols = [
        c for c in df.columns if c.startswith(("roi_coverage_", "tile_coverage_"))
    ]
    df = df.drop(columns=drop_cols)
    print(f"[{split_name}] after thresholds: {after_class_threshold} rows, joining embeddings...", flush=True)

    tiles_table = pa.Table.from_pandas(df, preserve_index=False)
    del df

    merged_table, dropped_no_embedding = join_embeddings(tiles_table, embedding_run_id, split_name)
    del tiles_table
    if dropped_no_embedding != 0:
        print(
            f"WARNING: {dropped_no_embedding} tiles in split '{split_name}' have "
            "no matching embedding and were dropped on join.",
            flush=True,
        )

    t0 = time.time()
    sort_indices = pc.sort_indices(merged_table, sort_keys=[("slide_id", "ascending")])
    merged_table = merged_table.take(sort_indices)
    print(f"[{split_name}] sort: {time.time() - t0:.1f}s", flush=True)

    output_split_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pq.write_table(merged_table, str(output_split_dir / "tiles.parquet"))
    print(f"[{split_name}] write parquet: {time.time() - t0:.1f}s", flush=True)

    print(f"[{split_name}] downloading slides.parquet...", flush=True)
    slides_local = mlflow.artifacts.download_artifacts(
        run_id=embedding_run_id, artifact_path=f"{split_name}/slides.parquet"
    )
    shutil.copy(slides_local, output_split_dir / "slides.parquet")

    log_label_distributions(split_name, merged_table)

    return {
        "input_count": input_count,
        "after_tissue_filter": after_tissue_filter,
        "after_class_threshold": after_class_threshold,
        "after_join": merged_table.num_rows,
        "dropped_no_embedding": dropped_no_embedding,
    }


def log_label_distributions(split_name: str, table: pa.Table) -> None:
    has_fold = "fold" in table.schema.names
    cols = ["label", "fold"] if has_fold else ["label"]
    df = table.select(cols).to_pandas()

    label_dist = (
        df["label"].value_counts().rename_axis("label").reset_index(name="count")
    )
    mlflow.log_table(
        data=label_dist,
        artifact_file=f"fold_statistics/{split_name}_label_distribution.json",
    )

    if has_fold:
        fold_dist = (
            df.groupby(["fold", "label"]).size().unstack(fill_value=0).reset_index()
        )
        mlflow.log_table(
            data=fold_dist,
            artifact_file=f"fold_statistics/{split_name}_fold_label_distribution.json",
        )


@with_cli_args(["+preprocessing=embedding_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    artifacts = config.dataset.mlflow_artifacts
    kfold_run_id = artifacts.kfold_run_id
    filter_tiles_run_id = artifacts.filter_tiles_run_id
    embedding_run_id = artifacts.embedding_run_id

    tissue_prop_min = float(config.tissue_prop_min)
    if tissue_prop_min <= 0:
        raise ValueError(
            f"tissue_prop_min must be > 0 (got {tissue_prop_min}); "
            "otherwise background tiles are not filtered out."
        )
    raw_thresholds = OmegaConf.to_container(config.thresholds, resolve=True)
    if not isinstance(raw_thresholds, dict):
        raise TypeError("config.thresholds must be a mapping of class -> threshold")
    thresholds = {str(k): float(v) for k, v in raw_thresholds.items()}

    splits = [
        ("train", kfold_run_id, "kfold_split/kfold_tiles.parquet", False),
        ("test", filter_tiles_run_id, "filter_tiles/test_tiles.parquet", True),
    ]

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root_path = Path(tmp_root)
        for split_name, src_run_id, src_artifact_path, derive in splits:
            stats = process_split(
                split_name=split_name,
                src_run_id=src_run_id,
                src_artifact_path=src_artifact_path,
                embedding_run_id=embedding_run_id,
                tissue_prop_min=tissue_prop_min,
                thresholds=thresholds,
                output_split_dir=tmp_root_path / split_name,
                derive=derive,
            )
            for key, value in stats.items():
                mlflow.log_metric(f"{split_name}_{key}", value)

        mlflow.log_artifacts(str(tmp_root_path), config.mlflow_artifact_path)


if __name__ == "__main__":
    main()
