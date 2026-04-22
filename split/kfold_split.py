from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold


def derive_labels_streaming(
    parquet_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive label, tissue_prop, and slide_id from a parquet file in a streaming fashion.

    Reads only the roi_coverage_* and slide_id columns, one row group at a time,
    to avoid loading the entire DataFrame into memory at once.
    """
    pf = pq.ParquetFile(parquet_path)
    roi_cols = [c for c in pf.schema_arrow.names if c.startswith("roi_coverage_")]
    labels = []
    tissue_props = []
    slide_ids = []

    for batch in pf.iter_batches(columns=["slide_id", *roi_cols], batch_size=1_000_000):
        batch_df = batch.to_pandas()
        roi_values = batch_df[roi_cols]
        tp = roi_values.sum(axis=1).values
        lbl = roi_values.idxmax(axis=1).str.removeprefix("roi_coverage_").values
        lbl[tp == 0] = "background"

        tissue_props.append(tp)
        labels.append(lbl)
        slide_ids.append(batch_df["slide_id"].values)

    return (
        np.concatenate(labels),
        np.concatenate(tissue_props),
        np.concatenate(slide_ids),
    )


def collapse_rare_labels(labels: np.ndarray, n_folds: int) -> np.ndarray:
    """Collapse rare labels into 'background' to prevent StratifiedKFold from crashing.

    StratifiedKFold requires at least n_folds samples per class. Classes with fewer
    samples than that are relabeled as 'background' so the split can proceed. A warning
    is printed listing every affected class and its tile count.
    """
    unique, counts = np.unique(labels, return_counts=True)
    rare = unique[counts < n_folds]
    if len(rare) > 0:
        print(
            f"WARNING: {len(rare)} label(s) have fewer than {n_folds} tiles and will "
            f"be collapsed into 'background' for stratification: "
            + ", ".join(f"{cls}({counts[unique == cls][0]})" for cls in rare),
            flush=True,
        )
        labels = labels.copy()
        labels[np.isin(labels, rare)] = "background"
    return labels


def assign_folds(labels: np.ndarray, n_folds: int, random_state: int) -> np.ndarray:
    """Assign each tile to a validation fold using stratified k-fold on tissue class label."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = np.full(len(labels), -1, dtype=np.int8)
    for fold_idx, (_, val_idx) in enumerate(skf.split(folds, labels)):
        folds[val_idx] = fold_idx
    return folds


def log_fold_statistics(
    labels: np.ndarray,
    tissue_props: np.ndarray,
    slide_ids: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
) -> None:
    total = len(labels)
    for fold in range(n_folds):
        mask = folds == fold
        mlflow.log_metric(f"fold_{fold}_train_tiles", int((~mask).sum()))
        mlflow.log_metric(f"fold_{fold}_val_tiles", int(mask.sum()))
        mlflow.log_metric(f"fold_{fold}_val_slides", len(np.unique(slide_ids[mask])))
        mlflow.log_metric(
            f"fold_{fold}_val_tissue_prop_mean",
            round(float(tissue_props[mask].mean()), 4),
        )
        mlflow.log_metric(
            f"fold_{fold}_val_tissue_prop_std",
            round(float(tissue_props[mask].std()), 4),
        )

    stats_df = pd.DataFrame({"fold": folds, "label": labels})
    label_dist = (
        stats_df.groupby(["fold", "label"]).size().unstack(fill_value=0).reset_index()
    )
    mlflow.log_table(
        data=label_dist,
        artifact_file="fold_statistics/label_distribution.json",
    )

    print(f"Total tiles: {total}")
    for fold in range(n_folds):
        mask = folds == fold
        n_val = int(mask.sum())
        n_slides = len(np.unique(slide_ids[mask]))
        tp_mean = float(tissue_props[mask].mean())
        tp_std = float(tissue_props[mask].std())
        print(
            f"Fold {fold}: {n_val} val tiles ({n_val / total * 100:.1f}%) "
            f"| {n_slides} slides "
            f"| tissue_prop {tp_mean:.3f} \u00b1 {tp_std:.3f}"
        )


@with_cli_args(defaults=["+split=kfold_split"])
@hydra.main(config_path="../configs", config_name="split", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    parquet_path = mlflow.artifacts.download_artifacts(
        run_id=config.dataset.mlflow_artifacts.tiling_run_id,
        artifact_path=config.dataset.mlflow_artifacts.train_tiles_filename,
    )

    # Derive label and tissue_prop by streaming through row groups.
    # Only roi_coverage_* and slide_id columns are read, one batch at a time,
    # to keep memory usage low on the ~80M-row tiles parquet.
    # roi_coverage_* measures the fraction of the central half-size ROI covered by each
    # class, which is more representative of tile content than the full-tile coverage.
    # tissue_prop: total annotated tissue fraction across all classes in the ROI.
    # label: the dominant class in the ROI (highest coverage).
    labels, tissue_props, slide_ids = derive_labels_streaming(parquet_path)

    labels = collapse_rare_labels(labels, n_folds=config.n_folds)
    folds = assign_folds(
        labels, n_folds=config.n_folds, random_state=config.random_state
    )

    log_fold_statistics(labels, tissue_props, slide_ids, folds, n_folds=config.n_folds)

    # Write the output parquet with fold assignments by streaming through the
    # original file and appending the computed columns, avoiding a full load.
    pf = pq.ParquetFile(parquet_path)
    offset = 0
    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir) / "kfold_tiles.parquet"
        writer = None
        for batch in pf.iter_batches(batch_size=1_000_000):
            batch_df = batch.to_pandas()
            n = len(batch_df)
            batch_df["label"] = labels[offset : offset + n]
            batch_df["tissue_prop"] = tissue_props[offset : offset + n]
            batch_df["fold"] = folds[offset : offset + n]
            offset += n

            table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)

        if writer is not None:
            writer.close()

        logger.log_artifacts(
            local_dir=str(Path(output_dir)), artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
