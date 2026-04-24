from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold


def derive_labels(
    dataset: Dataset,
    roi_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive label, tissue_prop, and slide_id arrays from the dataset."""

    def compute(batch):
        roi_df = pd.DataFrame({col: batch[col] for col in roi_cols})
        tp = roi_df.sum(axis=1).values
        lbl = roi_df.idxmax(axis=1).str.removeprefix("roi_coverage_").values
        lbl[tp == 0] = "background"
        return {"label": lbl.tolist(), "tissue_prop": tp.tolist()}

    label_ds = dataset.select_columns(["slide_id", *roi_cols]).map(
        compute, batched=True
    )
    return (
        np.array(label_ds["label"], dtype=object),
        np.array(label_ds["tissue_prop"]),
        np.array(label_ds["slide_id"]),
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
            f"| tissue_prop {tp_mean:.3f} ± {tp_std:.3f}"
        )


@with_cli_args(defaults=["+split=kfold_split"])
@hydra.main(config_path="../configs", config_name="split", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    parquet_path = mlflow.artifacts.download_artifacts(
        run_id=config.dataset.mlflow_artifacts.tiling_run_id,
        artifact_path=config.dataset.mlflow_artifacts.train_tiles_filename,
    )

    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    roi_cols = [c for c in dataset.column_names if c.startswith("roi_coverage_")]
    if not roi_cols:
        raise ValueError("No roi_coverage_* columns found in the dataset.")

    labels, tissue_props, slide_ids = derive_labels(dataset, roi_cols)

    labels = collapse_rare_labels(labels, n_folds=config.n_folds)
    folds = assign_folds(
        labels, n_folds=config.n_folds, random_state=config.random_state
    )

    log_fold_statistics(labels, tissue_props, slide_ids, folds, n_folds=config.n_folds)

    dataset = dataset.add_column("label", labels.tolist())
    dataset = dataset.add_column("tissue_prop", tissue_props.tolist())
    dataset = dataset.add_column("fold", folds.tolist())

    with TemporaryDirectory() as output_dir:
        out_path = str(Path(output_dir) / "kfold_tiles.parquet")
        dataset.to_parquet(out_path)
        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
