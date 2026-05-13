from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedGroupKFold


def derive_labels(
    dataset: Dataset,
    roi_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive label, tissue_prop, and slide_id arrays from the dataset."""

    def compute(batch: dict[str, Any]) -> dict[str, Any]:
        roi_df = pd.DataFrame({col: batch[col] for col in roi_cols})
        tp = roi_df.sum(axis=1).values
        lbl = roi_df.idxmax(axis=1).str.removeprefix("roi_coverage_").values
        lbl[tp == 0] = "background"
        return {"label": lbl.tolist(), "tissue_prop": tp.tolist()}

    label_ds = dataset.select_columns(["slide_id", *roi_cols]).map(
        compute, batched=True
    )
    table = label_ds.data.table
    return (
        table.column("label").to_numpy(zero_copy_only=False),
        table.column("tissue_prop").to_numpy(zero_copy_only=False),
        table.column("slide_id").to_numpy(zero_copy_only=False),
    )


def drop_rare_class_slides(
    labels: np.ndarray, slide_ids: np.ndarray, n_folds: int
) -> np.ndarray:
    """Return a boolean keep-mask excluding tiles whose label has fewer than ``n_folds`` distinct slides.

    StratifiedGroupKFold requires each stratification class to be present in at
    least ``n_folds`` groups (slides). Rather than collapsing rare classes into a
    surrogate category — unreliable here because the upstream tissue/annotation
    filter removes most background tiles — all tiles carrying a rare label are
    dropped regardless of which slide they come from. Lost tiles are bounded by
    definition (fewer than n_folds slides per dropped class).
    """
    pairs = pd.DataFrame({"slide": slide_ids, "label": labels}).drop_duplicates()
    slide_counts = pairs.groupby("label").size()
    rare = slide_counts[slide_counts < n_folds].index.to_numpy()
    if len(rare) == 0:
        return np.ones(len(labels), dtype=bool)

    print(
        f"WARNING: {len(rare)} label(s) appear in fewer than {n_folds} slides and "
        f"will be dropped from the split: "
        + ", ".join(f"{cls}({int(slide_counts[cls])} slides)" for cls in rare),
        flush=True,
    )
    keep = ~np.isin(labels, rare)
    if keep.sum() == 0:
        raise ValueError(
            "All tiles dropped after rare-class filtering. "
            f"Rare classes: {', '.join(rare)}."
        )
    return keep


def assign_folds(
    labels: np.ndarray, groups: np.ndarray, n_folds: int, random_state: int
) -> np.ndarray:
    """Assign each tile to a validation fold using stratified group k-fold.

    Stratifies on tissue-class proxy labels while keeping all tiles from a single
    slide in the same fold.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    folds = np.full(len(labels), -1, dtype=np.int8)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(folds, labels, groups)):
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
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    for fold in range(n_folds):
        mask = folds == fold
        n_val = int(mask.sum())
        fold_sizes[fold] = n_val
        mlflow.log_metric(f"fold_{fold}_train_tiles", int((~mask).sum()))
        mlflow.log_metric(f"fold_{fold}_val_tiles", n_val)
        mlflow.log_metric(f"fold_{fold}_val_tile_pct", round(n_val / total, 4))
        mlflow.log_metric(f"fold_{fold}_val_slides", len(np.unique(slide_ids[mask])))
        mlflow.log_metric(
            f"fold_{fold}_val_tissue_prop_mean",
            round(float(tissue_props[mask].mean()), 4),
        )
        mlflow.log_metric(
            f"fold_{fold}_val_tissue_prop_std",
            round(float(tissue_props[mask].std()), 4),
        )

    size_mean = float(fold_sizes.mean())
    size_cv = float(fold_sizes.std() / size_mean) if size_mean > 0 else 0.0
    mlflow.log_metric("fold_size_cv", round(size_cv, 4))

    stats_df = pd.DataFrame({"fold": folds, "label": labels})
    label_dist = (
        stats_df.groupby(["fold", "label"]).size().unstack(fill_value=0).reset_index()
    )
    mlflow.log_table(
        data=label_dist,
        artifact_file="fold_statistics/label_distribution.json",
    )

    print(f"Total tiles: {total} | fold size CV: {size_cv:.3f}")
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
        run_id=config.dataset.mlflow_artifacts.filter_tiles_run_id,
        artifact_path="filter_tiles/train_tiles.parquet",
    )

    dataset = load_dataset("parquet", data_files=parquet_path, split="train")
    roi_cols = [c for c in dataset.column_names if c.startswith("roi_coverage_")]
    if not roi_cols:
        raise ValueError("No roi_coverage_* columns found in the dataset.")

    labels, tissue_props, slide_ids = derive_labels(dataset, roi_cols)

    keep_mask = drop_rare_class_slides(labels, slide_ids, n_folds=config.n_folds)
    if not keep_mask.all():
        dropped = int((~keep_mask).sum())
        mlflow.log_metric("dropped_rare_class_tiles", dropped)
        labels = labels[keep_mask]
        tissue_props = tissue_props[keep_mask]
        slide_ids = slide_ids[keep_mask]
        dataset = dataset.select(np.where(keep_mask)[0].tolist())

    folds = assign_folds(
        labels,
        slide_ids,
        n_folds=config.n_folds,
        random_state=config.random_state,
    )

    log_fold_statistics(
        labels,
        tissue_props,
        slide_ids,
        folds,
        n_folds=config.n_folds,
    )

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
