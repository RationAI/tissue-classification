from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold


def load_parquet_artifact(
    run_id: str, artifact_path: str, columns: list[str] | None = None
) -> pd.DataFrame:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    return pd.read_parquet(local_path, columns=columns)
    return df


def collapse_rare_labels(df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Collapse rare labels into 'background' to prevent StratifiedKFold from crashing.

    StratifiedKFold requires at least n_folds samples per class. Classes with fewer
    samples than that are relabeled as 'background' so the split can proceed. A warning
    is printed listing every affected class and its tile count.
    """
    counts = df["label"].value_counts()
    rare = counts[counts < n_folds].index.tolist()
    if rare:
        print(
            f"WARNING: {len(rare)} label(s) have fewer than {n_folds} tiles and will "
            f"be collapsed into 'background' for stratification: "
            + ", ".join(f"{cls}({counts[cls]})" for cls in rare)
        )
        df = df.copy()
        df.loc[df["label"].isin(rare), "label"] = "background"
    return df


def assign_folds(df: pd.DataFrame, n_folds: int, random_state: int) -> pd.DataFrame:
    """Assign each tile to a validation fold using stratified k-fold on tissue class label."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    df = df.copy().reset_index(drop=True)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold_idx
    return df


def log_fold_statistics(df: pd.DataFrame, n_folds: int) -> None:
    total = len(df)
    for fold in range(n_folds):
        val = df[df["fold"] == fold]
        mlflow.log_metric(f"fold_{fold}_train_tiles", total - len(val))
        mlflow.log_metric(f"fold_{fold}_val_tiles", len(val))
        mlflow.log_metric(f"fold_{fold}_val_slides", val["slide_id"].nunique())
        mlflow.log_metric(
            f"fold_{fold}_val_tissue_prop_mean", round(val["tissue_prop"].mean(), 4)
        )
        mlflow.log_metric(
            f"fold_{fold}_val_tissue_prop_std", round(val["tissue_prop"].std(), 4)
        )

    label_dist = (
        df.groupby(["fold", "label"]).size().unstack(fill_value=0).reset_index()
    )
    mlflow.log_table(
        data=label_dist,
        artifact_file="fold_statistics/label_distribution.json",
    )

    print(f"Total tiles: {total}")
    for fold in range(n_folds):
        val = df[df["fold"] == fold]
        print(
            f"Fold {fold}: {len(val)} val tiles ({len(val) / total * 100:.1f}%) "
            f"| {val['slide_id'].nunique()} slides "
            f"| tissue_prop {val['tissue_prop'].mean():.3f} ± {val['tissue_prop'].std():.3f}"
        )


@with_cli_args(defaults=["+split=kfold_split"])
@hydra.main(config_path="../configs", config_name="split", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # Only load columns needed for fold assignment — the full parquet has ~80M rows
    # and loading all 17 columns would exceed the job's memory budget.
    roi_class_names = [
        "Nerve", "Blood", "Connective-Tissue", "Fat", "Epithelium", "Muscle", "Other",
    ]
    roi_cols = [f"roi_coverage_{name}" for name in roi_class_names]

    df = load_parquet_artifact(
        config.dataset.mlflow_artifacts.tiling_run_id,
        config.dataset.mlflow_artifacts.train_tiles_filename,
        columns=["slide_id", "x", "y"] + roi_cols,
    )

    # Derive label and tissue_prop from ROI coverage columns.
    # roi_coverage_* measures the fraction of the central half-size ROI covered by each
    # class, which is more representative of tile content than the full-tile coverage.
    # tissue_prop: total annotated tissue fraction across all classes in the ROI.
    # label: the dominant class in the ROI (highest coverage).
    df["tissue_prop"] = df[roi_cols].sum(axis=1)
    df["label"] = df[roi_cols].idxmax(axis=1).str.removeprefix("roi_coverage_")
    df.loc[df["tissue_prop"] == 0, "label"] = "background"

    df = collapse_rare_labels(df, n_folds=config.n_folds)
    df = assign_folds(df, n_folds=config.n_folds, random_state=config.random_state)

    log_fold_statistics(df, n_folds=config.n_folds)

    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir)
        df.to_parquet(out_path / "kfold_tiles.parquet", index=False)
        logger.log_artifacts(
            local_dir=str(out_path), artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
