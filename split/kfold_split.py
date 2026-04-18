from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold


def load_parquet_artifact(run_id: str, artifact_path: str) -> pd.DataFrame:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    print(f"[DEBUG] Downloaded to: {local_path}", flush=True)
    print(f"[DEBUG] File size: {Path(local_path).stat().st_size / 1024**2:.1f} MB", flush=True)
    print("[DEBUG] Reading parquet...", flush=True)
    df = pd.read_parquet(local_path)
    print(f"[DEBUG] Read complete: {df.shape}", flush=True)
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
    print("[DEBUG] Loading parquet artifact...", flush=True)
    df = load_parquet_artifact(
        config.dataset.mlflow_artifacts.tiling_run_id,
        config.dataset.mlflow_artifacts.train_tiles_filename,
    )
    print(f"[DEBUG] Loaded {len(df)} rows, columns: {list(df.columns)}", flush=True)

    # Derive label and tissue_prop from ROI coverage columns.
    # roi_coverage_* measures the fraction of the central half-size ROI covered by each
    # class, which is more representative of tile content than the full-tile coverage.
    # tissue_prop: total annotated tissue fraction across all classes in the ROI.
    # label: the dominant class in the ROI (highest coverage).
    roi_cols = [c for c in df.columns if c.startswith("roi_coverage_")]
    print(f"[DEBUG] ROI columns: {roi_cols}", flush=True)
    df["tissue_prop"] = df[roi_cols].sum(axis=1)
    print("[DEBUG] tissue_prop computed", flush=True)
    df["label"] = df[roi_cols].idxmax(axis=1).str.removeprefix("roi_coverage_")
    print("[DEBUG] label computed via idxmax", flush=True)
    df.loc[df["tissue_prop"] == 0, "label"] = "background"
    print(f"[DEBUG] Label distribution:\n{df['label'].value_counts()}", flush=True)

    df = collapse_rare_labels(df, n_folds=config.n_folds)
    print("[DEBUG] Rare labels collapsed", flush=True)
    df = assign_folds(df, n_folds=config.n_folds, random_state=config.random_state)
    print("[DEBUG] Folds assigned", flush=True)

    log_fold_statistics(df, n_folds=config.n_folds)
    print("[DEBUG] Statistics logged", flush=True)

    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir)
        df.to_parquet(out_path / "kfold_tiles.parquet", index=False)
        print("[DEBUG] Parquet written", flush=True)
        logger.log_artifacts(
            local_dir=str(out_path), artifact_path=config.mlflow_artifact_path
        )
        print("[DEBUG] Artifacts logged, done", flush=True)


if __name__ == "__main__":
    main()
