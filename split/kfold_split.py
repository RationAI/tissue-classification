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
    return pd.read_parquet(local_path)


def assign_folds(df: pd.DataFrame, n_folds: int, random_state: int) -> pd.DataFrame:
    """Assign each tile to a validation fold using stratified k-fold on tissue class label."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    df = df.copy()
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
    df = load_parquet_artifact(
        config.dataset.mlflow_artifacts.tiling_run_id,
        config.dataset.mlflow_artifacts.train_tiles_filename,
    )

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
