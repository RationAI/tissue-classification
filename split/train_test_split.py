from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.model_selection.split import train_test_split


def load_csv_artifact(run_id: str, artifact_path: str) -> pd.DataFrame:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    return pd.read_csv(local_path)


def assign_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign unique patient identifiers to ensure proper grouping during the split.

    Addresses specific naming conventions for TCGA data to prevent data leakage,
    as multiple slides can belong to the same patient.

    Args:
        df: The input DataFrame containing 'source' and 'id_stem' columns.

    Returns:
        The DataFrame with an appended 'group_id' column.
    """
    df["group_id"] = np.where(
        df["source"].astype(str).str.upper() == "TCGA",
        df["id_stem"].astype(str).str[:12],
        df["id_stem"].astype(str),
    )

    return df


def aggregate_rare_classes(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """Aggregate rare stratification classes to meet the minimum sample threshold.

    Executes a two-phase aggregation:
    1. Source-level aggregation: Combines different sources for the same organ
       if individual classes fall below min_samples.
    2. Absolute aggregation: Groups any remaining classes below min_samples
       into a fallback 'other_rare_organs' category.

    Args:
        df: The input DataFrame containing 'organ' and 'source' columns.
        min_samples: The minimum number of samples required per class.

    Returns:
        The DataFrame with an updated 'stratify_target' column.
    """
    # Correct known typos in the dataset
    df["organ"] = df["organ"].replace("gallblader", "gallbladder")
    df["stratify_target"] = (
        df["source"].astype(str).str.lower() + "_" + df["organ"].astype(str).str.lower()
    )

    class_counts = df["stratify_target"].value_counts()
    organs_with_rare_sources = df[
        df["stratify_target"].isin(class_counts[class_counts < min_samples].index)
    ]["organ"].unique()

    merged_target_map = {}
    for organ in organs_with_rare_sources:
        sources = df[df["organ"] == organ]["source"].unique()
        combined_sources = "_".join(sorted(str(s).lower() for s in sources))
        merged_target_map[organ] = f"{combined_sources}_{organ}"

    def resolve_rare_stratification(row: pd.Series) -> str:
        if row["organ"] in set(organs_with_rare_sources):
            return merged_target_map[row["organ"]]
        return row["stratify_target"]

    df["stratify_target"] = df.apply(resolve_rare_stratification, axis=1)

    final_counts = df["stratify_target"].value_counts()

    df["stratify_target"] = df["stratify_target"].apply(
        lambda x: (
            "other_rare_organs"
            if x in final_counts[final_counts < min_samples].index
            else x
        )
    )

    return df


def log_split_statistics(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    train_groups = set(df_train["group_id"])
    test_groups = set(df_test["group_id"])
    leakage = len(train_groups.intersection(test_groups))
    total_wsi = len(df_train) + len(df_test)

    mlflow.log_metric("train_wsi_count", len(df_train))
    mlflow.log_metric("test_wsi_count", len(df_test))
    mlflow.log_metric("train_patient_count", len(train_groups))
    mlflow.log_metric("test_patient_count", len(test_groups))
    mlflow.log_metric("patient_leakage_count", leakage)

    mlflow.log_metric("achieved_test_ratio", round(len(df_test) / total_wsi, 4))

    mlflow.log_metric("train_class_count", df_train["stratify_target"].nunique())
    mlflow.log_metric("test_class_count", df_test["stratify_target"].nunique())

    mlflow.log_metric("train_organ_count", df_train["organ"].nunique())
    mlflow.log_metric("test_organ_count", df_test["organ"].nunique())

    mlflow.log_metric("train_source_count", df_train["source"].nunique())
    mlflow.log_metric("test_source_count", df_test["source"].nunique())

    dist_df = (
        pd.DataFrame(
            {
                "Train_pct": df_train["stratify_target"].value_counts(normalize=True),
                "Test_pct": df_test["stratify_target"].value_counts(normalize=True),
            }
        )
        .fillna(0)
        .round(4)
        * 100
    )
    dist_df.index.name = "stratify_target"

    organ_dist = (
        pd.DataFrame(
            {
                "Train_pct": df_train["organ"].value_counts(normalize=True),
                "Test_pct": df_test["organ"].value_counts(normalize=True),
            }
        )
        .fillna(0)
        .round(4)
        * 100
    )
    organ_dist.index.name = "organ"

    source_dist = (
        pd.DataFrame(
            {
                "Train_pct": df_train["source"].value_counts(normalize=True),
                "Test_pct": df_test["source"].value_counts(normalize=True),
            }
        )
        .fillna(0)
        .round(4)
        * 100
    )
    source_dist.index.name = "source"

    mlflow.log_table(
        data=dist_df.reset_index(),
        artifact_file="split_statistics/stratification_distribution.json",
    )
    mlflow.log_table(
        data=organ_dist.reset_index(),
        artifact_file="split_statistics/organ_distribution.json",
    )
    mlflow.log_table(
        data=source_dist.reset_index(),
        artifact_file="split_statistics/source_distribution.json",
    )

    print("=== SPLIT STATISTICS ===")
    print(f"Train WSI: {len(df_train)} | Test WSI: {len(df_test)}")
    print(f"Train Patients: {len(train_groups)} | Test Patients: {len(test_groups)}")
    print(f"Patient Leakage: {leakage}")
    print(f"Achieved Test Ratio: {round(len(df_test) / total_wsi, 4)}")
    print(
        f"Train Classes: {df_train['stratify_target'].nunique()} | Test Classes: {df_test['stratify_target'].nunique()}"
    )
    print(
        f"Train Organs: {df_train['organ'].nunique()} | Test Organs: {df_test['organ'].nunique()}"
    )

    print("\n=== ORGAN DISTRIBUTION (%) ===")
    print(organ_dist.sort_values(by="Train_pct", ascending=False).to_string())

    print("\n=== SOURCE DISTRIBUTION (%) ===")
    print(source_dist.sort_values(by="Train_pct", ascending=False).to_string())

    print("\n=== STRATIFICATION DISTRIBUTION (%) ===")
    print(dist_df.sort_values(by="Train_pct", ascending=False).to_string())


@with_cli_args(defaults=["+split=train_test_split"])
@hydra.main(
    config_path="../configs",
    config_name="split",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = load_csv_artifact(
        config.dataset.mlflow_artifacts.mapping_run_id,
        config.dataset.mlflow_artifacts.mapping_filename,
    )
    df_missing = load_csv_artifact(
        config.dataset.mlflow_artifacts.missing_annotations_run_id,
        config.dataset.mlflow_artifacts.missing_annotations_filename,
    )

    df = df[~df["wsi_path"].isin(df_missing["wsi_path"])].reset_index(drop=True)

    df = assign_group_ids(df)
    df = aggregate_rare_classes(df, min_samples=config.min_samples)

    df_train, df_test = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["stratify_target"],
        groups=df["group_id"],
    )

    log_split_statistics(df_train, df_test)

    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir)
        df_train.to_csv(out_path / "train_split.csv", index=False)
        df_test.to_csv(out_path / "test_split.csv", index=False)

        logger.log_artifacts(
            local_dir=str(out_path), artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
