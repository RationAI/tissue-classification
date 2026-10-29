"""Slide map for inference-only datasets (no training, no annotation masks).

Unlike ``wsi_mapping`` + ``train_test_split``, every slide matched by
``dataset.wsi_sources`` is kept, whether or not it has an annotation, and all of
them land in the test split. One run therefore serves as both the dataset's
``mapping_run_id`` and ``train_test_split_run_id``. Annotations of these datasets
are evaluation targets read by dedicated loaders, so ``annotation_path`` is left
empty and tiling emits tiles with zero annotation coverage.
"""

import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing.wsi_mapping import extract_wsi_metadata, find_files


@with_cli_args(["+preprocessing=inference_mapping"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df = find_files(config.dataset.wsi_sources).rename(columns={"path": "wsi_path"})
    if df.empty:
        raise RuntimeError(f"No slides matched {dict(config.dataset.wsi_sources)}")

    df["organ"] = str(config.dataset.organ)
    df["annotation_path"] = ""
    df_meta = df["wsi_path"].apply(lambda p: pd.Series(extract_wsi_metadata(p)))
    df = pd.concat([df, df_meta], axis=1).sort_values(by=["source", "id_stem"])
    df = df[
        [
            "id_stem",
            "source",
            "organ",
            "wsi_path",
            "annotation_path",
            "mpp_x",
            "mpp_y",
            "magnification",
            "vendor",
        ]
    ]

    mlflow.log_metric("wsi_count", len(df))

    with tempfile.TemporaryDirectory() as tmp_dir:
        map_path = Path(tmp_dir) / config.dataset.mlflow_artifacts.mapping_filename
        split_path = Path(tmp_dir) / config.dataset.mlflow_artifacts.test_split_filename
        split_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(map_path, index=False, encoding="utf-8")
        df.to_csv(split_path, index=False, encoding="utf-8")
        logger.log_artifacts(tmp_dir)


if __name__ == "__main__":
    main()
