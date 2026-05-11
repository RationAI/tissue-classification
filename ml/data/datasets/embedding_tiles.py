"""Tile-embedding dataset.

Reads the parquet artifact produced by ``preprocessing.embedding_dataset``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from torch.utils.data import Dataset

from ml.typing import Sample


class EmbeddingTilesDataset(Dataset[Sample]):
    """Returns ``(embedding, class_index, slide_id)`` triples from a tiles parquet.

    A single dataset instance corresponds to one parquet (train or test);
    fold-based CV is expressed by ``include_folds`` / ``exclude_folds``
    filters applied to the train parquet via separate dataset configs.
    """

    REQUIRED_COLUMNS = ("embedding", "label", "slide_id")

    def __init__(
        self,
        path_or_uri: str | Path,
        class_indices: dict[str, int],
        include_folds: list[int] | None = None,
        exclude_folds: list[int] | None = None,
    ) -> None:
        df = self._load_parquet(path_or_uri)

        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"tiles parquet missing columns: {sorted(missing)}")

        if include_folds is not None or exclude_folds is not None:
            if "fold" not in df.columns:
                raise RuntimeError(
                    "fold filter requested but 'fold' column not in parquet"
                )
            if include_folds is not None:
                df = df[df["fold"].isin(include_folds)]
            if exclude_folds is not None:
                df = df[~df["fold"].isin(exclude_folds)]

        unknown = set(df["label"].unique()) - set(class_indices.keys())
        if unknown:
            raise ValueError(
                f"labels in tiles not present in class_indices: {sorted(unknown)}"
            )

        self.embeddings = np.stack(df["embedding"].tolist()).astype(np.float32)
        self.labels = df["label"].map(class_indices).to_numpy(dtype=np.int64)
        self.slide_ids = df["slide_id"].to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        return (
            torch.from_numpy(self.embeddings[idx]),
            int(self.labels[idx]),
            str(self.slide_ids[idx]),
        )

    @staticmethod
    def _load_parquet(path_or_uri: str | Path) -> pd.DataFrame:
        s = str(path_or_uri)
        if s.startswith(("mlflow-artifacts:/", "runs:/")):
            local = download_artifacts(artifact_uri=s)
        else:
            local = s
        return pd.read_parquet(local)
