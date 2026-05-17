"""Tile-embedding dataset.

Joins precomputed tile embeddings with tile metadata (k-fold parquet for train,
filter_tiles parquet for test) and applies tissue + per-class thresholds at
load time to produce ``(embedding, class_index, slide_id, x, y)`` samples.
"""

from functools import cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import torch
from mlflow.artifacts import download_artifacts
from torch.utils.data import Dataset

from ml.typing import Sample


class EmbeddingTilesDataset(Dataset[Sample]):
    """Tile-level embedding dataset with on-the-fly filtering and labeling.

    Inner-joins ``embedding`` parquet with ``metadata`` parquet on
    ``(slide_id, x, y)``. Metadata must contain ``roi_coverage_*`` columns;
    label is the dominant class whose coverage meets its threshold. Tiles
    failing the tissue proportion floor, with more than one annotated class,
    or whose dominant class falls below its threshold are dropped.

    For train/val: pass the k-fold parquet as ``metadata_uri`` and use
    ``include_folds`` / ``exclude_folds`` to split. For test: pass the
    filter_tiles parquet (no fold column).
    """

    def __init__(
        self,
        embedding_uri: str | Path,
        metadata_uri: str | Path,
        class_indices: dict[str, int],
        thresholds: dict[str, float],
        tissue_prop_min: float,
        include_folds: list[int] | None = None,
        exclude_folds: list[int] | None = None,
    ) -> None:
        t0 = perf_counter()

        def _diag(msg: str) -> None:
            print(
                f"[EmbeddingTilesDataset +{perf_counter() - t0:6.1f}s] {msg}",
                flush=True,
            )

        _diag("filtering metadata")
        meta_df = self._filter_metadata(
            metadata_uri,
            thresholds,
            tissue_prop_min,
            include_folds,
            exclude_folds,
        )
        _diag(f"metadata filtered: {len(meta_df)} rows; reading embeddings")

        emb_dir = self._resolve_uri(embedding_uri)
        emb_table = pads.dataset(emb_dir, format="parquet").to_table(
            columns=["slide_id", "x", "y", "embedding"]
        )
        _diag(f"embedding table loaded: {emb_table.num_rows} rows")

        emb_col = emb_table.column("embedding")
        if pa.types.is_list(emb_col.type):
            target_type = pa.large_list(emb_col.type.value_type)
            emb_col = pa.chunked_array(
                [c.cast(target_type) for c in emb_col.chunks], type=target_type
            )

        emb_idx = pa.array(range(emb_table.num_rows), type=pa.int64())
        emb_keys = emb_table.drop(["embedding"]).append_column("_emb_idx", emb_idx)
        del emb_table

        meta_table = pa.Table.from_pandas(meta_df, preserve_index=False)
        del meta_df
        joined_keys = meta_table.join(
            emb_keys, keys=["slide_id", "x", "y"], join_type="inner"
        )
        del emb_keys, meta_table
        if joined_keys.num_rows == 0:
            raise RuntimeError("inner join with embeddings produced empty dataset")
        _diag(f"join done: {joined_keys.num_rows} matched rows; filling embeddings")

        _idx_col = joined_keys.column("_emb_idx")
        if isinstance(_idx_col, pa.ChunkedArray):
            _idx_col = _idx_col.combine_chunks()
        indices_np = _idx_col.to_numpy()

        first_chunk = emb_col.chunks[0]
        embedding_dim = len(first_chunk.values) // len(first_chunk)

        # sort indices for sequential per-chunk access; restore order afterwards
        sort_order = np.argsort(indices_np)
        sorted_indices = indices_np[sort_order]

        chunk_offsets = np.concatenate(
            [[0], np.cumsum([len(c) for c in emb_col.chunks])]
        )
        embeddings = np.empty((len(indices_np), embedding_dim), dtype=np.float32)
        for ci, chunk in enumerate(emb_col.chunks):
            lo, hi = chunk_offsets[ci], chunk_offsets[ci + 1]
            mask = (sorted_indices >= lo) & (sorted_indices < hi)
            if not mask.any():
                continue
            local_idx = sorted_indices[mask] - lo
            chunk_np = (
                chunk.values.to_numpy(zero_copy_only=False)
                .reshape(len(chunk), embedding_dim)
                .astype(np.float32)
            )
            embeddings[sort_order[mask]] = chunk_np[local_idx]
        del emb_col

        self.embeddings = embeddings
        labels = joined_keys.column("label").to_pandas()
        unknown = set(labels.unique()) - set(class_indices.keys())
        if unknown:
            raise ValueError(
                f"labels in data not present in class_indices: {sorted(unknown)}"
            )
        self.labels = labels.map(class_indices).to_numpy(dtype=np.int64)
        self.slide_ids = joined_keys.column("slide_id").to_pandas().to_numpy()
        self.xs = joined_keys.column("x").to_pandas().to_numpy(dtype=np.int64)
        self.ys = joined_keys.column("y").to_pandas().to_numpy(dtype=np.int64)
        _diag(f"dataset ready: {len(self.labels)} samples, dim={embeddings.shape[1]}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        return (
            torch.from_numpy(self.embeddings[idx]),
            int(self.labels[idx]),
            str(self.slide_ids[idx]),
            int(self.xs[idx]),
            int(self.ys[idx]),
        )

    @staticmethod
    def _filter_metadata(
        metadata_uri: str | Path,
        thresholds: dict[str, float],
        tissue_prop_min: float,
        include_folds: list[int] | None,
        exclude_folds: list[int] | None,
    ) -> pd.DataFrame:
        local = EmbeddingTilesDataset._resolve_uri(metadata_uri)
        df = pd.read_parquet(local)

        roi_cols = [c for c in df.columns if c.startswith("roi_coverage_")]
        if not roi_cols:
            raise ValueError(
                "metadata parquet has no roi_coverage_* columns; cannot label"
            )

        classes_in_data = {c.removeprefix("roi_coverage_") for c in roi_cols}
        missing_thresholds = classes_in_data - set(thresholds.keys())
        if missing_thresholds:
            raise ValueError(
                f"thresholds missing entries for classes present in data: "
                f"{sorted(missing_thresholds)}"
            )

        tissue_prop = df[roi_cols].sum(axis=1).to_numpy()
        df = df.loc[tissue_prop >= tissue_prop_min]
        if df.empty:
            raise RuntimeError("all tiles dropped by tissue_prop_min filter")

        nonzero_classes = (df[roi_cols].to_numpy() > 0).sum(axis=1)
        df = df.loc[pd.Series(nonzero_classes <= 1, index=df.index)]
        if df.empty:
            raise RuntimeError("all tiles dropped by single-class filter")

        roi_only = df[roi_cols]
        dominant = roi_only.idxmax(axis=1).str.removeprefix("roi_coverage_")
        dominant_value = roi_only.max(axis=1).to_numpy()
        threshold_per_row = dominant.map(thresholds).to_numpy()
        keep = dominant_value >= threshold_per_row
        df = df.loc[pd.Series(keep, index=df.index)].copy()
        df["label"] = dominant.to_numpy()[keep]
        if df.empty:
            raise RuntimeError("all tiles dropped by per-class thresholds")

        if include_folds is not None or exclude_folds is not None:
            if "fold" not in df.columns:
                raise RuntimeError(
                    "fold filter requested but 'fold' column not in metadata"
                )
            if include_folds is not None:
                df = df[df["fold"].isin(include_folds)]
            if exclude_folds is not None:
                df = df[~df["fold"].isin(exclude_folds)]
            if df.empty:
                raise RuntimeError("all tiles dropped by fold filter")

        return df[["slide_id", "x", "y", "label"]]

    @staticmethod
    def _resolve_uri(path_or_uri: str | Path) -> str:
        return EmbeddingTilesDataset._resolve_uri_cached(str(path_or_uri))

    @staticmethod
    @cache
    def _resolve_uri_cached(uri: str) -> str:
        if uri.startswith(("mlflow-artifacts:/", "runs:/")):
            return download_artifacts(artifact_uri=uri)
        return uri


class UnlabeledEmbeddingTilesDataset(Dataset[Sample]):
    """Tile-embedding dataset for prediction over tiles without class labels."""

    def __init__(
        self,
        embedding_uri: str | Path,
        metadata_uri: str | Path,
        tissue_column: str = "tile_tissue_coverage",
        tissue_min: float = 0.0,
        label_value: int = -1,
    ) -> None:
        t0 = perf_counter()

        def _diag(msg: str) -> None:
            print(
                f"[UnlabeledEmbeddingTilesDataset +{perf_counter() - t0:6.1f}s] {msg}",
                flush=True,
            )

        _diag("filtering metadata")
        meta_df = self._filter_metadata(metadata_uri, tissue_column, tissue_min)
        _diag(f"metadata filtered: {len(meta_df)} rows; reading embeddings")

        emb_dir = EmbeddingTilesDataset._resolve_uri(embedding_uri)
        emb_table = pads.dataset(emb_dir, format="parquet").to_table(
            columns=["slide_id", "x", "y", "embedding"]
        )
        _diag(f"embedding table loaded: {emb_table.num_rows} rows")

        emb_col = emb_table.column("embedding")
        if pa.types.is_list(emb_col.type):
            target_type = pa.large_list(emb_col.type.value_type)
            emb_col = pa.chunked_array(
                [c.cast(target_type) for c in emb_col.chunks], type=target_type
            )

        emb_idx = pa.array(range(emb_table.num_rows), type=pa.int64())
        emb_keys = emb_table.drop(["embedding"]).append_column("_emb_idx", emb_idx)
        del emb_table

        meta_table = pa.Table.from_pandas(meta_df, preserve_index=False)
        del meta_df
        joined_keys = meta_table.join(
            emb_keys, keys=["slide_id", "x", "y"], join_type="inner"
        )
        del emb_keys, meta_table
        if joined_keys.num_rows == 0:
            raise RuntimeError("inner join with embeddings produced empty dataset")
        _diag(f"join done: {joined_keys.num_rows} matched rows; filling embeddings")

        _idx_col = joined_keys.column("_emb_idx")
        if isinstance(_idx_col, pa.ChunkedArray):
            _idx_col = _idx_col.combine_chunks()
        indices_np = _idx_col.to_numpy()

        first_chunk = emb_col.chunks[0]
        embedding_dim = len(first_chunk.values) // len(first_chunk)
        sort_order = np.argsort(indices_np)
        sorted_indices = indices_np[sort_order]

        chunk_offsets = np.concatenate(
            [[0], np.cumsum([len(c) for c in emb_col.chunks])]
        )
        embeddings = np.empty((len(indices_np), embedding_dim), dtype=np.float32)
        for ci, chunk in enumerate(emb_col.chunks):
            lo, hi = chunk_offsets[ci], chunk_offsets[ci + 1]
            mask = (sorted_indices >= lo) & (sorted_indices < hi)
            if not mask.any():
                continue
            local_idx = sorted_indices[mask] - lo
            chunk_np = (
                chunk.values.to_numpy(zero_copy_only=False)
                .reshape(len(chunk), embedding_dim)
                .astype(np.float32)
            )
            embeddings[sort_order[mask]] = chunk_np[local_idx]
        del emb_col

        self.embeddings = embeddings
        self.labels = np.full(len(joined_keys), label_value, dtype=np.int64)
        self.slide_ids = joined_keys.column("slide_id").to_pandas().to_numpy()
        self.xs = joined_keys.column("x").to_pandas().to_numpy(dtype=np.int64)
        self.ys = joined_keys.column("y").to_pandas().to_numpy(dtype=np.int64)
        _diag(f"dataset ready: {len(self.labels)} samples, dim={embeddings.shape[1]}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Sample:
        return (
            torch.from_numpy(self.embeddings[idx]),
            int(self.labels[idx]),
            str(self.slide_ids[idx]),
            int(self.xs[idx]),
            int(self.ys[idx]),
        )

    @staticmethod
    def _filter_metadata(
        metadata_uri: str | Path,
        tissue_column: str,
        tissue_min: float,
    ) -> pd.DataFrame:
        local = EmbeddingTilesDataset._resolve_uri(metadata_uri)
        columns = ["slide_id", "x", "y", tissue_column]
        df = pd.read_parquet(local, columns=columns)
        if tissue_column not in df.columns:
            raise ValueError(
                f"metadata parquet has no {tissue_column!r} column; cannot filter"
            )
        df = df.loc[df[tissue_column] > tissue_min, ["slide_id", "x", "y"]]
        if df.empty:
            raise RuntimeError(
                f"all tiles dropped by {tissue_column} > {tissue_min} filter"
            )
        return df
