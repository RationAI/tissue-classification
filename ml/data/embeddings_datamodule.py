import logging
import warnings
from pathlib import Path
from typing import Any

import lightning as pl
import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pad
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


class EmbeddingsDataModule(pl.LightningDataModule):
    """Linear-probe data module.

    Downloads embeddings and kfold-split artifacts from MLflow, joins them on
    (slide_id, x, y), applies class mapping and coverage filters, and exposes
    train/val splits per fold via set_val_fold().
    """

    def __init__(
        self,
        embeddings_run_id: str,
        kfold_run_id: str,
        class_mapping: dict[str, list[str]],
        class_indices: dict[str, int],
        kfold_artifact_path: str = "kfold_split/kfold_tiles.parquet",
        drop_unmapped: bool = True,
        tissue_prop_min: float = 0.0,
        class_coverage_min: float = 0.0,
        batch_size: int = 1024,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embeddings_run_id = embeddings_run_id
        self.kfold_run_id = kfold_run_id
        self.kfold_artifact_path = kfold_artifact_path
        self.drop_unmapped = drop_unmapped
        self.tissue_prop_min = tissue_prop_min
        self.class_coverage_min = class_coverage_min
        self.batch_size = batch_size
        self.num_workers = num_workers
        cm: Any = (
            OmegaConf.to_container(class_mapping, resolve=True)
            if OmegaConf.is_config(class_mapping)
            else class_mapping
        )
        ci: Any = (
            OmegaConf.to_container(class_indices, resolve=True)
            if OmegaConf.is_config(class_indices)
            else class_indices
        )
        self._class_mapping: dict[str, list[str]] = dict(cm)
        self._class_indices: dict[str, int] = dict(ci)
        self._raw_to_canonical: dict[str, str] = {
            raw: canonical
            for canonical, raws in self._class_mapping.items()
            for raw in raws
        }
        # Accept already-canonical labels as identity. The filtered tiles parquet
        # collapses ROI columns at tiling time, so kfold writes canonical names
        # (e.g. "Epithelium") directly into `label`; the raw→canonical lists in
        # the class-mapping YAML still cover legacy un-collapsed parquets.
        self._raw_to_canonical.update({c: c for c in self._class_indices})
        self._emb_dir: Path | None = None
        self._kfold_path: Path | None = None
        self.full_dataset: Dataset | None = None
        self.train_set: Dataset | None = None
        self.val_set: Dataset | None = None
        self._val_fold: int = 0
        self._fold_array: np.ndarray | None = None

    # ------------------------------------------------------------------
    # prepare_data - single-process MLflow download
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        emb_local = mlflow.artifacts.download_artifacts(
            run_id=self.embeddings_run_id,
            artifact_path="train",
        )
        self._emb_dir = Path(emb_local)

        kfold_local = mlflow.artifacts.download_artifacts(
            run_id=self.kfold_run_id,
            artifact_path=self.kfold_artifact_path,
        )
        self._kfold_path = Path(kfold_local)

    # ------------------------------------------------------------------
    # setup - join, filter, build full_dataset once
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if self.full_dataset is not None:
            return  # already built; use set_val_fold() to change splits

        assert self._emb_dir is not None and self._kfold_path is not None, (
            "Call prepare_data() before setup()"
        )

        # --- load kfold labels (small) ---
        labels_df = pd.read_parquet(self._kfold_path)
        n_initial = len(labels_df)
        roi_cols = [c for c in labels_df.columns if c.startswith("roi_coverage_")]

        # --- apply raw→canonical label mapping ---
        labels_df["label"] = labels_df["label"].map(self._raw_to_canonical)
        if self.drop_unmapped:
            n_before = len(labels_df)
            labels_df = labels_df[labels_df["label"].notna()].copy()
            dropped = n_before - len(labels_df)
            if dropped:
                log.warning("Dropping %d tiles with unmapped labels", dropped)
        mlflow.log_metric("n_tiles_initial", n_initial)
        mlflow.log_metric("n_tiles_after_label_map", len(labels_df))

        # --- tissue_prop filter ---
        if self.tissue_prop_min > 0.0:
            labels_df = labels_df[
                labels_df["tissue_prop"] >= self.tissue_prop_min
            ].copy()
        mlflow.log_metric("n_tiles_after_tissue_prop_filter", len(labels_df))

        # --- compute dominant_coverage (coverage of the assigned canonical class) ---
        dom_cov = np.zeros(len(labels_df), dtype=np.float32)
        label_arr = labels_df["label"].to_numpy()
        for canonical, raws in self._class_mapping.items():
            cols = [
                f"roi_coverage_{r}"
                for r in raws
                if f"roi_coverage_{r}" in labels_df.columns
            ]
            mask: np.ndarray = label_arr == canonical
            if cols and mask.any():
                dom_cov[mask] = labels_df.loc[mask, cols].sum(axis=1).to_numpy()
        labels_df["dominant_coverage"] = dom_cov

        if self.class_coverage_min > 0.0:
            labels_df = labels_df[
                labels_df["dominant_coverage"] >= self.class_coverage_min
            ].copy()
        mlflow.log_metric("n_tiles_after_class_coverage_filter", len(labels_df))

        labels_df = labels_df.drop(
            columns=[*roi_cols, "dominant_coverage"], errors="ignore"
        )

        # --- map label → integer class index (column name avoids collision with tile y-coord) ---
        labels_df["target"] = labels_df["label"].map(self._class_indices).astype(int)

        # --- load embeddings (memory-mapped Arrow) ---
        emb_files = sorted((self._emb_dir / "tiles").rglob("*.parquet"))
        emb_table = pad.dataset([str(f) for f in emb_files], format="parquet").to_table(
            columns=["slide_id", "x", "y", "embedding"]
        )
        log.info("Embeddings schema: %s", emb_table.schema)
        mlflow.log_metric("n_embeddings", len(emb_table))

        # --- inner-join on (slide_id, x, y) ---
        labels_table = pa.Table.from_pandas(
            labels_df[["slide_id", "x", "y", "fold", "tissue_prop", "target"]],
            preserve_index=False,
        )
        log.info("Labels schema: %s", labels_table.schema)
        mlflow.log_metric("n_labels", len(labels_table))

        # Acero join requires concrete, matching types on join keys. Normalise both
        # tables: slide_id → large_string, x/y → int64. Handles null-type columns
        # that can arise when Ray writes parquets with an inferred null schema, and
        # type mismatches between string vs large_string across files.
        # Acero also does not support list-typed non-key fields, so the embedding
        # column is excluded from the join and reattached via row-index lookup.
        join_key_types: dict[str, pa.DataType] = {
            "slide_id": pa.large_string(),
            "x": pa.int64(),
            "y": pa.int64(),
        }

        emb_keys = emb_table.select(["slide_id", "x", "y"]).append_column(
            "_emb_row", pa.array(range(len(emb_table)), type=pa.int64())
        )
        for tbl_name, tbl in [("emb_keys", emb_keys), ("labels", labels_table)]:
            for col_name, target_type in join_key_types.items():
                idx = tbl.schema.get_field_index(col_name)
                actual_type = tbl.schema.field(col_name).type
                if actual_type != target_type:
                    log.warning(
                        "%s.%s has type %s, casting to %s",
                        tbl_name, col_name, actual_type, target_type,
                    )
                    if tbl_name == "emb_keys":
                        emb_keys = emb_keys.set_column(
                            idx, col_name, emb_keys.column(col_name).cast(target_type)
                        )
                    else:
                        labels_table = labels_table.set_column(
                            idx, col_name, labels_table.column(col_name).cast(target_type)
                        )

        joined_meta = emb_keys.join(
            labels_table,
            keys=["slide_id", "x", "y"],
            join_type="inner",
        )
        emb_indices = joined_meta.column("_emb_row")
        joined = joined_meta.drop_columns(["_emb_row"]).append_column(
            pa.field("embedding", emb_table.schema.field("embedding").type),
            emb_table.column("embedding").take(emb_indices),
        )
        n_joined = len(joined)
        mlflow.log_metric("n_joined", n_joined)

        gap = len(labels_table) - n_joined
        if gap > 0:
            warnings.warn(
                f"Join gap: {gap} label tiles have no matching embedding "
                "(embed run may have dropped tiles due to upstream failures).",
                stacklevel=2,
            )

        self.full_dataset = Dataset(arrow_table=joined)
        self._fold_array = np.asarray(joined.column("fold"), dtype=np.int64)
        self.set_val_fold(self._val_fold)

    # ------------------------------------------------------------------
    # fold management
    # ------------------------------------------------------------------

    def set_val_fold(self, fold: int) -> None:
        self._val_fold = fold
        if self.full_dataset is None or self._fold_array is None:
            return
        train_idx = np.flatnonzero(self._fold_array != fold).tolist()
        val_idx = np.flatnonzero(self._fold_array == fold).tolist()
        self.train_set = self.full_dataset.select(train_idx).with_format(
            "torch", columns=["embedding", "target"]
        )
        self.val_set = self.full_dataset.select(val_idx).with_format(
            "torch", columns=["embedding", "target"]
        )

    def compute_class_weights(self, method: str = "balanced") -> list[float]:
        """Compute per-class loss weights from the current training fold.

        ``balanced`` follows sklearn's ``compute_class_weight``:
        ``w_c = n_samples / (n_classes * count_c)``. ``inverse`` uses
        ``1 / count_c`` normalised to mean 1.
        """
        if self.full_dataset is None or self._fold_array is None:
            raise RuntimeError("call setup()/set_val_fold() before compute_class_weights()")
        targets = np.asarray(self.full_dataset.data.column("target"), dtype=np.int64)
        train_mask = self._fold_array != self._val_fold
        train_targets = targets[train_mask]
        n_classes = len(self._class_indices)
        counts = np.bincount(train_targets, minlength=n_classes).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        if method == "balanced":
            weights = train_targets.size / (n_classes * counts)
        elif method == "inverse":
            weights = 1.0 / counts
            weights = weights / weights.mean()
        else:
            raise ValueError(f"Unknown class-weight method: {method!r}")
        return weights.tolist()

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------

    @staticmethod
    def _collate(batch: list[dict[str, Any]]) -> tuple[Any, Any]:
        import torch

        x = torch.stack([b["embedding"].float() for b in batch])
        y = torch.stack([b["target"].long() for b in batch])
        return x, y

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._loader(self.val_set, shuffle=False)
