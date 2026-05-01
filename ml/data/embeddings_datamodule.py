from pathlib import Path

import lightning as pl
import numpy as np
import pyarrow.dataset as pads
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingsDataset(Dataset):
    def __init__(
        self,
        parquet_dir: str | Path,
        class_names: list[str],
        coverage_prefix: str = "roi_coverage",
    ) -> None:
        ds = pads.dataset(str(parquet_dir), format="parquet")
        cols = ["embedding"] + [f"{coverage_prefix}_{c}" for c in class_names]
        table = ds.to_table(columns=cols)

        self.embeddings = np.stack(
            table.column("embedding").to_numpy(zero_copy_only=False)
        )
        coverages = np.stack(
            [table.column(f"{coverage_prefix}_{c}").to_numpy() for c in class_names],
            axis=1,
        )
        # hard labels via argmax over coverages; swap for soft targets if needed
        self.labels = coverages.argmax(axis=1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.embeddings[idx]).float(),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class EmbeddingsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        class_names: list[str],
        batch_size: int = 1024,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        coverage_prefix: str = "roi_coverage",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        full_train = EmbeddingsDataset(
            self.hparams.train_dir,
            self.hparams.class_names,
            self.hparams.coverage_prefix,
        )
        n_val = int(len(full_train) * self.hparams.val_fraction)
        n_train = len(full_train) - n_val
        self.train_set, self.val_set = torch.utils.data.random_split(
            full_train, [n_train, n_val], generator=torch.Generator().manual_seed(0)
        )
        self.test_set = EmbeddingsDataset(
            self.hparams.test_dir,
            self.hparams.class_names,
            self.hparams.coverage_prefix,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
