from pathlib import Path

import lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader


class EmbeddingsDataModule(pl.LightningDataModule):
    """Linear-probe data module backed by HuggingFace datasets.

    Expects parquet files with columns: `embedding` (list[float]), `label` (str),
    `fold` (int, train side only), and `tissue_prop` (float).
    """

    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        class_names: list[str],
        val_fold: int = 0,
        batch_size: int = 1024,
        num_workers: int = 4,
        tissue_prop_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

    def _load(self, parquet_dir: str) -> "Dataset":  # type: ignore[name-defined]
        files = sorted(str(p) for p in Path(parquet_dir).glob("*.parquet"))
        ds = load_dataset("parquet", data_files=files, split="train")
        if self.hparams.tissue_prop_min > 0:
            ds = ds.filter(
                lambda r: r["tissue_prop"] >= self.hparams.tissue_prop_min
            )
        ds = ds.map(lambda r: {"y": self.class_to_idx[r["label"]]})
        return ds

    def setup(self, stage: str) -> None:
        train_full = self._load(self.hparams.train_dir)
        self.train_set = train_full.filter(
            lambda r: r["fold"] != self.hparams.val_fold
        ).with_format("torch", columns=["embedding", "y"])
        self.val_set = train_full.filter(
            lambda r: r["fold"] == self.hparams.val_fold
        ).with_format("torch", columns=["embedding", "y"])
        self.test_set = self._load(self.hparams.test_dir).with_format(
            "torch", columns=["embedding", "y"]
        )

    @staticmethod
    def _collate(batch: list[dict]) -> tuple:
        import torch

        x = torch.stack([b["embedding"].float() for b in batch])
        y = torch.stack([b["y"].long() for b in batch])
        return x, y

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=self._collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=False)
