from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ml.typing import Input


class DataModule(LightningDataModule):
    """Generic Lightning datamodule that instantiates datasets lazily per stage.

    Mirrors the template pattern: ``**datasets`` accepts ``train``, ``val``,
    ``test`` (or ``predict``) DictConfigs whose targets resolve to ``Dataset``s.
    """

    def __init__(
        self,
        batch_size: int,
        eval_batch_size: int | None = None,
        num_workers: int = 0,
        train_shuffle: bool = True,
        train_drop_last: bool = True,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.train_drop_last = train_drop_last
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.val = (
                    instantiate(self.datasets["val"])
                    if self.datasets.get("val") is not None
                    else None
                )
            case "validate":
                self.val = (
                    instantiate(self.datasets["val"])
                    if self.datasets.get("val") is not None
                    else None
                )
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                dataset_cfg = self.datasets.get("predict") or self.datasets.get("test")
                if dataset_cfg is None:
                    raise KeyError("Neither 'predict' nor 'test' dataset configured")
                self.predict = instantiate(dataset_cfg)

    def train_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            drop_last=self.train_drop_last,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Input]:
        if self.val is None:
            return []
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.test, batch_size=self.eval_batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.predict, batch_size=self.eval_batch_size, num_workers=self.num_workers
        )
