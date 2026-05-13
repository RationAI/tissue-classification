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
        self, batch_size: int, num_workers: int = 0, **datasets: DictConfig
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
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
                self.val = instantiate(self.datasets["val"])
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
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Input] | None:
        if self.val is None:
            return None
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> Iterable[Input]:
        return DataLoader(
            self.predict, batch_size=self.batch_size, num_workers=self.num_workers
        )
