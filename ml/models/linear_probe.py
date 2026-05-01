import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics import Accuracy, F1Score, MetricCollection


class LinearProbe(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Linear(embed_dim, num_classes)

        metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1_macro": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def _step(self, batch, metrics, log_prefix: str) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log(f"{log_prefix}/loss", loss, prog_bar=True)
        self.log_dict(metrics(logits, y), prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, self.train_metrics, "train")

    def validation_step(self, batch, _):
        return self._step(batch, self.val_metrics, "val")

    def test_step(self, batch, _):
        return self._step(batch, self.test_metrics, "test")

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
