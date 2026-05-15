from typing import cast

import lightning as pl
import mlflow
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MetricCollection


class LinearProbe(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        class_names: list[str] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        normalize_input: bool = False,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.class_names = class_names
        self.lr = lr
        self.weight_decay = weight_decay
        self.normalize_input = normalize_input
        self.head = nn.Linear(embed_dim, num_classes)

        base_metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1_macro": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            }
        )
        self.train_metrics = base_metrics.clone(prefix="train/")
        self.val_metrics = base_metrics.clone(prefix="val/")

        self.val_f1_per_class = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.val_conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float)
            )
        else:
            self.class_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = F.normalize(x, dim=-1)
        return self.head(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics(logits, y), prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.log("val/loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics(logits, y), prog_bar=True)
        self.val_f1_per_class.update(logits, y)
        self.val_conf_matrix.update(logits, y)
        return loss

    def on_validation_epoch_end(self) -> None:
        f1_per_class = cast("torch.Tensor", self.val_f1_per_class.compute())
        class_names = self.class_names or [str(i) for i in range(self.num_classes)]
        for name, f1 in zip(class_names, f1_per_class, strict=True):
            self.log(f"val/f1_{name}", f1)

        conf_mat = cast("torch.Tensor", self.val_conf_matrix.compute()).cpu().numpy()
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(conf_mat, interpolation="nearest")
            ax.set(
                xticks=np.arange(len(class_names)),
                yticks=np.arange(len(class_names)),
                xticklabels=class_names,
                yticklabels=class_names,
                xlabel="Predicted",
                ylabel="True",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            mlflow.log_figure(fig, f"confusion_matrix_epoch{self.current_epoch}.png")
            plt.close(fig)
        except Exception:
            pass

        self.val_f1_per_class.reset()
        self.val_conf_matrix.reset()

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
