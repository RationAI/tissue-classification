from collections import defaultdict
from collections.abc import Iterable
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from ml.typing import Input, Outputs


class MetaArch(LightningModule):
    """Top-level classification architecture: backbone + decode_head.

    For linear probing on precomputed embeddings, ``backbone`` is typically
    ``nn.Identity`` and ``decode_head`` is a single ``nn.Linear``.
    Criterion is class-weighted CrossEntropyLoss, computed from training labels in setup().
    """

    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        class_indices: dict[str, int],
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "decode_head"])

        self.backbone = backbone
        self.decode_head = decode_head
        self.criterion: nn.Module

        self.class_names = [
            n for n, _ in sorted(class_indices.items(), key=lambda kv: kv[1])
        ]
        num_classes = len(self.class_names)

        macro_metrics = MetricCollection(
            {
                "acc_macro": MulticlassAccuracy(
                    num_classes=num_classes, average="macro"
                ),
                "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        per_class_metrics = MetricCollection(
            {
                "acc_per_class": MulticlassAccuracy(
                    num_classes=num_classes, average=None
                ),
                "f1_per_class": MulticlassF1Score(
                    num_classes=num_classes, average=None
                ),
            }
        )
        self.val_metrics = macro_metrics.clone(prefix="validation/")
        self.test_metrics = macro_metrics.clone(prefix="test/")
        self.val_per_class = per_class_metrics.clone(prefix="validation/")
        self.test_per_class = per_class_metrics.clone(prefix="test/")
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

        self._test_slide_correct: dict[str, int] = defaultdict(int)
        self._test_slide_total: dict[str, int] = defaultdict(int)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            datamodule = cast("Any", self.trainer).datamodule
            labels = datamodule.train.labels
            num_classes = len(self.class_names)
            counts = np.bincount(labels, minlength=num_classes).astype(float)
            counts = np.maximum(counts, 1.0)
            weights = len(labels) / (num_classes * counts)
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32)
            )
            for cls, w in zip(self.class_names, weights.tolist(), strict=True):
                mlflow.log_metric(f"class_weight/{cls}", w)
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.ones(len(self.class_names), dtype=torch.float32)
            )

    def forward(self, x: Tensor) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features)

    def training_step(self, batch: Input, batch_idx: int) -> Tensor:
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Input, batch_idx: int) -> None:
        inputs, targets, _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(outputs, targets)
        self.val_per_class.update(outputs, targets)
        self.val_confmat.update(outputs, targets)
        self.log_dict(self.val_metrics, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self._log_per_class(self.val_per_class, "validation")
        self._log_confmat(self.val_confmat, "validation")

    def test_step(self, batch: Input, batch_idx: int) -> None:
        inputs, targets, slide_ids = batch
        outputs = self(inputs)
        self.test_metrics.update(outputs, targets)
        self.test_per_class.update(outputs, targets)
        self.test_confmat.update(outputs, targets)
        self.log_dict(self.test_metrics, on_epoch=True)

        preds = outputs.argmax(dim=1)
        correct = (preds == targets).cpu().tolist()
        for slide_id, ok in zip(slide_ids, correct, strict=True):
            self._test_slide_correct[slide_id] += int(ok)
            self._test_slide_total[slide_id] += 1

    def on_test_epoch_end(self) -> None:
        self._log_per_class(self.test_per_class, "test")
        self._log_confmat(self.test_confmat, "test")
        self._log_per_slide_accuracy()
        self._test_slide_correct.clear()
        self._test_slide_total.clear()

    def predict_step(
        self, batch: Input, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        inputs, targets, slide_ids = batch
        outputs = self(inputs)
        probs = outputs.softmax(dim=1)
        preds = outputs.argmax(dim=1)
        return {
            "slide_id": list(slide_ids),
            "target": targets.cpu(),
            "pred": preds.cpu(),
            "probs": probs.cpu(),
        }

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

    def _log_per_class(self, collection: MetricCollection, split: str) -> None:
        computed = collection.compute()
        for metric_name, values in computed.items():
            tag = metric_name.split("/")[-1]  # e.g. "acc_per_class"
            for cls_name, val in zip(self.class_names, values.tolist(), strict=True):
                self.log(f"{split}/{tag}/{cls_name}", val, on_epoch=True)
        collection.reset()

    def _log_confmat(self, confmat: MulticlassConfusionMatrix, split: str) -> None:
        matrix = confmat.compute().cpu().numpy()
        confmat.reset()
        fig = _confmat_figure(matrix, self.class_names, title=f"{split} confmat")
        artifact_file = f"confusion_matrix/{split}_epoch_{self.current_epoch}.png"
        try:
            mlflow.log_figure(fig, artifact_file=artifact_file)
        finally:
            plt.close(fig)

    def _log_per_slide_accuracy(self) -> None:
        accs = [
            self._test_slide_correct[s] / self._test_slide_total[s]
            for s in self._test_slide_total
        ]
        if not accs:
            return
        self.log("test/slide_acc_mean", float(np.mean(accs)), on_epoch=True)
        self.log("test/slide_acc_median", float(np.median(accs)), on_epoch=True)
        self.log("test/slide_acc_min", float(np.min(accs)), on_epoch=True)

        rows = [
            {
                "slide_id": s,
                "tile_accuracy": self._test_slide_correct[s] / n,
                "n_tiles": n,
            }
            for s, n in self._test_slide_total.items()
        ]
        mlflow.log_table(
            data=pd.DataFrame(rows),
            artifact_file="per_slide/test_tile_accuracy.parquet",
        )


def _confmat_figure(
    matrix: np.ndarray, class_names: Iterable[str], title: str
) -> Figure:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_sums, where=row_sums > 0, out=np.zeros_like(matrix, dtype=float)
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    names = list(class_names)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
