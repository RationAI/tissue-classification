from collections import defaultdict
from collections.abc import Iterable
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
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
        optimizer: str = "adamw",
        lbfgs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "decode_head"])

        if optimizer not in {"adamw", "lbfgs"}:
            raise ValueError(f"Unsupported optimizer {optimizer!r}")
        if optimizer == "lbfgs":
            self.automatic_optimization = False

        self.backbone = backbone
        self.decode_head = decode_head
        self._lbfgs_batches: list[tuple[Tensor, Tensor]] = []

        self.class_names = [
            n for n, _ in sorted(class_indices.items(), key=lambda kv: kv[1])
        ]
        num_classes = len(self.class_names)
        # Placeholder weight so `criterion.weight` always exists in state_dict.
        # setup(stage="fit") overrides with class-balanced weights; checkpoints
        # then load with strict=True regardless of stage.
        self.criterion = nn.CrossEntropyLoss(weight=torch.ones(num_classes))

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
            if self.hparams["optimizer"] == "lbfgs":
                self._validate_lbfgs_full_batch(datamodule, len(labels))
            num_classes = len(self.class_names)
            counts = np.bincount(labels, minlength=num_classes).astype(float)
            counts = np.maximum(counts, 1.0)
            weights = len(labels) / (num_classes * counts)
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32)
            )
            for cls, w in zip(self.class_names, weights.tolist(), strict=True):
                mlflow.log_metric(f"class_weight/{cls}", w)
        # Non-fit stages keep the placeholder ones-weight criterion from
        # __init__ so `criterion.weight` stays in state_dict for strict load.

    def forward(self, x: Tensor) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features)

    def training_step(self, batch: Input, batch_idx: int) -> Tensor:
        if self.hparams["optimizer"] == "lbfgs":
            return self._lbfgs_training_step(batch, batch_idx)

        inputs, targets, *_ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.hparams["optimizer"] == "lbfgs":
            return
        norms = grad_norm(self, norm_type=2)
        self.log(
            "train/grad_norm",
            norms["grad_2.0_norm_total"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch: Input, batch_idx: int) -> None:
        inputs, targets, *_ = batch
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

    def test_step(self, batch: Input, batch_idx: int) -> dict[str, Any]:
        inputs, targets, slide_ids, xs, ys = batch
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
        return {
            "slide_id": list(slide_ids),
            "x": xs.cpu(),
            "y": ys.cpu(),
            "target": targets.cpu(),
            "pred": preds.cpu(),
            "probs": outputs.softmax(dim=1).cpu(),
        }

    def on_test_epoch_end(self) -> None:
        self._log_per_class(self.test_per_class, "test")
        self._log_confmat(self.test_confmat, "test")
        self._log_per_slide_accuracy()
        self._test_slide_correct.clear()
        self._test_slide_total.clear()

    def predict_step(
        self, batch: Input, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        inputs, targets, slide_ids, xs, ys = batch
        outputs = self(inputs)
        probs = outputs.softmax(dim=1)
        preds = outputs.argmax(dim=1)
        return {
            "slide_id": list(slide_ids),
            "x": xs.cpu(),
            "y": ys.cpu(),
            "target": targets.cpu(),
            "pred": preds.cpu(),
            "probs": probs.cpu(),
        }

    def configure_optimizers(self) -> Optimizer:
        if self.hparams["optimizer"] == "lbfgs":
            lbfgs = self.hparams.get("lbfgs") or {}
            return torch.optim.LBFGS(
                self.parameters(),
                lr=self.hparams["learning_rate"],
                max_iter=lbfgs.get("max_iter", 100),
                max_eval=lbfgs.get("max_eval"),
                tolerance_grad=lbfgs.get("tolerance_grad", 1.0e-7),
                tolerance_change=lbfgs.get("tolerance_change", 1.0e-9),
                history_size=lbfgs.get("history_size", 100),
                line_search_fn=lbfgs.get("line_search_fn", "strong_wolfe"),
            )

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

    def _lbfgs_training_step(self, batch: Input, batch_idx: int) -> Tensor:
        inputs, targets, *_ = batch
        self._lbfgs_batches.append(self._prepare_lbfgs_batch(inputs, targets))
        lbfgs = self.hparams.get("lbfgs") or {}
        accumulation_steps = int(lbfgs.get("accumulate_batches", 1))
        is_last_batch = batch_idx + 1 == self.trainer.num_training_batches
        should_step = len(self._lbfgs_batches) >= accumulation_steps or is_last_batch
        if not should_step:
            with torch.no_grad():
                return self.criterion(self(inputs), targets)

        optimizer = cast("Any", self.optimizers())
        total_samples = sum(targets.numel() for _, targets in self._lbfgs_batches)

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss, _, _ = self._lbfgs_buffered_loss(total_samples)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite LBFGS loss: {loss.item()}")
            self.manual_backward(loss)
            return loss

        step_loss = optimizer.step(closure=closure)
        if not isinstance(step_loss, Tensor):
            step_loss = torch.as_tensor(step_loss, device=self.device)

        optimizer.zero_grad()
        loss, ce_loss, l2_loss = self._lbfgs_buffered_loss(total_samples)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite LBFGS post-step loss: {loss.item()}")
        self.manual_backward(loss)
        grad_norm = self._total_grad_norm()
        if grad_norm is not None and not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite LBFGS gradient norm: {grad_norm.item()}"
            )
        optimizer.zero_grad()
        self._lbfgs_batches.clear()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True)
        self.log("train/l2_loss", l2_loss, on_step=True, on_epoch=True)
        if grad_norm is not None:
            self.log(
                "train/grad_norm",
                grad_norm,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        self.log("train/lbfgs_step_loss", step_loss.detach(), on_step=True)
        return loss

    def _prepare_lbfgs_batch(
        self, inputs: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor]:
        lbfgs = self.hparams.get("lbfgs") or {}
        if lbfgs.get("accumulate_on_cpu", False):
            return inputs.detach().cpu(), targets.detach().cpu()
        return inputs, targets

    def _validate_lbfgs_full_batch(self, datamodule: Any, train_size: int) -> None:
        lbfgs = self.hparams.get("lbfgs") or {}
        train_batch_size = int(datamodule.train_batch_size)
        accumulation_steps = int(lbfgs.get("accumulate_batches", 1))
        effective_batch_size = train_batch_size * accumulation_steps

        if datamodule.train_shuffle:
            raise ValueError("LBFGS requires data.train_shuffle=false.")
        if datamodule.train_drop_last:
            raise ValueError("LBFGS requires data.train_drop_last=false.")
        if effective_batch_size < train_size:
            raise ValueError(
                "LBFGS requires a deterministic full-batch objective. Set "
                "data.train_batch_size >= len(train) or set "
                "model.lbfgs.accumulate_batches >= ceil(len(train) / "
                "data.train_batch_size). Current effective batch size is "
                f"{effective_batch_size} for {train_size} training samples."
            )

    def _lbfgs_buffered_loss(self, total_samples: int) -> tuple[Tensor, Tensor, Tensor]:
        ce_loss = torch.zeros((), device=self.device)
        for micro_inputs, micro_targets in self._lbfgs_batches:
            micro_inputs = micro_inputs.to(self.device)
            micro_targets = micro_targets.to(self.device)
            outputs = self(micro_inputs)
            weight = micro_targets.numel() / total_samples
            ce_loss = ce_loss + self.criterion(outputs, micro_targets) * weight

        l2_loss = self._l2_loss()
        return ce_loss + l2_loss, ce_loss, l2_loss

    def _objective_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(outputs, targets) + self._l2_loss()

    def _l2_loss(self) -> Tensor:
        weight_decay = self.hparams["weight_decay"]
        if weight_decay == 0:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        for name, param in self.named_parameters():
            if param.requires_grad and name.endswith("weight"):
                penalty = penalty + param.square().sum()
        return 0.5 * weight_decay * penalty

    def _total_grad_norm(self) -> Tensor | None:
        grads = [
            param.grad.detach().norm(2)
            for param in self.parameters()
            if param.grad is not None
        ]
        if not grads:
            return None
        return torch.linalg.vector_norm(torch.stack(grads), ord=2)

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
