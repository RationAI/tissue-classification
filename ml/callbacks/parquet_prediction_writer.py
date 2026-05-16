"""Aggregate ``predict_step`` outputs and write them as a parquet artifact."""

from pathlib import Path
from typing import Any

import lightning as pl
import mlflow
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class ParquetPredictionWriter(BasePredictionWriter):
    """Collect per-tile predictions and write them as a parquet artifact.

    Aggregates ``predict_step`` outputs across the predict loop, writes one
    parquet file with ``slide_id``, ``target``, ``pred``, ``prob_<class>``
    columns and logs it to the active MLflow run.
    """

    def __init__(self, output_filename: str = "predictions.parquet") -> None:
        super().__init__(write_interval="epoch")
        self.output_filename = output_filename

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Any,
        batch_indices: Any,
    ) -> None:
        if trainer.global_rank != 0:
            return

        batches = (
            predictions
            if not predictions or isinstance(predictions[0], dict)
            else [b for dataloader_preds in predictions for b in dataloader_preds]
        )

        slide_ids: list[str] = []
        targets: list[int] = []
        preds: list[int] = []
        probs: list[np.ndarray] = []
        for b in batches:
            slide_ids.extend(b["slide_id"])
            targets.extend(b["target"].tolist())
            preds.extend(b["pred"].tolist())
            probs.append(b["probs"].numpy())

        if not slide_ids:
            return

        prob_matrix = np.concatenate(probs, axis=0)

        class_names = getattr(pl_module, "class_names", None)
        prob_columns = (
            [f"prob_{c}" for c in class_names]
            if class_names is not None and len(class_names) == prob_matrix.shape[1]
            else [f"prob_{i}" for i in range(prob_matrix.shape[1])]
        )

        df = pd.DataFrame({"slide_id": slide_ids, "target": targets, "pred": preds})
        df = pd.concat([df, pd.DataFrame(prob_matrix, columns=prob_columns)], axis=1)

        out_path = Path(trainer.default_root_dir) / self.output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

        active = mlflow.active_run()
        if active is not None:
            mlflow.log_artifact(str(out_path), artifact_path="predictions")
