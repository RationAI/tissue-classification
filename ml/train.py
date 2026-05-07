import secrets
from typing import TYPE_CHECKING, Any

import hydra
import lightning as pl
import mlflow
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


if TYPE_CHECKING:
    from ml.data.embeddings_datamodule import EmbeddingsDataModule


if not OmegaConf.has_resolver("random_seed"):
    OmegaConf.register_new_resolver(
        "random_seed", lambda: secrets.randbits(32), use_cache=True
    )
if not OmegaConf.has_resolver("len"):
    OmegaConf.register_new_resolver("len", len)


@with_cli_args(["+ml=linear_probe"])
@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.mode == "fit":
        _fit(config, logger)
    elif config.mode == "test":
        raise NotImplementedError("Test mode arrives in the test-set PR")
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def _fit(config: DictConfig, logger: MLFlowLogger) -> None:
    datamodule: EmbeddingsDataModule = instantiate(config.data)
    datamodule.prepare_data()
    datamodule.setup("fit")

    per_fold_metrics: list[dict[str, Any]] = []
    for fold in range(config.n_folds):
        pl.seed_everything(config.seed + fold)
        datamodule.set_val_fold(fold)

        model: pl.LightningModule = instantiate(config.model)
        trainer: pl.Trainer = instantiate(config.trainer, logger=logger)
        trainer.fit(model, datamodule=datamodule)

        fold_metrics = {
            k: float(v)
            for k, v in trainer.callback_metrics.items()
            if k.startswith("val/")
        }
        per_fold_metrics.append(fold_metrics)
        for k, v in fold_metrics.items():
            mlflow.log_metric(f"fold_{fold}/{k}", v)

    if per_fold_metrics:
        keys = per_fold_metrics[0].keys()
        for k in keys:
            vals = np.array([m[k] for m in per_fold_metrics])
            mlflow.log_metric(f"{k}_mean", float(vals.mean()))
            mlflow.log_metric(f"{k}_std", float(vals.std()))


if __name__ == "__main__":
    main()
