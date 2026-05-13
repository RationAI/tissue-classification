from random import randint

import hydra
import mlflow
from lightning import seed_everything
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml.data import DataModule
from ml.meta_arch import MetaArch


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)
OmegaConf.register_new_resolver("len", lambda x: len(x))


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(config.data, _recursive_=False, _target_=DataModule)
    model = hydra.utils.instantiate(config.model, _target_=MetaArch)
    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    allowed_modes = ["fit", "test", "validate", "predict"]
    if config.mode not in allowed_modes:
        raise ValueError(f"Invalid mode {config.mode!r}. Allowed: {allowed_modes}")
    getattr(trainer, config.mode)(
        model,
        datamodule=data,
        ckpt_path=_resolve_checkpoint(config.checkpoint),
        weights_only=False,
    )
    mlflow.end_run()


def _resolve_checkpoint(checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    if checkpoint.startswith(("mlflow-artifacts:/", "runs:/")):
        return download_artifacts(artifact_uri=checkpoint)
    return checkpoint


if __name__ == "__main__":
    main()
