from random import randint

import hydra
import mlflow
from lightning import seed_everything
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
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)
    mlflow.end_run()


if __name__ == "__main__":
    main()
