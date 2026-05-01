import hydra
import lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@with_cli_args(["+ml=linear_probe"])
@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    pl.seed_everything(config.seed)

    datamodule: pl.LightningDataModule = instantiate(config.data)
    model: pl.LightningModule = instantiate(config.model)
    trainer: pl.Trainer = instantiate(config.trainer, logger=logger)

    if config.mode == "fit":
        trainer.fit(model, datamodule=datamodule, ckpt_path=config.checkpoint)
        trainer.test(model, datamodule=datamodule)
    elif config.mode == "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=config.checkpoint)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")


if __name__ == "__main__":
    main()
