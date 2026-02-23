from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks import process_items, tissue_mask, write_big_tiff
from rationai.masks.vips_filters import (
    VipsClosing,
    VipsCompose,
    VipsGrayScaleFilter,
    VipsOpening,
    VipsOtsu,
)
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


@ray.remote(memory=3 * 1024**3)
def process_slide(
    slide_path: Path, mpp: float, output_path: Path, disk_factor: int
) -> None:
    with OpenSlide(slide_path) as slide:
        level = slide.closest_level(mpp)
        mpp_x, mpp_y = slide.slide_resolution(level)

    slide = pyvips.Image.openslideload(str(slide_path), level=level)
    custom_filter = VipsCompose(
        [
            VipsGrayScaleFilter(),
            VipsOtsu(),
            VipsClosing(disc_factor=disk_factor),
            VipsOpening(disc_factor=5),
        ]
    )
    mask, _ = tissue_mask(slide, mpp=(mpp_x, mpp_y), filter=custom_filter)
    mask_path = output_path / slide_path.with_suffix(".tiff").name
    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)


def load_slide_paths(run_id: str, artifact_path: str) -> list[Path]:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    df = pd.read_csv(local_path)

    return [Path(p) for p in df["wsi_path"]]


@with_cli_args(defaults=["+preprocessing=tissue_masks"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_path: list[Path] = load_slide_paths(
        run_id=config.dataset.mlflow_artifacts.mapping_run_id,
        artifact_path=config.dataset.mlflow_artifacts.mapping_filename,
    )

    with TemporaryDirectory() as output_dir:
        process_items(
            slides_path,
            process_item=process_slide,
            fn_kwargs={
                "mpp": config.mpp,
                "output_path": Path(output_dir),
                "disk_factor": config.disk_factor,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(
            local_dir=output_dir, artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
