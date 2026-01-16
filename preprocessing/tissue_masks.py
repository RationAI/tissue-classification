from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
import pyvips
import ray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import process_items, slide_resolution, tissue_mask, write_big_tiff
from rationai.masks.vips_filters import (
    VipsClosing,
    VipsCompose,
    VipsGrayScaleFilter,
    VipsOpening,
    VipsOtsu,
)
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(memory=3 * 1024**3)
def process_slide(
    slide_path: Path, level: int, output_path: Path, disk_factor: int
) -> None:
    with OpenSlide(slide_path) as slide:
        # Handle slides with missing pyramid levels (e.g. MUG dataset).
        # Safely fallback to the last available level instead of crashing.
        if level >= slide.level_count:
            level = slide.level_count - 1
        mpp_x, mpp_y = slide_resolution(slide, level=level)

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


@hydra.main(
    config_path="../configs",
    config_name="preprocessing/tissue_masks",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides_path: list[Path] = load_slide_paths(
        run_id=config.mapping_run_id, artifact_path=config.mapping_artifact_path
    )

    with TemporaryDirectory() as output_dir:
        process_items(
            slides_path,
            process_item=process_slide,
            fn_kwargs={
                "level": config.level,
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
