import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from preprocessing._tiff_utils import rewrite_tiff


@ray.remote(num_cpus=1, memory=(4 * 1024**3))
def remap_mask(
    item: str,
    output_dir: str,
    lut: list[int],
) -> None:
    src_path = Path(item)
    dst_path = Path(output_dir) / src_path.name
    lut_array = np.array(lut, dtype=np.uint8)
    rewrite_tiff(src_path, dst_path, transform=lambda data: lut_array[data])


@with_cli_args(["+preprocessing=remap_annotation_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    n_classes = config.n_classes

    # Build LUT: evenly spread class values across [1, 255], background (255) -> 0
    lut = [0] * 256
    for i in range(n_classes):
        lut[i] = round(255 * (i + 1) / n_classes)
    lut[255] = 0

    local_masks_dir = Path(
        download_artifacts(
            run_id=config.source_run_id, artifact_path=config.source_artifact_path
        )
    )
    mask_files = sorted(local_masks_dir.glob("*.tiff"))

    with TemporaryDirectory(dir=Path.cwd()) as output_dir:
        for f in local_masks_dir.iterdir():
            if f.suffix != ".tiff":
                shutil.copy2(f, Path(output_dir) / f.name)

        process_items(
            [str(p) for p in mask_files],
            process_item=remap_mask,
            fn_kwargs={
                "output_dir": output_dir,
                "lut": lut,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(
            local_dir=output_dir,
            artifact_path=config.mlflow_artifact_path,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
