from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import ray
import tifffile
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(num_cpus=1, memory=(4 * 1024**3))
def remap_mask(
    item: str,
    output_dir: str,
    lut: list[int],
) -> None:
    src_path = Path(item)
    dst_path = Path(output_dir) / src_path.name
    lut_array = np.array(lut, dtype=np.uint8)

    with tifffile.TiffFile(str(src_path)) as tif:
        num_pages = len(tif.pages)

        with tifffile.TiffWriter(str(dst_path), bigtiff=tif.is_bigtiff) as writer:
            for page_idx, page in enumerate(tif.pages):
                data = page.asarray()
                if data.ndim == 3:
                    data = data[..., 0]

                data = lut_array[data]

                write_kwargs = {
                    "photometric": "minisblack",
                    "compression": page.compression,
                }

                if page.is_tiled:
                    write_kwargs["tile"] = (page.tilelength, page.tilewidth)

                pred_tag = page.tags.get(317)
                if pred_tag and pred_tag.value > 1:
                    write_kwargs["predictor"] = pred_tag.value

                xres_tag = page.tags.get(282)
                yres_tag = page.tags.get(283)
                res_unit_tag = page.tags.get(296)
                if xres_tag and yres_tag:
                    write_kwargs["resolution"] = (xres_tag.value, yres_tag.value)
                    if res_unit_tag:
                        write_kwargs["resolutionunit"] = res_unit_tag.value

                if page_idx > 0:
                    write_kwargs["subfiletype"] = 1  # REDUCEDIMAGE

                writer.write(data, **write_kwargs)


@with_cli_args(["+preprocessing=remap_annotation_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    source_run_id = config.source_run_id
    artifact_path = config.source_artifact_path
    n_classes = config.n_classes

    # Build LUT: evenly spread class values across [1, 255], background (255) -> 0
    lut = [0] * 256
    for i in range(n_classes):
        lut[i] = round(255 * (i + 1) / n_classes)
    lut[255] = 0

    local_masks_dir = Path(download_artifacts(run_id=source_run_id, artifact_path=artifact_path))
    mask_files = sorted(local_masks_dir.glob("*.tiff"))

    with TemporaryDirectory(dir=Path.cwd()) as output_dir:
        # Copy non-TIFF files (e.g. missing_annotations.csv) as-is
        import shutil
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
