import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import ray
import tifffile
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.masks import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def _fix_tiff_icc(src_path: Path, dst_path: Path) -> None:
    """Re-write a grayscale TIFF, stripping the ICC profile.

    Preserves ALL pyramid levels, pixel data, compression, tiling,
    predictor, and resolution metadata.
    """
    with tifffile.TiffFile(str(src_path)) as tif:
        if tif.pages.first.tags.get(34675) is None:
            shutil.copy2(src_path, dst_path)
            return

        with tifffile.TiffWriter(str(dst_path), bigtiff=tif.is_bigtiff) as writer:
            for page_idx, page in enumerate(tif.pages):
                assert isinstance(page, tifffile.TiffPage)
                data = page.asarray()
                if data.ndim == 3:
                    data = data[..., 0]

                write_kwargs: dict[str, Any] = {
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


@ray.remote(num_cpus=1, memory=(2 * 1024**3))
def process_mask(item: tuple[Path, Path]) -> None:
    src_path, dst_path = item
    _fix_tiff_icc(src_path, dst_path)


@with_cli_args(["+preprocessing=fix_mask_icc_profile"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    download_dir = Path(
        download_artifacts(
            run_id=config.source_run_id, artifact_path=config.artifact_dir
        )
    )

    with TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        fixed_dir = Path(tmp_dir) / "fixed"
        fixed_dir.mkdir()

        files = sorted(download_dir.glob("*.tiff"))
        if not files:
            raise RuntimeError(
                f"No .tiff files found in artifact dir '{config.artifact_dir}' of run {config.source_run_id}."
            )

        items = [(src, fixed_dir / src.name) for src in files]

        process_items(
            items,
            process_item=process_mask,
            fn_kwargs={},
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(
            local_dir=str(fixed_dir), artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
