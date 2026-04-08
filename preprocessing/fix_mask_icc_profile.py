import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import mlflow
import tifffile
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def fix_tiff_icc(src_path: Path, dst_path: Path) -> str:
    """Re-write a grayscale TIFF, stripping the ICC profile.

    Preserves ALL pyramid levels, pixel data, compression, tiling,
    predictor, and resolution metadata.
    Returns a short status string.
    """
    with tifffile.TiffFile(str(src_path)) as tif:
        first_page = tif.pages.first
        if first_page.tags.get(34675) is None:
            shutil.copy2(src_path, dst_path)
            return "skipped (no ICC)"

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

    return f"fixed (ICC stripped, {len(tif.pages)} pages preserved)"


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
        print(f"Processing {len(files)} masks...")

        stats: dict[str, float] = {"fixed": 0, "skipped": 0, "errors": 0}
        for src_path in files:
            try:
                status = fix_tiff_icc(src_path, fixed_dir / src_path.name)
                if "fixed" in status:
                    stats["fixed"] += 1
                else:
                    stats["skipped"] += 1
                print(f"  {src_path.name}: {status}")
            except Exception as e:
                stats["errors"] += 1
                print(f"  ERROR {src_path.name}: {e}")

        mlflow.log_metrics(stats)

        logger.log_artifacts(
            local_dir=str(fixed_dir), artifact_path=config.mlflow_artifact_path
        )

        if stats["errors"] > 0:
            raise RuntimeError(
                f"{stats['errors']} mask(s) failed to process; see logs for details."
            )


if __name__ == "__main__":
    main()
