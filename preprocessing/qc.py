import asyncio
import tempfile
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
import rationai
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.types.qc import SlideCheckConfig
from ratiopath.openslide import OpenSlide
from tqdm.asyncio import tqdm


def load_slide_path(run_id: str, artifact_path: str) -> list[str]:
    """Downloads a dataset map from MLflow and extracts WSI paths."""
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    df = pd.read_csv(local_path)

    return df["wsi_path"].astype(str).tolist()


def get_level_from_mpp(slide_path: str, target_mpp: float) -> int:
    """Calculates the level index for a given MPP."""
    with OpenSlide(slide_path) as slide:
        return slide.closest_level(target_mpp)


def group_slides_by_level(slides: list[str], target_mpp: float) -> dict[int, list[str]]:
    """Groups slides by their calculated level index for a specified MPP."""
    groups: dict[int, list[str]] = defaultdict(list)
    for slide_path in tqdm(slides):
        groups[get_level_from_mpp(slide_path, target_mpp)].append(slide_path)

    return groups


def append_error_log(output_path: str, message: str) -> None:
    """Append one QC failure to the error log.

    Kept synchronous and dispatched with `asyncio.to_thread` by the async caller, so
    the blocking write never stalls the event loop.
    """
    with open(Path(output_path) / "qc_errors.log", "a") as log_file:
        log_file.write(message)


async def qc_main(
    output_path: str,
    report_path: str,
    slides: list[str],
    mask_mpp: float,
    sample_mpp: float,
    bad_slide_names: list[str],
) -> None:
    # Added an exception for one erroneous slide, with damaged data on level 1.
    # This damage fails the WB correction, which can be omitted, since this slide is fairly clean
    processing_groups = [
        (
            [s for s in slides if Path(s).name not in bad_slide_names],
            True,
        ),
        (
            [s for s in slides if Path(s).name in bad_slide_names],
            False,
        ),
    ]

    async with rationai.AsyncClient() as client:  # type: ignore[attr-defined]
        tasks = []

        for group, use_wb in processing_groups:
            level_groups = group_slides_by_level(group, sample_mpp)

            for actual_level, batch in level_groups.items():
                config = SlideCheckConfig(
                    mask_level=get_level_from_mpp(batch[0], mask_mpp),
                    sample_level=actual_level,
                    check_residual=True,
                    check_folding=True,
                    check_focus=True,
                    wb_correction=use_wb,
                )

                tasks.append(client.qc.check_slides(batch, output_path, config=config))

        async for result in tqdm(
            merge_generators(tasks), total=len(slides), desc="QC Progress"
        ):
            if not result.success:
                error_msg = f"Failed to process {result.wsi_path}: {result.error}\n"
                print(f"❌ {error_msg.strip()}")
                await asyncio.to_thread(append_error_log, output_path, error_msg)

        await client.qc.generate_report(
            backgrounds=slides,
            mask_dir=output_path,
            save_location=report_path,
            compute_metrics=True,
        )


async def merge_generators(generators: Iterable[AsyncGenerator[Any, None]]) -> Any:
    """Helper to iterate over multiple async generators sequentially."""
    for gen in generators:
        async for item in gen:
            yield item


@with_cli_args(["+preprocessing=qc"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    slides: list[str] = load_slide_path(
        run_id=config.dataset.mlflow_artifacts.mapping_run_id,
        artifact_path=config.dataset.mlflow_artifacts.mapping_filename,
    )

    bad_slide_names = list(config.dataset.get("exclusions", {}).get("bad_slides", []))

    with tempfile.TemporaryDirectory(
        prefix="qc_masks_report_", dir=output_path.as_posix()
    ) as tmp_dir:
        report_file = Path(tmp_dir) / "report.html"

        asyncio.run(
            qc_main(
                output_path=output_path.absolute().as_posix(),
                report_path=report_file.absolute().as_posix(),
                slides=slides,
                mask_mpp=config.mask_mpp,
                sample_mpp=config.sample_mpp,
                bad_slide_names=bad_slide_names,
            )
        )
        logger.log_artifacts(local_dir=tmp_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
