import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import hydra
import mlflow.artifacts
import pandas as pd
import pyarrow.dataset as pads
import ray
from omegaconf import DictConfig
from rationai import AsyncClient  # type: ignore[attr-defined]
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


class EmbedTiles:
    def __init__(self, model: str, concurrency: int) -> None:
        self.model = model
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            ),
            timeout=200,
        )

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            last_exc: httpx.HTTPError | None = None
            for attempt in range(6):
                try:
                    embedding = (
                        (await self.client.models.embed_image(self.model, row["tile"]))
                        .reshape(-1)
                        .tolist()
                    )
                    break
                except httpx.HTTPError as exc:
                    last_exc = exc
                    if attempt < 5:
                        await asyncio.sleep(min(60, 2**attempt))
            else:
                raise RuntimeError("embed_image failed after 6 attempts") from last_exc
        finally:
            del row["tile"]
        row["embedding"] = embedding
        return row


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tile_source_run_id = config.get(
        "tile_source_run_id", config.dataset.mlflow_artifacts.filter_tiles_run_id
    )
    tile_source_artifact_template = config.get(
        "tile_source_artifact_template", "filter_tiles/{split}_tiles.parquet"
    )
    tile_filter_column = config.get("tile_filter_column")

    for name in config.get("splits", ["train", "test"]):
        split_folder = Path(
            mlflow.artifacts.download_artifacts(
                run_id=config.dataset.mlflow_artifacts.tiling_run_id,
                artifact_path=f"{name}_split",
            )
        )
        slides = pd.read_parquet(split_folder / "slides.parquet")
        slide_info = slides.set_index("id")[
            ["path", "level", "tile_extent_x", "tile_extent_y"]
        ].to_dict("index")

        tiles_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=tile_source_run_id,
                artifact_path=tile_source_artifact_template.format(split=name),
            )
        )
        row_filter = (
            pads.field(tile_filter_column) > 0
            if tile_filter_column is not None
            else None
        )
        num_rows = pads.dataset(str(tiles_path), format="parquet").count_rows(
            filter=row_filter
        )
        num_blocks = max(1, num_rows // config.block_size)

        columns = ["slide_id", "x", "y"]
        if tile_filter_column is not None:
            columns.append(tile_filter_column)
        ds = ray.data.read_parquet(
            str(tiles_path),
            columns=columns,
            ray_remote_args={"memory": 8 * 1024**3},
            override_num_blocks=num_blocks,
        )
        if tile_filter_column is not None:
            ds = ds.filter(
                lambda row, c: row[c] > 0, fn_kwargs={"c": tile_filter_column}
            )
            ds = ds.drop_columns([tile_filter_column])
        ds = ds.map(
            lambda row, si: {**row, **si[row["slide_id"]]},
            fn_kwargs={"si": slide_info},
        )
        ds = ds.with_column(
            "tile",
            read_slide_tiles(  # pyright: ignore[reportCallIssue]
                col("path"),
                col("x"),
                col("y"),
                col("tile_extent_x"),
                col("tile_extent_y"),
                col("level"),
            ),
            num_cpus=1,
            memory=4 * 1024**3,
        )
        ds = ds.drop_columns(["path", "level", "tile_extent_x", "tile_extent_y"])
        ds = ds.map(
            EmbedTiles,  # pyright: ignore[reportArgumentType]
            fn_constructor_args=(config.model, config.concurrency),
            compute=ray.data.ActorPoolStrategy(
                min_size=4,
                max_size=4,
                max_tasks_in_flight_per_actor=max(1, config.concurrency // 4),
            ),
            max_concurrency=config.concurrency,
        )

        split_dir = Path(config.output_dir) / str(name)
        split_dir.mkdir(parents=True, exist_ok=True)
        tiles_parquet_dir = split_dir / "tiles"
        if tiles_parquet_dir.exists():
            shutil.rmtree(tiles_parquet_dir)

        slides.to_parquet(split_dir / "slides.parquet", index=False)

        t = time.monotonic()
        print(f"[main] starting write_parquet for split={name}")
        ds.write_parquet(str(tiles_parquet_dir), min_rows_per_file=config.rows_per_file)
        print(f"[main] write_parquet finished in {time.monotonic() - t:.1f}s")

        logger.log_artifacts(str(split_dir), str(name))


if __name__ == "__main__":
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = True
    ctx.target_max_block_size = 64 * 1024 * 1024

    with ray.init(
        runtime_env={"excludes": [".git", ".venv"]},
        object_store_memory=16 * 1024**3,
    ):
        main()
