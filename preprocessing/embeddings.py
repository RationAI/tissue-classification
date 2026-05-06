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
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class EmbedTiles:
    def __init__(self, model: str, concurrency: int) -> None:
        self.model = model
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            ),
            timeout=200,
        )
        self._retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        )
        print(
            f"[EmbedTiles] actor initialized, model={model}, concurrency={concurrency}"
        )

    async def _embed_once(self, tile: Any) -> list[float]:
        return (
            (await self.client.models.embed_image(self.model, tile))
            .reshape(-1)
            .tolist()
        )

    async def _embed(self, tile: Any) -> list[float]:
        return await self._retryer(self._embed_once, tile)

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        try:
            embedding = await self._embed(row["tile"])
        finally:
            del row["tile"]
        row["embedding"] = embedding
        return row


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    for name in ["train", "test"]:
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
                run_id=config.dataset.mlflow_artifacts.filter_tiles_run_id,
                artifact_path=f"filter_tiles/{name}_tiles.parquet",
            )
        )
        num_rows = pads.dataset(str(tiles_path), format="parquet").count_rows()
        num_blocks = max(1, num_rows // config.block_size)

        ds = ray.data.read_parquet(
            str(tiles_path),
            columns=["slide_id", "x", "y"],
            ray_remote_args={"memory": 8 * 1024**3},
            override_num_blocks=num_blocks,
        ).map(
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
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False
    ctx.target_max_block_size = 64 * 1024 * 1024

    with ray.init(
        runtime_env={"excludes": [".git", ".venv"]},
        object_store_memory=16 * 1024**3,
    ):
        main()
