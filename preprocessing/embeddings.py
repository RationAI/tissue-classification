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
from rationai import AsyncClient
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


class JoinSlideInfo:
    def __init__(self, slide_info: dict) -> None:
        self.slide_info = slide_info
        self.count = 0
        self.start = time.monotonic()

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.count == 0:
            print(f"[JoinSlideInfo] first row at t={time.monotonic() - self.start:.2f}s")
        self.count += 1
        if self.count % 50000 == 0:
            elapsed = time.monotonic() - self.start
            print(f"[JoinSlideInfo] {self.count} rows in {elapsed:.1f}s ({self.count / elapsed:.0f} rows/s)")
        return {**row, **self.slide_info[row["slide_id"]]}


class EmbedTiles:
    def __init__(self, model: str, concurrency: int) -> None:
        self.model = model
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            )
        )
        self.count = 0
        self.start = time.monotonic()
        print(f"[EmbedTiles] actor initialized, model={model}, concurrency={concurrency}")

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.count == 0:
            print(f"[EmbedTiles] first row at t={time.monotonic() - self.start:.2f}s")
        t0 = time.monotonic()
        embedding = (
            (await self.client.models.embed_image(self.model, row["tile"]))
            .reshape(-1)
            .tolist()
        )
        latency = time.monotonic() - t0
        self.count += 1
        if self.count % 100 == 0:
            elapsed = time.monotonic() - self.start
            print(f"[EmbedTiles] {self.count} embeddings in {elapsed:.1f}s ({self.count / elapsed:.1f}/s, last latency={latency * 1000:.0f}ms)")
        del row["tile"]
        row["embedding"] = embedding
        return row


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    run_id = config.dataset.mlflow_artifacts.tiling_run_id
    for name in ["train", "test"]:
        t = time.monotonic()
        print(f"[main] === split={name} ===")
        folder = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{name}_split"
            )
        )
        print(f"[main] artifacts downloaded in {time.monotonic() - t:.1f}s")

        t = time.monotonic()
        slides = pd.read_parquet(folder / "slides.parquet")
        slide_info = slides.set_index("id")[
            ["path", "level", "tile_extent_x", "tile_extent_y"]
        ].to_dict("index")
        print(f"[main] loaded {len(slide_info)} slides in {time.monotonic() - t:.1f}s")

        t = time.monotonic()
        tiles_path = folder / "tiles.parquet"
        num_rows = pads.dataset(str(tiles_path), format="parquet").count_rows()
        num_blocks = max(1, num_rows // config.block_size)
        print(f"[main] {num_rows} tile rows, {num_blocks} blocks (parquet metadata in {time.monotonic() - t:.1f}s)")

        ds = ray.data.read_parquet(
            str(tiles_path),
            ray_remote_args={"memory": 8 * 1024**3},
            override_num_blocks=num_blocks,
        ).map(
            JoinSlideInfo,  # pyright: ignore[reportArgumentType]
            fn_constructor_args=(slide_info,),
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
            num_cpus=0.25,
            memory=1 * 1024**3,
        )
        ds = ds.drop_columns(["path", "level", "tile_extent_x", "tile_extent_y"])
        ds = ds.map(
            EmbedTiles,  # pyright: ignore[reportArgumentType]
            fn_constructor_args=(config.model, config.concurrency),
            compute=ray.data.ActorPoolStrategy(
                max_size=4,
                max_tasks_in_flight_per_actor=config.concurrency // 4,
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
        object_store_memory=32 * 1024**3,
    ):
        main()
