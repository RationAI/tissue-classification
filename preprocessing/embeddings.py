import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import hydra
import mlflow.artifacts
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import ray
from omegaconf import DictConfig
from rationai import AsyncClient  # type: ignore[attr-defined]
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


def filter_tiles(folder: Path, config: DictConfig, split_name: str) -> Path:
    """Drop tiles that will never be used for training and return the parquet path to use.

    Removes tiles where all annotation class coverages are zero (unlabeled) and
    tiles with zero tissue coverage. Uses PyArrow predicate pushdown so only
    matching rows are materialised — safe for 80M-row datasets.
    """
    tiles_path = folder / "tiles.parquet"
    tiles_ds = pads.dataset(str(tiles_path), format="parquet")
    original_count = tiles_ds.count_rows()

    ann_cols = [f.name for f in tiles_ds.schema if f.name.startswith("tile_coverage_")]
    ann_filter = None
    if ann_cols:
        ann_filter = pads.field(ann_cols[0]) > 0
        for c in ann_cols[1:]:
            ann_filter = pc.or_(ann_filter, pads.field(c) > 0)

    tiles_table = tiles_ds.to_table(filter=ann_filter)
    ann_count = len(tiles_table)
    print(
        f"[filter_tiles] {split_name}: annotation filter"
        f" {original_count} → {ann_count} ({ann_count / original_count:.1%} kept)"
    )

    tissue_local = mlflow.artifacts.download_artifacts(
        run_id=config.tile_filters.tissue_stats_run_id,
        artifact_path=f"{config.tile_filters.tissue_stats_artifact_path}/{split_name}_tiles.parquet",
    )
    tissue_table = pads.dataset(tissue_local, format="parquet").to_table(
        columns=["slide_id", "x", "y"],
        filter=pads.field("tile_tissue_coverage") > 0,
    )
    tiles_table = tiles_table.join(
        tissue_table, keys=["slide_id", "x", "y"], join_type="inner"
    )
    tissue_count = len(tiles_table)
    print(
        f"[filter_tiles] {split_name}: tissue filter"
        f" {ann_count} → {tissue_count} ({tissue_count / ann_count:.1%} kept)"
    )

    if tissue_count == original_count:
        return tiles_path

    filtered_path = folder / "tiles_filtered.parquet"
    pq.write_table(tiles_table, str(filtered_path))
    return filtered_path


class EmbedTiles:
    def __init__(self, model: str, concurrency: int) -> None:
        self.model = model
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency, max_keepalive_connections=concurrency
            ),
            timeout=200,
        )
        self.count = 0
        self.in_flight = 0
        self.first_row_logged = False
        self.start = time.monotonic()
        print(
            f"[EmbedTiles] actor initialized, model={model}, concurrency={concurrency}"
        )

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        if not self.first_row_logged:
            self.first_row_logged = True
            print(f"[EmbedTiles] first row at t={time.monotonic() - self.start:.2f}s")
        self.in_flight += 1
        t0 = time.monotonic()
        try:
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    embedding = (
                        (await self.client.models.embed_image(self.model, row["tile"]))
                        .reshape(-1)
                        .tolist()
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        await asyncio.sleep(2**attempt)
            else:
                raise RuntimeError("embed_image failed after 3 attempts") from last_exc
        finally:
            del row["tile"]
        latency = time.monotonic() - t0
        self.count += 1
        self.in_flight -= 1
        if self.count % 50 == 0:
            elapsed = time.monotonic() - self.start
            print(
                f"[EmbedTiles] {self.count} done in {elapsed:.1f}s ({self.count / elapsed:.1f}/s, in_flight={self.in_flight}, latency={latency * 1000:.0f}ms)"
            )
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
        tiles_path = filter_tiles(folder, config, name)
        num_rows = pads.dataset(str(tiles_path), format="parquet").count_rows()
        num_blocks = max(1, num_rows // config.block_size)
        print(
            f"[main] {num_rows} tile rows, {num_blocks} blocks (filter+metadata in {time.monotonic() - t:.1f}s)"
        )

        ds = ray.data.read_parquet(
            str(tiles_path),
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
        object_store_memory=16 * 1024**3,
    ):
        main()
