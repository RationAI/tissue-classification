import tempfile
import time
from pathlib import Path

import hydra
import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import ray
import tifffile
from datasets import load_dataset
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def _log(msg: str) -> None:
    print(f"[tile_stats {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_parquet_artifact(run_id: str, artifact_path: str) -> pd.DataFrame:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    return pd.read_parquet(local_path)


def load_tiles_columns(run_id: str, artifact_path: str, columns: list[str]) -> pa.Table:
    """Load the specified columns of a tiles parquet as a memory-mapped Arrow table.

    HF datasets caches the parquet into Arrow IPC on first load, after which the data
    stays memory-mapped on disk. We return the underlying pyarrow Table without going
    through pandas — materializing 80M rows into a DataFrame is the dominant cost, and
    we only need to convert small per-slide slices before dispatching Ray tasks.
    """
    t0 = time.perf_counter()
    _log(f"downloading tiles artifact {artifact_path}...")
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    _log(f"  download done in {time.perf_counter() - t0:.1f}s -> {local_path}")

    t1 = time.perf_counter()
    _log("calling load_dataset (HF cache build if first run)...")
    dataset = load_dataset("parquet", data_files=local_path, split="train")
    _log(f"  load_dataset done in {time.perf_counter() - t1:.1f}s, num_rows={dataset.num_rows}")

    t2 = time.perf_counter()
    table = dataset.select_columns(columns).data.table
    _log(
        f"  select_columns done in {time.perf_counter() - t2:.1f}s, "
        f"chunks={[table.column(c).num_chunks for c in columns]}"
    )
    return table


def sat_coverage(
    sat: np.ndarray,
    mask_h: int,
    mask_w: int,
    y0: np.ndarray,
    x0: np.ndarray,
    extent: int,
) -> np.ndarray:
    """Compute mean foreground fraction over [y0:y0+extent, x0:x0+extent] rectangles using a SAT.

    Rectangles that partially fall outside the mask are cropped; rectangles fully outside return 0.
    """
    y1 = y0 + extent
    x1 = x0 + extent

    cy0 = np.clip(y0, 0, mask_h)
    cx0 = np.clip(x0, 0, mask_w)
    cy1 = np.clip(y1, 0, mask_h)
    cx1 = np.clip(x1, 0, mask_w)

    areas = (cy1 - cy0) * (cx1 - cx0)
    sums = sat[cy1, cx1] - sat[cy0, cx1] - sat[cy1, cx0] + sat[cy0, cx0]
    return np.where(areas > 0, sums / areas, 0.0)


def compute_tissue_coverages(
    tiles: pd.DataFrame,
    mask: np.ndarray,
    tile_mpp: float,
    tile_extent: int,
    tissue_mpp: float,
) -> pd.DataFrame:
    """Add tile_tissue_coverage and roi_tissue_coverage columns to tiles for a single slide.

    Uses a summed area table for O(1) per-tile rectangle queries.
    The ROI is the central half-size region of the tile (matching the roi_coverage_{class} convention).
    """
    import os
    mask_h, mask_w = mask.shape
    t0 = time.perf_counter()

    sat = np.zeros((mask_h + 1, mask_w + 1), dtype=np.int32)
    sat[1:, 1:] = mask > 0
    np.cumsum(sat, axis=0, out=sat)
    np.cumsum(sat, axis=1, out=sat)
    print(
        f"[compute pid={os.getpid()}] SAT built in {time.perf_counter() - t0:.1f}s "
        f"mask={mask.shape} sat_bytes={sat.nbytes / 1e6:.1f}MB",
        flush=True,
    )

    t1 = time.perf_counter()
    scale = tile_mpp / tissue_mpp
    tm_extent = max(1, round(tile_extent * scale))
    roi_offset = tm_extent // 4
    roi_extent = max(1, tm_extent // 2)

    xs = np.round(tiles["x"].values * scale).astype(int)
    ys = np.round(tiles["y"].values * scale).astype(int)

    tile_cov = sat_coverage(sat, mask_h, mask_w, ys, xs, tm_extent)
    roi_cov = sat_coverage(sat, mask_h, mask_w, ys + roi_offset, xs + roi_offset, roi_extent)
    print(
        f"[compute pid={os.getpid()}] coverage done in {time.perf_counter() - t1:.1f}s "
        f"tiles={len(tiles):,} tm_extent={tm_extent}",
        flush=True,
    )

    tiles = tiles.copy()
    tiles["tile_tissue_coverage"] = tile_cov
    tiles["roi_tissue_coverage"] = roi_cov
    return tiles


@ray.remote(num_cpus=1, max_calls=1)
def process_slide(
    slide_tiles: pd.DataFrame,
    mask_path: str,
    tile_extent: int,
    tile_mpp: float,
    tissue_mpp: float,
) -> pd.DataFrame:
    import os
    t0 = time.perf_counter()
    print(
        f"[worker pid={os.getpid()}] start mask={Path(mask_path).name} "
        f"tiles={len(slide_tiles):,}",
        flush=True,
    )
    mask = tifffile.imread(mask_path)
    t_read = time.perf_counter() - t0
    if mask.ndim > 2:
        mask = mask[..., 0]
    print(
        f"[worker pid={os.getpid()}] mask loaded in {t_read:.1f}s "
        f"shape={mask.shape} dtype={mask.dtype}",
        flush=True,
    )
    t1 = time.perf_counter()
    out = compute_tissue_coverages(slide_tiles, mask, tile_mpp, tile_extent, tissue_mpp)
    print(
        f"[worker pid={os.getpid()}] done mask={Path(mask_path).name} "
        f"compute={time.perf_counter() - t1:.1f}s total={time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def add_tissue_coverage(
    slides: pd.DataFrame,
    tiles: pa.Table,
    tissue_mask_dir: Path,
    tissue_mpp: float,
    output_path: Path,
) -> dict[str, float]:
    """Dispatch per-slide tissue coverage computation as Ray tasks.

    Writes per-slide results to `output_path` via a streaming ParquetWriter as each Ray
    task finishes, so the driver never holds more than one slide's DataFrame at a time.
    Returns the aggregate stats (total count, mean coverages) for metric logging.

    Tiling emits slides interleaved across Ray Data blocks, so we dictionary-encode
    slide_id and argsort the int indices to group row positions per slide. Avoids the
    pyarrow int32-offset overflow that a full string-column sort/take would hit on
    ~80M rows, while keeping per-slide takes small enough to stay under the limit.
    """
    slide_info = slides.set_index("id")[["path", "tile_extent_x", "mpp_x"]]
    _log(f"add_tissue_coverage: {len(tiles):,} tiles, {len(slide_info)} slides in catalog")

    t0 = time.perf_counter()
    encoded = pc.dictionary_encode(tiles.column("slide_id"))
    _log(f"  dictionary_encode done in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.unify_dictionaries().combine_chunks()
    indices = encoded.indices.to_numpy(zero_copy_only=False)
    dictionary = encoded.dictionary
    _log(
        f"  unify+to_numpy done in {time.perf_counter() - t1:.1f}s, "
        f"unique_slides_in_tiles={len(dictionary)}"
    )

    t2 = time.perf_counter()
    sort_order = np.argsort(indices, kind="stable")
    sorted_indices = indices[sort_order]
    change_points = np.where(sorted_indices[1:] != sorted_indices[:-1])[0] + 1
    boundaries = np.concatenate([[0], change_points, [len(sorted_indices)]])
    _log(
        f"  argsort+boundaries done in {time.perf_counter() - t2:.1f}s, "
        f"groups={len(boundaries) - 1}"
    )

    t_xy = time.perf_counter()
    x_all = tiles.column("x").combine_chunks().to_numpy(zero_copy_only=False)
    y_all = tiles.column("y").combine_chunks().to_numpy(zero_copy_only=False)
    _log(
        f"  x/y to numpy done in {time.perf_counter() - t_xy:.1f}s "
        f"x_dtype={x_all.dtype} bytes={(x_all.nbytes + y_all.nbytes) / 1e6:.0f}MB"
    )

    t3 = time.perf_counter()
    futures = []
    missing_slides = []

    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        slide_id = dictionary[int(sorted_indices[start])].as_py()

        if slide_id not in slide_info.index:
            missing_slides.append(slide_id)
            continue

        info = slide_info.loc[slide_id]
        wsi_path = Path(str(info["path"]))
        tile_extent = int(info["tile_extent_x"])
        tile_mpp = float(info["mpp_x"])

        mask_path = tissue_mask_dir / wsi_path.with_suffix(".tiff").name
        if not mask_path.exists():
            missing_slides.append(slide_id)
            continue

        positions = sort_order[start:end]
        slide_tiles = pd.DataFrame({"x": x_all[positions], "y": y_all[positions]})
        futures.append(
            process_slide.remote(slide_tiles, str(mask_path), tile_extent, tile_mpp, tissue_mpp)
        )

    _log(
        f"  dispatch loop done in {time.perf_counter() - t3:.1f}s, "
        f"futures={len(futures)}, missing={len(missing_slides)}"
    )

    if missing_slides:
        print(f"Warning: {len(missing_slides)} slides had no matching tissue mask and were dropped")

    if not futures:
        raise RuntimeError(
            f"No tissue masks matched any slide. Check tissue_masks_run_id and artifact path. "
            f"Missing slides: {len(missing_slides)}"
        )

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    count = 0
    tile_cov_sum = 0.0
    roi_cov_sum = 0.0

    t4 = time.perf_counter()
    total = len(futures)
    done_count = 0
    pending = list(futures)
    _log(f"  ray cluster_resources={ray.cluster_resources()}")
    _log(f"  ray available_resources={ray.available_resources()}")
    try:
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=15.0)
            if not done:
                _log(
                    f"  ray.wait 15s timeout: pending={len(pending)} "
                    f"available_resources={ray.available_resources()}"
                )
                continue
            t_get = time.perf_counter()
            result = ray.get(done[0])
            _log(
                f"  task done: ray.get={time.perf_counter() - t_get:.2f}s "
                f"rows={len(result)}"
            )
            t_pa = time.perf_counter()
            if schema is None:
                table = pa.Table.from_pandas(result, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(str(output_path), schema)
            else:
                table = pa.Table.from_pandas(result, preserve_index=False, schema=schema)
            t_write = time.perf_counter()
            writer.write_table(table)
            t_sum = time.perf_counter()
            count += len(result)
            tile_cov_sum += float(result["tile_tissue_coverage"].sum())
            roi_cov_sum += float(result["roi_tissue_coverage"].sum())
            done_count += 1
            _log(
                f"  ray progress {done_count}/{total} "
                f"(arrow={t_write - t_pa:.2f}s write={t_sum - t_write:.2f}s "
                f"sum={time.perf_counter() - t_sum:.2f}s, "
                f"{time.perf_counter() - t4:.1f}s elapsed, {count:,} tiles written)"
            )
    finally:
        if writer is not None:
            writer.close()
    _log(f"  ray loop done in {time.perf_counter() - t4:.1f}s")

    return {
        "tile_count": count,
        "mean_tile_tissue_coverage": tile_cov_sum / count,
        "mean_roi_tissue_coverage": roi_cov_sum / count,
    }


@with_cli_args(["+preprocessing=tile_stats"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_run_id = config.dataset.mlflow_artifacts.tiling_run_id
    tissue_masks_run_id = config.dataset.mlflow_artifacts.tissue_masks_run_id

    t0 = time.perf_counter()
    _log(f"downloading tissue_masks artifact from run {tissue_masks_run_id}...")
    tissue_mask_dir = Path(
        mlflow.artifacts.download_artifacts(
            run_id=tissue_masks_run_id,
            artifact_path="tissue_masks",
        )
    )
    _log(f"  tissue_masks download done in {time.perf_counter() - t0:.1f}s -> {tissue_mask_dir}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split_name in ("train", "test"):
            split_t = time.perf_counter()
            _log(f"=== split={split_name} ===")
            slides = load_parquet_artifact(
                tiling_run_id, f"{split_name}_split/slides.parquet"
            )
            _log(f"loaded slides parquet, rows={len(slides)}")
            tiles = load_tiles_columns(
                tiling_run_id,
                f"{split_name}_split/tiles.parquet",
                columns=["slide_id", "x", "y"],
            )

            output_path = Path(tmp_dir) / f"{split_name}_tiles.parquet"
            stats = add_tissue_coverage(
                slides, tiles, tissue_mask_dir, config.tissue_mpp, output_path
            )
            _log(f"split={split_name} total={time.perf_counter() - split_t:.1f}s")

            mlflow.log_metric(f"{split_name}_tile_count", stats["tile_count"])
            mlflow.log_metric(
                f"{split_name}_mean_tile_tissue_coverage",
                stats["mean_tile_tissue_coverage"],
            )
            mlflow.log_metric(
                f"{split_name}_mean_roi_tissue_coverage",
                stats["mean_roi_tissue_coverage"],
            )

        mlflow.log_artifacts(tmp_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    with ray.init(num_cpus=4):
        main()
