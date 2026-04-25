import tempfile
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


def load_parquet_artifact(run_id: str, artifact_path: str) -> pd.DataFrame:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    return pd.read_parquet(local_path)


def load_tiles_columns(run_id: str, artifact_path: str, columns: list[str]) -> pa.Table:
    """Load a tiles parquet column subset as a memory-mapped Arrow table.

    HF datasets caches the parquet into Arrow IPC on first load. We return the
    underlying pyarrow Table without materializing 80M rows into pandas — only
    small per-slide slices are converted before dispatching Ray tasks.
    """
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    dataset = load_dataset("parquet", data_files=local_path, split="train")
    return dataset.select_columns(columns).data.table


def sat_coverage(
    sat: np.ndarray,
    mask_h: int,
    mask_w: int,
    y0: np.ndarray,
    x0: np.ndarray,
    extent: int,
) -> np.ndarray:
    """Compute foreground fraction over [y0:y0+extent, x0:x0+extent] rectangles using a SAT.

    Coverage is normalized by the full rectangle area, so tiles partially outside the
    mask are penalized in proportion to how much of them sits outside.
    """
    cy0 = np.clip(y0, 0, mask_h)
    cx0 = np.clip(x0, 0, mask_w)
    cy1 = np.clip(y0 + extent, 0, mask_h)
    cx1 = np.clip(x0 + extent, 0, mask_w)

    sums = sat[cy1, cx1] - sat[cy0, cx1] - sat[cy1, cx0] + sat[cy0, cx0]
    return sums / (extent * extent)


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
    mask_h, mask_w = mask.shape
    sat = np.zeros((mask_h + 1, mask_w + 1), dtype=np.int32)
    sat[1:, 1:] = mask > 0
    np.cumsum(sat, axis=0, out=sat)
    np.cumsum(sat, axis=1, out=sat)

    scale = tile_mpp / tissue_mpp
    tm_extent = max(1, round(tile_extent * scale))
    roi_offset = tm_extent // 4
    roi_extent = max(1, tm_extent // 2)

    xs = np.round(tiles["x"].to_numpy() * scale).astype(int)
    ys = np.round(tiles["y"].to_numpy() * scale).astype(int)

    tile_cov = sat_coverage(sat, mask_h, mask_w, ys, xs, tm_extent)
    roi_cov = sat_coverage(
        sat, mask_h, mask_w, ys + roi_offset, xs + roi_offset, roi_extent
    )

    tiles = tiles.copy()
    tiles["tile_tissue_coverage"] = tile_cov
    tiles["roi_tissue_coverage"] = roi_cov
    return tiles


# max_calls=1 restarts the worker between tasks so RSS doesn't accumulate across
# successive multi-GB SATs (Python doesn't return freed memory to the OS).
@ray.remote(num_cpus=1, max_calls=1)
def process_slide(
    slide_tiles: pd.DataFrame,
    mask_path: str,
    tile_extent: int,
    tile_mpp: float,
    tissue_mpp: float,
) -> pd.DataFrame:
    mask = tifffile.imread(mask_path)
    if mask.ndim > 2:
        mask = mask[..., 0]
    return compute_tissue_coverages(
        slide_tiles, mask, tile_mpp, tile_extent, tissue_mpp
    )


def add_tissue_coverage(
    slides: pd.DataFrame,
    tiles: pa.Table,
    tissue_mask_dir: Path,
    tissue_mpp: float,
    output_path: Path,
) -> dict[str, float]:
    """Dispatch per-slide tissue coverage computation as Ray tasks.

    Writes per-slide results to `output_path` via a streaming ParquetWriter as each
    Ray task finishes, so the driver never holds more than one slide's DataFrame
    at a time. Returns aggregate stats (total count, mean coverages) for metric logging.

    Tiling emits slides interleaved across blocks. We dictionary-encode slide_id
    and argsort the int indices to group row positions per slide — a full
    string-column sort/take on 80M rows would overflow pyarrow's int32 string offsets.
    """
    slide_info = slides.set_index("id")[["path", "tile_extent_x", "mpp_x"]]

    encoded = pc.dictionary_encode(tiles.column("slide_id"))
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.unify_dictionaries().combine_chunks()
    indices = encoded.indices.to_numpy(zero_copy_only=False)
    dictionary = encoded.dictionary

    sort_order = np.argsort(indices, kind="stable")
    sorted_indices = indices[sort_order]
    change_points = np.where(sorted_indices[1:] != sorted_indices[:-1])[0] + 1
    boundaries = np.concatenate([[0], change_points, [len(sorted_indices)]])

    x_all = tiles.column("x").combine_chunks().to_numpy(zero_copy_only=False)
    y_all = tiles.column("y").combine_chunks().to_numpy(zero_copy_only=False)

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
        slide_tiles = pd.DataFrame(
            {
                "slide_id": slide_id,
                "x": x_all[positions],
                "y": y_all[positions],
            }
        )
        futures.append(
            process_slide.remote(
                slide_tiles, str(mask_path), tile_extent, tile_mpp, tissue_mpp
            )
        )

    if missing_slides:
        print(
            f"Warning: {len(missing_slides)} slides had no matching tissue mask and were dropped"
        )

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

    pending = list(futures)
    try:
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            result = ray.get(done[0])
            if writer is None:
                table = pa.Table.from_pandas(result, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(str(output_path), schema)
            else:
                table = pa.Table.from_pandas(
                    result, preserve_index=False, schema=schema
                )
            writer.write_table(table)
            count += len(result)
            tile_cov_sum += float(result["tile_tissue_coverage"].sum())
            roi_cov_sum += float(result["roi_tissue_coverage"].sum())
    finally:
        if writer is not None:
            writer.close()

    return {
        "tile_count": count,
        "mean_tile_tissue_coverage": tile_cov_sum / count,
        "mean_roi_tissue_coverage": roi_cov_sum / count,
    }


@with_cli_args(["+preprocessing=tissue_stats"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_run_id = config.dataset.mlflow_artifacts.tiling_run_id
    tissue_masks_run_id = config.dataset.mlflow_artifacts.tissue_masks_run_id

    tissue_mask_dir = Path(
        mlflow.artifacts.download_artifacts(
            run_id=tissue_masks_run_id,
            artifact_path="tissue_masks",
        )
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split_name in ("train", "test"):
            slides = load_parquet_artifact(
                tiling_run_id, f"{split_name}_split/slides.parquet"
            )
            tiles = load_tiles_columns(
                tiling_run_id,
                f"{split_name}_split/tiles.parquet",
                columns=["slide_id", "x", "y"],
            )

            output_path = Path(tmp_dir) / f"{split_name}_tiles.parquet"
            stats = add_tissue_coverage(
                slides, tiles, tissue_mask_dir, config.tissue_mpp, output_path
            )

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
