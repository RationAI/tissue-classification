import tempfile
from pathlib import Path
from typing import cast

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
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def load_tiles_columns(run_id: str, artifact_path: str, columns: list[str]) -> pa.Table:
    """Load a tiles parquet column subset as a memory-mapped Arrow table.

    HF datasets caches the parquet into Arrow IPC on first load. We return the
    underlying pyarrow Table without materializing 80M rows into pandas — only
    small per-slide slices are converted before dispatching Ray tasks.
    """
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    dataset = load_dataset(
        "parquet", data_files=local_path, split="train", columns=columns
    )
    return dataset.data.table


def resolve_mask_dir(mask_source: DictConfig) -> Path:
    if "local_path" in mask_source:
        return Path(mask_source.local_path)
    if "mlflow_run_id" in mask_source:
        return Path(
            mlflow.artifacts.download_artifacts(
                run_id=mask_source.mlflow_run_id,
                artifact_path=mask_source.artifact_path,
            )
        )
    raise ValueError(
        "mask_source must contain either 'local_path' or "
        "'mlflow_run_id' + 'artifact_path'"
    )


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


@ray.remote(num_cpus=1, max_calls=1)
def process_slide(
    slide_tiles: pd.DataFrame,
    mask_paths: dict[str, str],
    tile_extent: int,
    tile_mpp: float,
    mask_mpp: float,
) -> pd.DataFrame:
    scale = tile_mpp / mask_mpp
    tm_extent = max(1, round(tile_extent * scale))
    roi_offset = tm_extent // 4
    roi_extent = max(1, tm_extent // 2)

    xs = np.round(slide_tiles["x"].to_numpy() * scale).astype(int)
    ys = np.round(slide_tiles["y"].to_numpy() * scale).astype(int)

    result = slide_tiles.copy()
    for name, path in mask_paths.items():
        mask = tifffile.imread(path)
        if mask.ndim > 2:
            mask = mask[..., 0]

        mask_h, mask_w = mask.shape
        sat = np.zeros((mask_h + 1, mask_w + 1), dtype=np.int64)
        np.greater(mask, 0, out=sat[1:, 1:], casting="unsafe")
        del mask
        np.cumsum(sat, axis=0, out=sat)
        np.cumsum(sat, axis=1, out=sat)

        result[f"tile_{name}_coverage"] = sat_coverage(
            sat, mask_h, mask_w, ys, xs, tm_extent
        )
        result[f"roi_{name}_coverage"] = sat_coverage(
            sat, mask_h, mask_w, ys + roi_offset, xs + roi_offset, roi_extent
        )
        del sat

    return result


def add_coverage(
    slides: pd.DataFrame,
    tiles: pa.Table,
    mask_dir: Path,
    masks: dict[str, str],
    mask_mpp: float,
    output_path: Path,
) -> dict[str, float]:
    """Dispatch per-slide coverage computation as Ray tasks.

    Writes per-slide results to `output_path` via a streaming ParquetWriter as each
    Ray task finishes. Slides missing any of the configured masks are skipped.
    Returns aggregate stats for metric logging.

    Tiling emits slides interleaved across blocks. We dictionary-encode slide_id
    and argsort the int indices to group row positions per slide — a full
    string-column sort/take on 80M rows would overflow pyarrow's int32 string offsets.
    """
    slide_info = slides.set_index("id")[["path", "tile_extent_x", "mpp_x"]]

    if len(tiles) == 0:
        raise ValueError("tiles table is empty")

    slide_id_col = tiles.column("slide_id")
    if slide_id_col.null_count > 0:
        raise ValueError(
            f"tiles.slide_id contains {slide_id_col.null_count} null values"
        )
    encoded = pc.dictionary_encode(slide_id_col)
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

        mask_paths = {
            name: str(mask_dir / template.format(stem=wsi_path.stem))
            for name, template in masks.items()
        }
        if any(not Path(p).exists() for p in mask_paths.values()):
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
                slide_tiles, mask_paths, tile_extent, tile_mpp, mask_mpp
            )
        )

    if missing_slides:
        print(
            f"Warning: {len(missing_slides)} slides had missing masks and were dropped"
        )

    if not futures:
        raise RuntimeError(
            f"No masks matched any slide. Check mask_source and mask templates. "
            f"Missing slides: {len(missing_slides)}"
        )

    cov_cols = [
        f"{prefix}_{name}_coverage" for name in masks for prefix in ("tile", "roi")
    ]

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    count = 0
    cov_sums: dict[str, float] = dict.fromkeys(cov_cols, 0.0)

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
            for col in cov_cols:
                cov_sums[col] += float(result[col].sum())
    finally:
        if writer is not None:
            writer.close()

    return {
        "tile_count": count,
        **{f"mean_{col}": cov_sums[col] / count for col in cov_cols},
    }


@with_cli_args(["+preprocessing=coverage_stats"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_run_id = config.dataset.mlflow_artifacts.tiling_run_id
    mask_dir = resolve_mask_dir(config.mask_source)
    masks = cast("dict[str, str]", OmegaConf.to_container(config.masks, resolve=True))

    with tempfile.TemporaryDirectory() as tmp_dir:
        for split_name in ("train", "test"):
            local_path = mlflow.artifacts.download_artifacts(
                run_id=tiling_run_id,
                artifact_path=f"{split_name}_split/slides.parquet",
            )
            slides = pd.read_parquet(local_path)
            tiles = load_tiles_columns(
                tiling_run_id,
                f"{split_name}_split/tiles.parquet",
                columns=["slide_id", "x", "y"],
            )

            output_path = Path(tmp_dir) / f"{split_name}_tiles.parquet"
            stats = add_coverage(
                slides, tiles, mask_dir, masks, config.mpp, output_path
            )

            mlflow.log_metric(f"{split_name}_tile_count", stats["tile_count"])
            for key, value in stats.items():
                if key != "tile_count":
                    mlflow.log_metric(f"{split_name}_{key}", value)

        mlflow.log_artifacts(tmp_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    with ray.init(num_cpus=4):
        main()
