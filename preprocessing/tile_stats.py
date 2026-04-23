import tempfile
from pathlib import Path

import hydra
import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
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


def load_tiles_columns(run_id: str, artifact_path: str, columns: list[str]) -> pd.DataFrame:
    """Lazy-load only the specified columns from a tiles parquet artifact.

    The tiles parquet has one column per class for both tile and ROI coverage; reading the
    full table for slides with millions of tiles would consume many GB of RAM. Projecting
    to the columns we actually use keeps memory bounded and makes Ray serialization cheap.
    """
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    dataset = load_dataset("parquet", data_files=local_path, split="train")
    df = dataset.select_columns(columns).to_pandas()
    if "slide_id" in df.columns:
        df["slide_id"] = df["slide_id"].astype("category")
    return df


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
    mask_h, mask_w = mask.shape

    sat = np.zeros((mask_h + 1, mask_w + 1), dtype=np.int32)
    sat[1:, 1:] = np.cumsum(np.cumsum(mask > 0, axis=0, dtype=np.int32), axis=1)

    scale = tile_mpp / tissue_mpp
    tm_extent = max(1, round(tile_extent * scale))
    roi_offset = tm_extent // 4
    roi_extent = max(1, tm_extent // 2)

    xs = np.round(tiles["x"].values * scale).astype(int)
    ys = np.round(tiles["y"].values * scale).astype(int)

    tile_cov = sat_coverage(sat, mask_h, mask_w, ys, xs, tm_extent)
    roi_cov = sat_coverage(sat, mask_h, mask_w, ys + roi_offset, xs + roi_offset, roi_extent)

    tiles = tiles.copy()
    tiles["tile_tissue_coverage"] = tile_cov
    tiles["roi_tissue_coverage"] = roi_cov
    return tiles


@ray.remote(num_cpus=1, memory=1 * 1024**3)
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
    return compute_tissue_coverages(slide_tiles, mask, tile_mpp, tile_extent, tissue_mpp)


def add_tissue_coverage(
    slides: pd.DataFrame,
    tiles: pd.DataFrame,
    tissue_mask_dir: Path,
    tissue_mpp: float,
) -> pd.DataFrame:
    """Dispatch per-slide tissue coverage computation as Ray tasks and collect results."""
    slide_info = slides.set_index("id")[["path", "tile_extent_x", "mpp_x"]]

    futures = []
    missing_slides = []

    for slide_id, slide_tiles in tiles.groupby("slide_id"):
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

        futures.append(
            process_slide.remote(slide_tiles, str(mask_path), tile_extent, tile_mpp, tissue_mpp)
        )

    if missing_slides:
        print(f"Warning: {len(missing_slides)} slides had no matching tissue mask and were dropped")

    if not futures:
        raise RuntimeError(
            f"No tissue masks matched any slide. Check tissue_masks_run_id and artifact path. "
            f"Missing slides: {len(missing_slides)}"
        )

    return pd.concat(ray.get(futures), ignore_index=True)


@with_cli_args(["+preprocessing=tile_stats"])
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

            tiles = add_tissue_coverage(slides, tiles, tissue_mask_dir, config.tissue_mpp)

            tiles.to_parquet(Path(tmp_dir) / f"{split_name}_tiles.parquet", index=False)

            mlflow.log_metric(f"{split_name}_tile_count", len(tiles))
            mlflow.log_metric(
                f"{split_name}_mean_tile_tissue_coverage",
                float(tiles["tile_tissue_coverage"].mean()),
            )
            mlflow.log_metric(
                f"{split_name}_mean_roi_tissue_coverage",
                float(tiles["roi_tissue_coverage"].mean()),
            )

        mlflow.log_artifacts(tmp_dir, config.mlflow_artifact_path)


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):
        main()
