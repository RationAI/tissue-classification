"""Write tile predictions as WSI-aligned BigTIFF masks."""

from collections.abc import Mapping
from pathlib import Path
from re import sub
from tempfile import TemporaryDirectory
from typing import Any, cast

import lightning as pl
import mlflow
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from mlflow.artifacts import download_artifacts


class TiffPredictionMapWriter(Callback):
    """Collect test/predict batches and log per-slide prediction TIFFs.

    The output masks use the same coordinate space and MPP as the tiling
    ``slides.parquet`` artifact. Class maps store the predicted class index as
    ``uint8`` and use ``background_value`` for pixels without predictions.
    """

    def __init__(
        self,
        slides_uri: str,
        artifact_path: str = "prediction_maps_tiff",
        background_value: int = 255,
        draw_region: str = "central_stride",
        write_errors: bool = True,
        max_slides: int | None = None,
        slide_selection: str = "all",
    ) -> None:
        super().__init__()
        if draw_region != "central_stride":
            raise ValueError(
                "draw_region must be 'central_stride'; 'tile' is unsupported "
                "for class maps (overlapping tiles would average categorical "
                f"class indices). got {draw_region!r}"
            )
        if slide_selection not in {"all", "worst"}:
            raise ValueError(
                "slide_selection must be either 'all' or 'worst', "
                f"got {slide_selection!r}"
            )
        if max_slides is not None and max_slides <= 0:
            raise ValueError(f"max_slides must be positive or None, got {max_slides}")
        self.slides_uri = slides_uri
        self.artifact_path = artifact_path
        self.background_value = background_value
        self.draw_region = draw_region
        self.write_errors = write_errors
        self.max_slides = max_slides
        self.slide_selection = slide_selection
        self._batches: list[dict[str, Any]] = []

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._batches.clear()
        print("[TiffPredictionMapWriter] test loop started", flush=True)

    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._batches.clear()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank == 0 and isinstance(outputs, Mapping):
            self._batches.append(_to_cpu_batch(outputs))
            if batch_idx % 50 == 0:
                print(
                    f"[TiffPredictionMapWriter] test batch {batch_idx} "
                    f"({len(self._batches)} buffered)",
                    flush=True,
                )

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank == 0 and outputs is not None:
            self._batches.append(_to_cpu_batch(outputs))

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._write_maps(trainer)

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._write_maps(trainer)

    def _write_maps(self, trainer: pl.Trainer) -> None:
        if trainer.global_rank != 0 or not self._batches:
            self._batches.clear()
            return

        predictions = _batches_to_dataframe(self._batches)
        self._batches.clear()
        if predictions.empty:
            return

        slides = pd.read_parquet(_resolve_uri(self.slides_uri))
        slides_by_id = {
            str(row["id"]): cast("dict[str, Any]", row)
            for row in slides.to_dict(orient="records")
        }

        with TemporaryDirectory(dir=Path(trainer.default_root_dir)) as output_dir:
            output_path = Path(output_dir)
            Path(output_path, "pred").mkdir(parents=True, exist_ok=True)
            if self.write_errors:
                Path(output_path, "errors").mkdir(parents=True, exist_ok=True)

            slide_groups = self._select_slide_groups(predictions)
            print(
                f"[TiffPredictionMapWriter] writing {len(slide_groups)} "
                f"prediction map(s)"
            )
            for index, (slide_id, slide_predictions) in enumerate(
                slide_groups, start=1
            ):
                slide = slides_by_id.get(str(slide_id))
                if slide is None:
                    raise KeyError(
                        f"slide_id {slide_id!r} not found in slides artifact "
                        f"{self.slides_uri!r}"
                    )
                print(
                    f"[TiffPredictionMapWriter] {index}/{len(slide_groups)} "
                    f"{Path(str(slide['path'])).name}"
                )
                self._write_slide_maps(slide, slide_predictions, output_path)

            active = mlflow.active_run()
            if active is not None:
                print(
                    f"[TiffPredictionMapWriter] logging artifacts to "
                    f"{self.artifact_path}"
                )
                mlflow.log_artifacts(output_dir, artifact_path=self.artifact_path)

    def _select_slide_groups(
        self, predictions: pd.DataFrame
    ) -> list[tuple[str, pd.DataFrame]]:
        if self.slide_selection == "worst":
            predictions = predictions.assign(
                _correct=predictions["pred"] == predictions["target"]
            )
            slide_ids = (
                predictions.groupby("slide_id", sort=False)["_correct"]
                .mean()
                .sort_values()
                .index
            )
            if self.max_slides is not None:
                slide_ids = slide_ids[: self.max_slides]
            selected = predictions[predictions["slide_id"].isin(slide_ids)]
            return [
                (str(slide_id), slide_predictions.drop(columns=["_correct"]))
                for slide_id, slide_predictions in selected.groupby(
                    "slide_id", sort=False
                )
            ]

        groups = [
            (str(slide_id), slide_predictions)
            for slide_id, slide_predictions in predictions.groupby(
                "slide_id", sort=False
            )
        ]
        return groups[: self.max_slides] if self.max_slides is not None else groups

    def _write_slide_maps(
        self,
        slide: dict[str, Any],
        predictions: pd.DataFrame,
        output_path: Path,
    ) -> None:
        filename = f"{_safe_filename(Path(str(slide['path'])).stem)}.tiff"
        extent = (int(slide["extent_x"]), int(slide["extent_y"]))
        tile_extent = (int(slide["tile_extent_x"]), int(slide["tile_extent_y"]))
        stride = (int(slide["stride_x"]), int(slide["stride_y"]))
        mpp = (float(slide["mpp_x"]), float(slide["mpp_y"]))

        xs = predictions["x"].to_numpy(dtype=np.int64)
        ys = predictions["y"].to_numpy(dtype=np.int64)
        preds = predictions["pred"].to_numpy(dtype=np.int64)

        _write_assembled_map(
            values=preds,
            xs=xs,
            ys=ys,
            extent=extent,
            tile_extent=tile_extent,
            stride=stride,
            background_value=self.background_value,
            path=Path(output_path, "pred", filename),
            mpp=mpp,
        )

        if not self.write_errors:
            return

        errors = (
            predictions["pred"].to_numpy() != predictions["target"].to_numpy()
        ).astype(np.int64)
        _write_assembled_map(
            values=errors,
            xs=xs,
            ys=ys,
            extent=extent,
            tile_extent=tile_extent,
            stride=stride,
            background_value=self.background_value,
            path=Path(output_path, "errors", filename),
            mpp=mpp,
        )


def _write_assembled_map(
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    extent: tuple[int, int],
    tile_extent: tuple[int, int],
    stride: tuple[int, int],
    background_value: int,
    path: Path,
    mpp: tuple[float, float],
) -> None:
    """Assemble per-tile scalar predictions into a WSI-aligned uint8 BigTIFF.

    Uses ``HeatmapAssembler`` with the tile footprint set to ``stride`` so
    tiles are non-overlapping (``central_stride`` semantics): the count grid
    stays <= 1, so no categorical class-index averaging occurs. The assembler
    keeps a GCD-compressed grid (extent / stride), avoiding a full-extent
    in-RAM buffer. Pixels never covered by a tile are written as
    ``background_value``; the grid is recentered by ``(tile - stride) // 2``
    on embed to match the tile's central receptive region.
    """
    import pyvips
    from rationai.masks import write_big_tiff
    from rationai.masks.heatmap_assembler import HeatmapAssembler

    path.parent.mkdir(parents=True, exist_ok=True)
    extent_x, extent_y = extent
    stride_x, stride_y = stride

    assembler = HeatmapAssembler(
        extent_x,
        extent_y,
        stride_x,
        stride_y,
        stride_x,
        stride_y,
        dtype=torch.float32,
    )
    assembler.update(
        torch.from_numpy(values.astype(np.float32)),
        torch.from_numpy(xs),
        torch.from_numpy(ys),
    )

    grid = assembler.compute().round().to(torch.uint8).numpy()
    grid[assembler._count.numpy() == 0] = background_value
    grid = np.ascontiguousarray(grid)

    mask = pyvips.Image.new_from_array(grid).cast(pyvips.BandFormat.UCHAR)
    mask = mask.resize(
        assembler.common_divisor_x,
        vscale=assembler.common_divisor_y,
        kernel=pyvips.enums.Kernel.NEAREST,
    )
    offset_x = (tile_extent[0] - stride_x) // 2
    offset_y = (tile_extent[1] - stride_y) // 2
    mask = mask.embed(
        offset_x,
        offset_y,
        extent_x,
        extent_y,
        extend=pyvips.enums.Extend.BACKGROUND,
        background=[background_value],
    )
    write_big_tiff(mask, path, mpp[0], mpp[1])


def _to_cpu_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _batches_to_dataframe(batches: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for batch in batches:
        rows.append(
            pd.DataFrame(
                {
                    "slide_id": list(batch["slide_id"]),
                    "x": batch["x"].numpy(),
                    "y": batch["y"].numpy(),
                    "target": batch["target"].numpy(),
                    "pred": batch["pred"].numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _resolve_uri(uri: str) -> str:
    if uri.startswith(("mlflow-artifacts:/", "runs:/")):
        return download_artifacts(artifact_uri=uri)
    return uri


def _safe_filename(value: str) -> str:
    return sub(r"[^A-Za-z0-9_.-]+", "_", value)
