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
        probs = np.stack(
            [np.asarray(prob, dtype=np.float32) for prob in predictions["probs"]]
        )

        _write_class_map(
            probs=probs,
            xs=xs,
            ys=ys,
            extent=extent,
            tile_extent=tile_extent,
            stride=stride,
            path=Path(output_path, "pred", filename),
            mpp=mpp,
        )

        if not self.write_errors:
            return

        errors = (
            predictions["pred"].to_numpy() != predictions["target"].to_numpy()
        ).astype(np.float32)
        _write_error_map(
            errors=errors,
            xs=xs,
            ys=ys,
            extent=extent,
            tile_extent=tile_extent,
            stride=stride,
            path=Path(output_path, "errors", filename),
            mpp=mpp,
        )


class _ClassVoteAssembler:
    """Confidence-weighted per-class accumulator over the full tile footprint.

    Mirrors ``HeatmapAssembler``'s GCD-compressed grid + ``(x, y)`` -> ROI
    mapping, but accumulates the per-class softmax of every tile that covers a
    cell instead of summing a categorical index. Overlapping strided tiles are
    fused by ``argmax`` over the summed probabilities, so heavy stride overlap
    (e.g. tile 224 / stride 112) is used instead of discarded. ``count == 0``
    marks never-covered cells as background.
    """

    def __init__(
        self,
        extent_x: int,
        extent_y: int,
        tile_x: int,
        tile_y: int,
        stride_x: int,
        stride_y: int,
        n_classes: int,
    ) -> None:
        from math import gcd

        self.cdx = gcd(stride_x, tile_x)
        self.cdy = gcd(stride_y, tile_y)
        self._sx = stride_x // self.cdx
        self._sy = stride_y // self.cdy
        self._tx = tile_x // self.cdx
        self._ty = tile_y // self.cdy
        self._gx = extent_x // self.cdx
        self._gy = extent_y // self.cdy
        self._n_classes = n_classes
        self._acc = torch.zeros(n_classes, self._gy, self._gx, dtype=torch.float32)
        self._count = torch.zeros(self._gy, self._gx, dtype=torch.int32)

    def update(self, probs: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> None:
        for prob, x, y in zip(probs, xs, ys, strict=False):
            cx = int(x.item()) // self.cdx
            cy = int(y.item()) // self.cdy
            x0, y0 = cx * self._sx, cy * self._sy
            x1 = min(x0 + self._tx, self._gx)
            y1 = min(y0 + self._ty, self._gy)
            if x1 <= x0 or y1 <= y0:
                continue
            self._acc[:, y0:y1, x0:x1] += prob[:, None, None]
            self._count[y0:y1, x0:x1] += 1

    def labels(self) -> np.ndarray:
        """Encode prediction labels like ``remap_annotation_masks``.

        Class ``i`` (0-based) maps to
        ``round(255 * (i + 1) / n_classes)``. Never-covered pixels map to ``0``.

        The reporting tool expects GT and prediction masks in the same
        evenly-spread value space, so this must mirror that LUT exactly.
        """
        idx = self._acc.argmax(0).numpy()
        out = _spread_lut(self._n_classes)[idx].astype(np.uint8)
        out[self._count.numpy() == 0] = 0
        return np.ascontiguousarray(out)


def _emit_mask(
    grid: np.ndarray,
    cdx: int,
    cdy: int,
    extent: tuple[int, int],
    background_value: int,
    path: Path,
    mpp: tuple[float, float],
) -> None:
    """Upscale the GCD-compressed grid back to WSI extent and write a BigTIFF.

    NEAREST resize by the GCD factor (tile footprint placed at its true
    position, no recenter), background-padded to the full extent.
    """
    import pyvips
    from rationai.masks import write_big_tiff

    path.parent.mkdir(parents=True, exist_ok=True)
    mask = pyvips.Image.new_from_array(grid).cast(pyvips.BandFormat.UCHAR)
    mask = mask.resize(cdx, vscale=cdy, kernel=pyvips.enums.Kernel.NEAREST)
    mask = mask.embed(
        0,
        0,
        extent[0],
        extent[1],
        extend=pyvips.enums.Extend.BACKGROUND,
        background=[background_value],
    )
    write_big_tiff(mask, path, mpp[0], mpp[1])


def _write_class_map(
    probs: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    extent: tuple[int, int],
    tile_extent: tuple[int, int],
    stride: tuple[int, int],
    path: Path,
    mpp: tuple[float, float],
) -> None:
    assembler = _ClassVoteAssembler(
        extent[0],
        extent[1],
        tile_extent[0],
        tile_extent[1],
        stride[0],
        stride[1],
        n_classes=probs.shape[1],
    )
    assembler.update(
        torch.from_numpy(probs),
        torch.from_numpy(xs),
        torch.from_numpy(ys),
    )
    _emit_mask(
        assembler.labels(),
        assembler.cdx,
        assembler.cdy,
        extent,
        0,
        path,
        mpp,
    )


def _write_error_map(
    errors: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    extent: tuple[int, int],
    tile_extent: tuple[int, int],
    stride: tuple[int, int],
    path: Path,
    mpp: tuple[float, float],
) -> None:
    """Per-tile error (pred != target) averaged over the full tile footprint.

    Averaging a 0/1 error fraction across overlapping tiles is meaningful
    (fraction of covering tiles that were wrong); thresholded back to a 2-class
    map encoded in the same spread space as the GT (correct -> low value,
    wrong -> 255), background -> 0.
    """
    from rationai.masks.heatmap_assembler import HeatmapAssembler

    assembler = HeatmapAssembler(
        extent[0],
        extent[1],
        tile_extent[0],
        tile_extent[1],
        stride[0],
        stride[1],
        dtype=torch.float32,
    )
    assembler.update(
        torch.from_numpy(errors),
        torch.from_numpy(xs),
        torch.from_numpy(ys),
    )
    wrong = (assembler.compute() >= 0.5).numpy().astype(np.intp)
    grid = _spread_lut(2)[wrong]  # correct -> 128, wrong -> 255
    grid[assembler._count.numpy() == 0] = 0
    _emit_mask(
        np.ascontiguousarray(grid),
        assembler.common_divisor_x,
        assembler.common_divisor_y,
        extent,
        0,
        path,
        mpp,
    )


def _to_cpu_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _batches_to_dataframe(batches: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for batch in batches:
        frame = pd.DataFrame(
            {
                "slide_id": list(batch["slide_id"]),
                "x": batch["x"].numpy(),
                "y": batch["y"].numpy(),
                "target": batch["target"].numpy(),
                "pred": batch["pred"].numpy(),
            }
        )
        # per-row softmax vector kept as an object column so it survives the
        # per-slide groupby; stacked back to (n, C) in the assembler.
        frame["probs"] = list(batch["probs"].numpy())
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _resolve_uri(uri: str) -> str:
    if uri.startswith(("mlflow-artifacts:/", "runs:/")):
        return download_artifacts(artifact_uri=uri)
    return uri


def _safe_filename(value: str) -> str:
    return sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _spread_lut(n_classes: int) -> np.ndarray:
    """0-based class index -> evenly-spread uint8 value.

    Mirrors ``preprocessing/remap_annotation_masks.py`` exactly so prediction
    masks share the GT value space the reporting tool expects:
    class ``i`` -> ``round(255 * (i + 1) / n_classes)``. Index 0 of the
    returned array is unused by callers (background is set separately to 0).
    """
    return np.array(
        [round(255 * (i + 1) / n_classes) for i in range(n_classes)],
        dtype=np.uint8,
    )
