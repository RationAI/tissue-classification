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

from ml.prediction_maps import ClassVoteAssembler, emit_mask


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
        background_value: int = 0,
        draw_region: str = "central_stride",
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
        self.max_slides = max_slides
        self.slide_selection = slide_selection
        self._batches: list[dict[str, Any]] = []
        self._written = False

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._batches.clear()
        self._written = False
        print("[TiffPredictionMapWriter] test loop started", flush=True)

    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._batches.clear()
        self._written = False

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if isinstance(outputs, Mapping):
            self._batches.append(_to_cpu_batch(outputs))
            if trainer.global_rank == 0 and batch_idx % 50 == 0:
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
        if outputs is not None:
            self._batches.append(_to_cpu_batch(outputs))

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._write_maps(trainer)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._write_maps(trainer)

    def on_predict_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._write_maps(trainer)

    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._write_maps(trainer)

    def _write_maps(self, trainer: pl.Trainer) -> None:
        if self._written:
            return
        batches = _gather_batches(self._batches)
        if trainer.global_rank != 0:
            self._batches.clear()
            return
        if not batches:
            print(
                "[TiffPredictionMapWriter] no buffered batches at write time "
                f"(local buffer={len(self._batches)}, "
                f"dist_init={torch.distributed.is_available() and torch.distributed.is_initialized()}); "
                "skipping",
                flush=True,
            )
            return

        predictions = _batches_to_dataframe(batches)
        if predictions.empty:
            print(
                "[TiffPredictionMapWriter] predictions dataframe empty; skipping",
                flush=True,
            )
            return

        slides = pd.read_parquet(_resolve_uri(self.slides_uri))
        slides_by_id = {
            str(row["id"]): cast("dict[str, Any]", row)
            for row in slides.to_dict(orient="records")
        }

        with TemporaryDirectory(dir=Path(trainer.default_root_dir)) as output_dir:
            output_path = Path(output_dir)
            Path(output_path, "pred").mkdir(parents=True, exist_ok=True)
            Path(output_path, "prob").mkdir(parents=True, exist_ok=True)

            slide_groups = self._select_slide_groups(predictions)
            print(
                f"[TiffPredictionMapWriter] writing {len(slide_groups)} "
                f"prediction map(s)",
                flush=True,
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
                    f"{Path(str(slide['path'])).name}",
                    flush=True,
                )
                self._write_slide_maps(
                    slide,
                    slide_predictions,
                    output_path,
                    class_names=getattr(trainer.lightning_module, "class_names", None),
                )

            active = mlflow.active_run()
            if active is not None:
                print(
                    f"[TiffPredictionMapWriter] logging artifacts to "
                    f"{self.artifact_path}",
                    flush=True,
                )
                mlflow.log_artifacts(output_dir, artifact_path=self.artifact_path)

        self._written = True
        self._batches.clear()

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
        class_names: list[str] | None,
    ) -> None:
        filename = _slide_prediction_filename(slide["path"])
        extent = (int(slide["extent_x"]), int(slide["extent_y"]))
        tile_extent = (int(slide["tile_extent_x"]), int(slide["tile_extent_y"]))
        stride = (int(slide["stride_x"]), int(slide["stride_y"]))
        mpp = (float(slide["mpp_x"]), float(slide["mpp_y"]))

        xs = predictions["x"].to_numpy(dtype=np.int64)
        ys = predictions["y"].to_numpy(dtype=np.int64)
        probs = np.stack(
            [np.asarray(prob, dtype=np.float32) for prob in predictions["probs"]]
        )

        pred_path = Path(output_path, "pred", filename)
        print(
            f"[TiffPredictionMapWriter] writing pred/{filename}",
            flush=True,
        )
        _write_class_map(
            probs=probs,
            xs=xs,
            ys=ys,
            extent=extent,
            tile_extent=tile_extent,
            stride=stride,
            path=pred_path,
            mpp=mpp,
            background_value=self.background_value,
        )
        print(
            f"[TiffPredictionMapWriter] wrote pred/{filename}",
            flush=True,
        )

        names = (
            class_names
            if class_names is not None and len(class_names) == probs.shape[1]
            else [f"class_{i}" for i in range(probs.shape[1])]
        )
        _write_per_class_probability_maps(
            probs=probs,
            class_names=names,
            xs=xs,
            ys=ys,
            extent=extent,
            tile_extent=tile_extent,
            stride=stride,
            output_dir=Path(output_path, "prob"),
            filename=filename,
            mpp=mpp,
        )


def _write_class_map(
    probs: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    extent: tuple[int, int],
    tile_extent: tuple[int, int],
    stride: tuple[int, int],
    path: Path,
    mpp: tuple[float, float],
    background_value: int = 0,
) -> None:
    assembler = ClassVoteAssembler(
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
    assembler.write_tiff(extent, path, mpp, background_value)


def _write_per_class_probability_maps(
    probs: np.ndarray,
    class_names: list[str],
    xs: np.ndarray,
    ys: np.ndarray,
    extent: tuple[int, int],
    tile_extent: tuple[int, int],
    stride: tuple[int, int],
    output_dir: Path,
    filename: str,
    mpp: tuple[float, float],
) -> None:
    """Write one grayscale probability map per class.

    Each class channel is assembled independently with HeatmapAssembler, so
    overlapping tiles are averaged exactly like scalar heatmaps. Values are
    encoded as uint8 probabilities in [0, 255], with never-covered pixels set
    to 0.
    """
    from rationai.masks.heatmap_assembler import HeatmapAssembler

    xs_t = torch.from_numpy(xs)
    ys_t = torch.from_numpy(ys)
    for class_idx, class_name in enumerate(class_names):
        class_dir = _safe_filename(class_name)
        print(
            f"[TiffPredictionMapWriter] writing prob/{class_dir}/{filename}",
            flush=True,
        )
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
            torch.from_numpy(probs[:, class_idx].astype(np.float32, copy=False)),
            xs_t,
            ys_t,
        )
        grid = np.clip(assembler.compute().numpy() * 255.0, 0, 255).astype(np.uint8)
        grid[assembler._count.numpy() == 0] = 0
        emit_mask(
            np.ascontiguousarray(grid),
            assembler.common_divisor_x,
            assembler.common_divisor_y,
            extent,
            0,
            output_dir / class_dir / filename,
            mpp,
        )
        print(
            f"[TiffPredictionMapWriter] wrote prob/{class_dir}/{filename}",
            flush=True,
        )


def _gather_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return batches

    gathered: list[list[dict[str, Any]]] = [
        [] for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(gathered, batches)
    return [batch for rank_batches in gathered for batch in rank_batches]


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
    return sub(r"[^A-Za-z0-9 _.-]+", "_", value)


def _slide_prediction_filename(path: str | Path) -> str:
    return Path(path).with_suffix(".tiff").name
