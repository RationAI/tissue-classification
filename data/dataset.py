from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from rationai.mlkit.data.datasets import MetaTiledSlides, OpenSlideTilesDataset
from torch.utils.data import ConcatDataset


class LabeledTilesDataset(OpenSlideTilesDataset):
    """Extends OpenSlideTilesDataset to return (image, class_index) pairs."""

    def __init__(self, *args, class_to_idx: dict[str, int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx: int) -> tuple[NDArray[np.uint8], int]:
        image = super().__getitem__(idx)
        label: str = self.tiles.iloc[idx]["label"]
        return image, self.class_to_idx[label]


class TissueClassificationDataset(MetaTiledSlides[tuple[NDArray[np.uint8], int]]):
    """Dataset for tissue classification backed by tiled WSIs and kfold metadata.

    Can be constructed either from MLflow artifact URIs or from pre-loaded HFDatasets.
    The latter is useful when the caller has already filtered tiles (e.g. by fold).

    Usage — loading from MLflow and splitting by fold:

        full = TissueClassificationDataset(
            slides_uri=..., tiles_uri=..., class_to_idx=config.class_indices
        )
        train_tiles, val_tiles = filter_tiles_by_fold(full.tiles, val_fold=0)
        train_ds = TissueClassificationDataset(
            slides_and_tiles=(full.slides, train_tiles), class_to_idx=config.class_indices
        )
        val_ds = TissueClassificationDataset(
            slides_and_tiles=(full.slides, val_tiles), class_to_idx=config.class_indices
        )
    """

    def __init__(
        self,
        *,
        class_to_idx: dict[str, int],
        slides_uri: str | None = None,
        tiles_uri: str | None = None,
        slides_and_tiles: tuple[HFDataset, HFDataset] | None = None,
    ) -> None:
        self.class_to_idx = class_to_idx
        if slides_and_tiles is not None:
            slides, tiles = slides_and_tiles
        elif slides_uri is not None and tiles_uri is not None:
            slides_dir = Path(download_artifacts(artifact_uri=slides_uri))
            tiles_dir = Path(download_artifacts(artifact_uri=tiles_uri))

            slides = cast(
                HFDataset,
                load_dataset(
                    "parquet",
                    data_files=str(slides_dir / "slides.parquet"),
                    split="train",
                ),
            )
            tiles = cast(
                HFDataset,
                load_dataset(
                    "parquet",
                    data_files=str(tiles_dir / "kfold_tiles.parquet"),
                    split="train",
                ),
            )
        else:
            raise ValueError(
                "Provide either slides_and_tiles or both slides_uri and tiles_uri."
            )

        self.slides = slides
        self.tiles = tiles.sort("slide_id")
        self._slide_id_to_indices = self._build_tile_index(self.tiles)
        ConcatDataset.__init__(self, self.generate_datasets())

    def generate_datasets(self) -> Iterable[LabeledTilesDataset]:
        return (
            LabeledTilesDataset(
                slide_path=slide["path"],
                level=slide["level"],
                tile_extent_x=slide["tile_extent_x"],
                tile_extent_y=slide["tile_extent_y"],
                tiles=self.filter_tiles_by_slide(slide["id"]).to_pandas(),
                class_to_idx=self.class_to_idx,
            )
            for slide in self.slides
        )


def filter_tiles_by_fold(
    tiles: HFDataset, val_fold: int
) -> tuple[HFDataset, HFDataset]:
    """Split an HFDataset of tiles into (train, val) by fold number.

    Args:
        tiles: The full tiles dataset, as stored in TissueClassificationDataset.tiles.
        val_fold: The fold index to use as the validation set.

    Returns:
        A (train_tiles, val_tiles) tuple of HFDatasets.
    """
    train = tiles.filter(
        lambda batch: [f != val_fold for f in batch["fold"]], batched=True
    )
    val = tiles.filter(
        lambda batch: [f == val_fold for f in batch["fold"]], batched=True
    )
    return train, val
