import tempfile
from pathlib import Path
from typing import cast

import hydra
import mlflow
import mlflow.artifacts
import numpy as np
import openslide
import pandas as pd
import timm
import torch
import torchvision.transforms.v2 as T
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from timm.layers.mlp import SwiGLUPacked
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Virchow2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Requires HF_TOKEN env variable; model is gated on HuggingFace.
        self.module = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        ).eval()
        self.embed_dim: int = cast("int", self.module.embed_dim) * 2

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = cast("torch.Tensor", self.module(x))  # B x 261 x 1280

        class_token = output[:, 0]  # B x 1280
        patch_tokens = output[:, 5:]  # B x 256 x 1280; tokens 1-4 are register tokens

        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # B x 2560


class TileDataset(Dataset):
    """PyTorch Dataset that reads tiles from a single WSI on-the-fly via OpenSlide."""

    def __init__(
        self,
        slide_path: str,
        tiles: np.ndarray,
        level: int,
        tile_size: int,
        transform: T.Compose,
    ) -> None:
        self.slide_path = slide_path
        self.tiles = tiles  # shape (N, 2): columns are [x, y]
        self.level = level
        self.tile_size = tile_size
        self.transform = transform

        # Opened lazily so each DataLoader worker gets its own file handle.
        self._slide: openslide.OpenSlide | None = None
        self._downsample: float | None = None

    def _open_slide(self) -> tuple[openslide.OpenSlide, float]:
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)
            self._downsample = self._slide.level_downsamples[self.level]
        return self._slide, cast("float", self._downsample)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int]]:
        x, y = int(self.tiles[idx, 0]), int(self.tiles[idx, 1])
        slide, downsample = self._open_slide()

        # OpenSlide read_region expects level-0 coordinates.
        location = (round(x * downsample), round(y * downsample))
        region = slide.read_region(location, self.level, (self.tile_size, self.tile_size))
        image = np.array(region.convert("RGB"))

        return self.transform(image), {"x": x, "y": y}


def compute_slide_embeddings(
    slide_path: str,
    slide_tiles: pd.DataFrame,
    level: int,
    model: Virchow2,
    transform: T.Compose,
    tile_size: int,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    device: torch.device,
) -> pd.DataFrame:
    dataset = TileDataset(
        slide_path=slide_path,
        tiles=slide_tiles[["x", "y"]].values,
        level=level,
        tile_size=tile_size,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    all_embeddings = torch.zeros((len(dataset), model.embed_dim), dtype=torch.float32)
    all_x = torch.zeros(len(dataset), dtype=torch.int32)
    all_y = torch.zeros(len(dataset), dtype=torch.int32)

    for i, (images, metadata) in enumerate(dataloader):
        images = images.to(device)
        batch_embeddings = model(images)

        start = i * batch_size
        end = start + batch_embeddings.size(0)
        all_embeddings[start:end] = batch_embeddings.cpu()
        all_x[start:end] = metadata["x"].to(torch.int32)
        all_y[start:end] = metadata["y"].to(torch.int32)

    return pd.DataFrame(
        {
            "x": all_x.numpy(),
            "y": all_y.numpy(),
            "embedding": [emb.numpy() for emb in all_embeddings],
        }
    )


@with_cli_args(["+preprocessing=embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Virchow2().to(device)

    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for split_name in ["train", "test"]:
        slides_df = pd.read_parquet(
            mlflow.artifacts.download_artifacts(
                run_id=config.tiling_run_id,
                artifact_path=f"{split_name}_split/slides.parquet",
            )
        )
        tiles_df = pd.read_parquet(
            mlflow.artifacts.download_artifacts(
                run_id=config.tiling_run_id,
                artifact_path=f"{split_name}_split/tiles.parquet",
            )
        )

        tiles_with_path = tiles_df.merge(
            slides_df[["id", "path", "level"]],
            left_on="slide_id",
            right_on="id",
            how="left",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for slide_id, slide_tiles in tqdm(
                tiles_with_path.groupby("slide_id"),
                desc=f"Slides ({split_name})",
            ):
                slide_row = slide_tiles.iloc[0]

                try:
                    df = compute_slide_embeddings(
                        slide_path=str(slide_row["path"]),
                        slide_tiles=slide_tiles,
                        level=int(slide_row["level"]),
                        model=model,
                        transform=transform,
                        tile_size=config.tile_size,
                        batch_size=config.dataloader.batch_size,
                        num_workers=config.dataloader.num_workers,
                        persistent_workers=config.dataloader.persistent_workers,
                        device=device,
                    )
                    df.to_parquet(tmp_path / f"{slide_id}.parquet", index=False)
                except Exception as e:
                    print(f"Error processing slide {slide_id}: {e}")

            mlflow.log_artifacts(str(tmp_path), f"{split_name}_split/embeddings")


if __name__ == "__main__":
    main()
