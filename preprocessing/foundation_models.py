"""Local (in-process) foundation-model embedding.

Registry of pathology foundation models that can be run directly on a local GPU,
without the RationAI model service. Kept free of Ray / MLflow imports so the same
code can be reused by the standalone inference CLI.

Registry keys intentionally match the model-service route names, so
``model: virchow2`` means the same model under both embedding backends.

Preprocessing and pooling must reproduce what the model service does, otherwise
embeddings computed locally are not interchangeable with the ones the linear
heads were trained on. Verified for Virchow2 on 4917 test tiles against the
model-service run: cosine similarity mean 0.999999, min 0.999989.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


TILE_SIZE = 224


@dataclass(frozen=True)
class FoundationModelSpec:
    name: str
    hf_repo: str
    embedding_dim: int
    build: Callable[[], nn.Module]
    # Returns (mean, std) in [0, 1] RGB space; receives the built model so it can
    # read ``pretrained_cfg`` when the values come from the hub config.
    normalization: Callable[[nn.Module], tuple[tuple[float, ...], tuple[float, ...]]]
    pool: Callable[[torch.Tensor], torch.Tensor]
    tile_size: int = TILE_SIZE


# --- Virchow2 -----------------------------------------------------------------
# Mirrors model-service/models/virchow2.py: timm hub model with SwiGLU MLP, output
# is the token sequence (N, 261, 1280) = 1 CLS + 4 register + 256 patch tokens.
# Embedding = concat(CLS, mean(patch tokens)) -> 2560.


def _build_virchow2() -> nn.Module:
    import timm
    from timm.layers.mlp import SwiGLUPacked

    return timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        num_classes=0,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )


def _timm_normalization(
    model: nn.Module,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read mean/std from the hub config, the same way the model service does.

    The service applies ``create_transform(**resolve_data_config(...))`` to a
    224x224 PIL tile. That transform is Resize -> CenterCrop -> ToTensor ->
    Normalize; on a 224 tile Resize/CenterCrop are identities only when the
    configured input size is 224 and crop_pct is 1.0, which is asserted here so
    the GPU-side normalization below stays equivalent.
    """
    from timm.data.config import resolve_data_config

    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    input_size = tuple(cfg["input_size"])
    crop_pct = float(cfg.get("crop_pct", 1.0))
    if input_size[-2:] != (TILE_SIZE, TILE_SIZE) or crop_pct != 1.0:
        raise ValueError(
            f"Unexpected timm data config {cfg}: local GPU preprocessing assumes "
            f"input_size={TILE_SIZE} and crop_pct=1.0 (identity resize/crop)."
        )
    return tuple(cfg["mean"]), tuple(cfg["std"])


def _pool_virchow2(out: torch.Tensor) -> torch.Tensor:
    if out.ndim != 3:
        raise ValueError(f"Virchow2 expected token output (N, T, D), got {out.shape}")
    cls_token = out[:, 0]
    patch_tokens = out[:, 5:]
    return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


# --- H-Optimus-0 ----------------------------------------------------------------
# https://huggingface.co/bioptimus/H-optimus-0 : ViT-g/14, 224 px at 0.5 mpp,
# ToTensor + Normalize with the constants below, model(x) -> (N, 1536).


def _build_hoptimus0() -> nn.Module:
    import timm

    return timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=False,
    )


def _hoptimus0_normalization(
    _model: nn.Module,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)


def _pool_identity(out: torch.Tensor) -> torch.Tensor:
    if out.ndim != 2:
        raise ValueError(f"Expected pooled output (N, D), got {out.shape}")
    return out


REGISTRY: dict[str, FoundationModelSpec] = {
    "virchow2": FoundationModelSpec(
        name="virchow2",
        hf_repo="paige-ai/Virchow2",
        embedding_dim=2560,
        build=_build_virchow2,
        normalization=_timm_normalization,
        pool=_pool_virchow2,
    ),
    "h-optimus-0": FoundationModelSpec(
        name="h-optimus-0",
        hf_repo="bioptimus/H-optimus-0",
        embedding_dim=1536,
        build=_build_hoptimus0,
        normalization=_hoptimus0_normalization,
        pool=_pool_identity,
    ),
}


def get_spec(name: str) -> FoundationModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown foundation model {name!r}; available: {sorted(REGISTRY)}"
        ) from None


class FoundationModel:
    """A foundation model loaded once and applied to batches of uint8 RGB tiles."""

    def __init__(self, name: str, device: str | None = None) -> None:
        self.spec = get_spec(name)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # fp16 autocast on GPU, as in the model service and the model cards;
        # CPU autocast only supports bfloat16.
        self.autocast_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.bfloat16
        )

        model = self.spec.build()
        mean, std = self.spec.normalization(model)
        self.model = model.to(self.device).eval()
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

        dummy = np.zeros((1, self.spec.tile_size, self.spec.tile_size, 3), np.uint8)
        dim = self.embed(dummy).shape[1]
        if dim != self.spec.embedding_dim:
            raise ValueError(
                f"{self.spec.name}: expected embedding dim "
                f"{self.spec.embedding_dim}, model produced {dim}"
            )

    def embed(self, tiles: np.ndarray) -> np.ndarray:
        """Embed tiles of shape (N, H, W, 3) uint8 -> float32 (N, D)."""
        size = self.spec.tile_size
        if tiles.ndim != 4 or tiles.shape[1:] != (size, size, 3):
            raise ValueError(
                f"Expected tiles of shape (N, {size}, {size}, 3), got {tiles.shape}"
            )
        if tiles.dtype != np.uint8:
            raise ValueError(f"Expected uint8 tiles, got {tiles.dtype}")

        x = torch.from_numpy(np.ascontiguousarray(tiles)).to(
            self.device, non_blocking=True
        )
        # Equivalent to torchvision ToTensor + Normalize, done on the GPU.
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype),
        ):
            out = self.spec.pool(self.model(x))
        return out.float().cpu().numpy()
