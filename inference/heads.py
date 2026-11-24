"""Linear classification heads shipped with the repo.

Each head is ``inference/heads/<model>.safetensors`` (``weight``: (C, D), ``bias``: (C,))
plus ``<model>.json`` with the metadata needed to apply it (class names, embedding
model, tile geometry and the training run it came from). Heads are exported from
the Lightning checkpoints with ``python -m inference.export_head``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from inference.slides import TILE_MPP, TILE_SIZE, TILE_STRIDE


HEADS_DIR = Path(__file__).parent / "heads"


@dataclass(frozen=True)
class LinearHead:
    model: str
    weight: np.ndarray  # (C, D) float32
    bias: np.ndarray  # (C,) float32
    class_names: list[str]
    meta: dict[str, Any]

    @property
    def embedding_dim(self) -> int:
        return int(self.weight.shape[1])

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """Softmax class probabilities for embeddings of shape (N, D)."""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"{self.model} head expects {self.embedding_dim}-d embeddings, "
                f"got {embeddings.shape[1]}"
            )
        logits = embeddings.astype(np.float32) @ self.weight.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def load_head(model: str, heads_dir: Path = HEADS_DIR) -> LinearHead:
    weights_path = heads_dir / f"{model}.safetensors"
    meta_path = heads_dir / f"{model}.json"
    if not weights_path.exists() or not meta_path.exists():
        available = sorted(p.stem for p in heads_dir.glob("*.safetensors"))
        raise FileNotFoundError(
            f"No classification head for {model!r} in {heads_dir} "
            f"(available: {available or 'none'})"
        )
    tensors = load_file(str(weights_path))
    meta = json.loads(meta_path.read_text())
    tiling = {"tile_size": TILE_SIZE, "stride": TILE_STRIDE, "mpp": TILE_MPP}
    head_tiling = {key: meta.get(key) for key in tiling}
    if head_tiling != tiling:
        raise ValueError(
            f"{meta_path.name} was trained on tiling {head_tiling}, "
            f"but inference tiles with {tiling}"
        )
    return LinearHead(
        model=model,
        weight=tensors["weight"].astype(np.float32),
        bias=tensors["bias"].astype(np.float32),
        class_names=list(meta["class_names"]),
        meta=meta,
    )
