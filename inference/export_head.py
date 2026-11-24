"""Export a trained linear head from its Lightning checkpoint to ``inference/heads``.

Runs where MLflow is reachable (MUNI cluster / JupyterHub):

    MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000 \
        uv run python -m inference.export_head \
        --model virchow2 --run-id a0ec1995d1b1472785d30ccbe15e84ef

The resulting ``<model>.safetensors`` + ``<model>.json`` are small (tens of KB) and
are committed to the repo, so inference needs no MLflow access.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from inference.heads import HEADS_DIR
from inference.slides import TILE_MPP, TILE_SIZE, TILE_STRIDE
from preprocessing.foundation_models import get_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, help="foundation model key, e.g. virchow2"
    )
    parser.add_argument(
        "--run-id", required=True, help="MLflow run id of the final training run"
    )
    parser.add_argument("--experiment-id", default="104")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="local checkpoint path or URI; defaults to the run's last checkpoint",
    )
    parser.add_argument("--out-dir", type=Path, default=HEADS_DIR)
    args = parser.parse_args()

    spec = get_spec(args.model)
    uri = args.checkpoint or (
        f"mlflow-artifacts:/{args.experiment_id}/{args.run_id}"
        "/artifacts/checkpoints/last/checkpoint.ckpt"
    )
    if uri.startswith(("mlflow-artifacts:/", "runs:/")):
        from mlflow.artifacts import download_artifacts

        path = download_artifacts(artifact_uri=uri)
    else:
        path = uri

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    weight = state["decode_head.weight"].float().numpy()
    bias = state["decode_head.bias"].float().numpy()

    class_indices: dict[str, int] = dict(ckpt["hyper_parameters"]["class_indices"])
    class_names = [n for n, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]
    if weight.shape != (len(class_names), spec.embedding_dim):
        raise ValueError(
            f"head weight {weight.shape} does not match "
            f"({len(class_names)} classes, {spec.embedding_dim}-d {args.model})"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"weight": np.ascontiguousarray(weight), "bias": np.ascontiguousarray(bias)},
        str(args.out_dir / f"{args.model}.safetensors"),
    )
    meta = {
        "model": args.model,
        "hf_repo": spec.hf_repo,
        "embedding_dim": spec.embedding_dim,
        "class_names": class_names,
        "tile_size": TILE_SIZE,
        "stride": TILE_STRIDE,
        "mpp": TILE_MPP,
        "train_run_id": args.run_id,
        "checkpoint": uri,
        "optimizer": ckpt["hyper_parameters"].get("optimizer"),
        "weight_decay": ckpt["hyper_parameters"].get("weight_decay"),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (args.out_dir / f"{args.model}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out_dir / args.model}.safetensors/.json: classes={class_names}")


if __name__ == "__main__":
    main()
