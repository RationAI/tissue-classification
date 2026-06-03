"""Export the linear-head decoder of a trained MetaArch to ONNX.

The backbone is `nn.Identity` for linear probing, so only the
`decode_head` (a single `nn.Linear(embedding_dim, num_classes)`) needs
to be exported. The exported graph operates on a batch of precomputed
embeddings produced upstream by the Virchow2 foundation model.

Usage:
    uv run python scripts/export_onnx.py \\
        --checkpoint runs:/<RUN_ID>/<artifact_path>/best.ckpt \\
        --output linear_head.onnx \\
        --embedding-dim 2560 \\
        --num-classes 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mlflow.artifacts import download_artifacts
from torch import nn

from ml.meta_arch import MetaArch


def resolve_checkpoint(uri: str) -> str:
    if uri.startswith(("mlflow-artifacts:/", "runs:/")):
        return download_artifacts(artifact_uri=uri)
    return uri


def load_decode_head(
    checkpoint_path: str,
    embedding_dim: int,
    num_classes: int,
    class_indices: dict[str, int],
) -> nn.Module:
    backbone = nn.Identity()
    decode_head = nn.Linear(embedding_dim, num_classes)
    model = MetaArch.load_from_checkpoint(
        checkpoint_path,
        backbone=backbone,
        decode_head=decode_head,
        class_indices=class_indices,
        map_location="cpu",
        strict=True,
    )
    model.eval()
    return model.decode_head


def export(
    decode_head: nn.Module,
    output_path: Path,
    embedding_dim: int,
) -> None:
    dummy = torch.randn(1, embedding_dim, dtype=torch.float32)
    torch.onnx.export(
        decode_head,
        dummy,
        str(output_path),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )


DEFAULT_CLASS_INDICES = {
    "Nerve": 0,
    "Blood": 1,
    "Connective-Tissue": 2,
    "Fat": 3,
    "Epithelium": 4,
    "Muscle": 5,
    "Other": 6,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Lightning .ckpt path or MLflow URI (runs:/... / mlflow-artifacts:/...).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("linear_head.onnx"),
        help="Destination ONNX file.",
    )
    parser.add_argument("--embedding-dim", type=int, default=2560)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument(
        "--log-to-run-id",
        default=None,
        help="If set, upload the ONNX file as an artifact to this MLflow run.",
    )
    parser.add_argument(
        "--mlflow-artifact-path",
        default="onnx",
        help="Subdirectory inside the MLflow run for the uploaded artifact.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    decode_head = load_decode_head(
        checkpoint_path=checkpoint_path,
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        class_indices=DEFAULT_CLASS_INDICES,
    )
    export(decode_head, args.output, args.embedding_dim)
    print(f"Wrote {args.output}")

    if args.log_to_run_id is not None:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        client.log_artifact(
            run_id=args.log_to_run_id,
            local_path=str(args.output),
            artifact_path=args.mlflow_artifact_path,
        )
        print(
            f"Uploaded {args.output} -> run {args.log_to_run_id} "
            f"under '{args.mlflow_artifact_path}/'"
        )


if __name__ == "__main__":
    main()
