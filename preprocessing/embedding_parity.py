"""Compare local-backend embeddings against model-service embeddings.

Joins two embedding runs on (slide_id, x, y) and reports per-tile cosine
similarity. Used to verify that ``backend: local`` reproduces the model service
(same preprocessing, pooling, precision), which the linear heads depend on.

Usage:
    uv run python -m preprocessing.embedding_parity \
        --local-run-id <parity run> --server-run-id <embedding_run_ids.virchow2>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
from mlflow.artifacts import download_artifacts


KEYS = ["slide_id", "x", "y"]


def _load(path: Path, slide_ids: pa.Array | None = None) -> tuple[pa.Table, np.ndarray]:
    dataset = pads.dataset(str(path), format="parquet")
    row_filter = (
        pads.field("slide_id").isin(slide_ids) if slide_ids is not None else None
    )
    table = dataset.to_table(columns=[*KEYS, "embedding"], filter=row_filter)
    emb = table.column("embedding").combine_chunks()
    dim = len(emb.values) // len(emb)
    matrix = emb.values.to_numpy(zero_copy_only=False).reshape(len(emb), dim)
    keys = table.drop(["embedding"]).append_column(
        "_idx", pa.array(np.arange(table.num_rows), type=pa.int64())
    )
    return keys, matrix.astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local-run-id", required=True)
    parser.add_argument("--server-run-id", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.999,
        help="fail if the 1st percentile of cosine similarity is below this",
    )
    args = parser.parse_args()

    local_dir = Path(
        download_artifacts(
            run_id=args.local_run_id, artifact_path=f"{args.split}/tiles"
        )
    )
    local_keys, local_emb = _load(local_dir)
    slide_ids = pc.unique(local_keys.column("slide_id"))
    print(
        f"local: {len(local_emb)} tiles, {len(slide_ids)} slides, dim={local_emb.shape[1]}"
    )

    server_dir = Path(
        download_artifacts(
            run_id=args.server_run_id, artifact_path=f"{args.split}/tiles"
        )
    )
    server_keys, server_emb = _load(server_dir, slide_ids)
    print(
        f"server: {len(server_emb)} tiles on the same slides, dim={server_emb.shape[1]}"
    )

    if local_emb.shape[1] != server_emb.shape[1]:
        print("FAIL: embedding dimensions differ")
        return 1

    joined = local_keys.join(
        server_keys.rename_columns([*KEYS, "_idx_server"]), keys=KEYS, join_type="inner"
    )
    if joined.num_rows == 0:
        print(
            "FAIL: no tiles in common; were both runs made from the same tile source?"
        )
        return 1
    if joined.num_rows < len(local_emb):
        print(f"warning: only {joined.num_rows}/{len(local_emb)} local tiles matched")

    a = local_emb[joined.column("_idx").to_numpy()]
    b = server_emb[joined.column("_idx_server").to_numpy()]
    cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    rel_l2 = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)

    p = np.percentile
    print(f"matched tiles: {joined.num_rows}")
    print(
        f"cosine: mean={cosine.mean():.6f} min={cosine.min():.6f} "
        f"p01={p(cosine, 1):.6f} p50={p(cosine, 50):.6f}"
    )
    print(
        f"relative L2: mean={rel_l2.mean():.6f} p99={p(rel_l2, 99):.6f} max={rel_l2.max():.6f}"
    )

    if p(cosine, 1) < args.min_cosine:
        print(f"FAIL: cosine p01 below {args.min_cosine}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
