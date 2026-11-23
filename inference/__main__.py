"""Tissue classification inference CLI.

    uv run python -m inference predict --slides /data/graz/*.svs --out runs/graz15 \
        --models virchow2

See ``inference/README.md``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from inference.heads import load_head


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def _environment() -> dict[str, Any]:
    import torch

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return env


def embed(args: argparse.Namespace) -> int:
    """Tissue masks, tiles and cached embeddings only (no classification head needed)."""
    return _run(args, embed_only=True)


def predict(args: argparse.Namespace) -> int:
    return _run(args, embed_only=False)


def _run(args: argparse.Namespace, embed_only: bool) -> int:
    from inference.predict import embed_slide, prepare_slide, write_predictions
    from inference.slides import find_slides
    from preprocessing.foundation_models import FoundationModel, get_spec

    started = time.monotonic()
    slides = find_slides(args.slides)
    for model in args.models:  # fail fast on typos
        get_spec(model)
    heads = {} if embed_only else {model: load_head(model) for model in args.models}
    args.out.mkdir(parents=True, exist_ok=True)

    run: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k != "func"
        },
        "command": "embed" if embed_only else "predict",
        "environment": _environment(),
        "heads": {m: h.meta for m, h in heads.items()},
        "slides": {},
    }
    run["args"]["slides"] = [str(p) for p in slides]
    print(f"{len(slides)} slides, models: {', '.join(args.models)}")
    print(json.dumps(run["environment"]), flush=True)

    # 1) tissue mask + tiles, once per slide
    prepared = {}
    for i, path in enumerate(slides, 1):
        slide_dir = args.out / path.stem
        slide_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(slides)}] {path.name}: tissue mask + tiles", flush=True)
        try:
            geometry, tiles = prepare_slide(
                path, slide_dir, args.mpp, args.tile_size, args.stride, args.tissue_min
            )
        except Exception as exc:
            traceback.print_exc()
            run["slides"][path.name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        prepared[path] = (slide_dir, geometry, tiles)
        run["slides"][path.name] = {
            "level": geometry.level,
            "mpp": [geometry.mpp_x, geometry.mpp_y],
            "extent": [geometry.extent_x, geometry.extent_y],
            "tissue_tiles": len(tiles),
        }
        print(
            f"  level {geometry.level} ({geometry.mpp_x:.3f} mpp), {len(tiles)} tissue tiles"
        )

    # 2) per model: load once, embed + classify every slide
    summary_rows = []
    for model_name in args.models:
        print(f"loading {model_name}", flush=True)
        model = FoundationModel(model_name, device=args.device)
        head = heads.get(model_name)
        for i, (path, (slide_dir, geometry, tiles)) in enumerate(prepared.items(), 1):
            print(f"[{i}/{len(prepared)}] {path.name}: {model_name}", flush=True)
            t0 = time.monotonic()
            try:
                embeddings = embed_slide(
                    model,
                    geometry,
                    tiles,
                    slide_dir / f"embeddings_{model_name}.npy",
                    args.batch_size,
                    args.num_workers,
                )
                if head is not None:
                    frame = write_predictions(
                        head,
                        embeddings,
                        geometry,
                        tiles,
                        slide_dir,
                        args.thumbnail_size,
                    )
            except Exception as exc:
                traceback.print_exc()
                run["slides"][path.name][f"error_{model_name}"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            run["slides"][path.name][f"seconds_{model_name}"] = round(
                time.monotonic() - t0, 1
            )
            if head is None:
                summary_rows.append(
                    {"slide": path.name, "model": model_name, "tiles": len(embeddings)}
                )
                continue
            fractions = frame["pred_name"].value_counts(normalize=True)
            summary_rows.append(
                {
                    "slide": path.name,
                    "model": model_name,
                    "tiles": len(frame),
                    **{
                        f"frac_{c}": float(fractions.get(c, 0.0))
                        for c in head.class_names
                    },
                }
            )
        del model

    command = run["command"]
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            args.out / f"summary_{command}.csv", index=False
        )
    run["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    run["seconds"] = round(time.monotonic() - started, 1)
    (args.out / f"run_{command}.json").write_text(json.dumps(run, indent=2) + "\n")

    failed = [
        s
        for s, info in run["slides"].items()
        if any(k.startswith("error") for k in info)
    ]
    print(f"done in {run['seconds']} s -> {args.out}")
    if failed:
        print(
            f"FAILED slides ({len(failed)}): {', '.join(failed)} (see run_{command}.json)"
        )
        return 1
    return 0


DEFAULT_LABEL_MAP = Path(__file__).parent / "label_maps" / "graz.yaml"


def evaluate_cmd(args: argparse.Namespace) -> int:
    """Compare cached predictions with GeoJSON ground truth."""
    from inference.evaluate import LabelMap, evaluate, find_annotation
    from inference.slides import find_slides, slide_geometry

    label_map = LabelMap.load(args.label_map)
    slides = find_slides(args.slides)
    items, annotation_files, skipped = [], [], []
    for path in slides:
        annotation = find_annotation(
            path, args.annotations, label_map.annotation_patterns
        )
        slide_dir = args.out / path.stem
        if annotation is None:
            skipped.append(f"{path.name} (no annotation)")
            continue
        missing = [
            f.name
            for f in [slide_dir / "tiles.parquet"]
            + [slide_dir / f"predictions_{m}.parquet" for m in args.models]
            if not f.exists()
        ]
        if missing:
            skipped.append(
                f"{path.name} (run predict first: missing {', '.join(missing)})"
            )
            continue
        geometry = slide_geometry(path, args.mpp, args.tile_size, args.stride)
        items.append((path, annotation, geometry, slide_dir))
        annotation_files.append(annotation)
    for s in skipped:
        print(f"[skip] {s}", flush=True)
    if not items:
        print("nothing to evaluate")
        return 1
    evaluate(
        items,
        args.models,
        label_map,
        args.label_map,
        annotation_files,
        args.out,
        args.thumbnail_size,
    )
    return 1 if skipped else 0


def labels_cmd(args: argparse.Namespace) -> int:
    """List every annotation label and how the label map handles it."""
    from inference.evaluate import LabelMap, label_inventory

    label_map = LabelMap.load(args.label_map)
    files = sorted(
        p
        for p in args.annotations.iterdir()
        if p.name.endswith((".geojson", ".geo.json"))
    )
    if not files:
        print(f"no .geojson files in {args.annotations}")
        return 1
    inventory = label_inventory(files, label_map)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(f"{len(files)} annotation files\n")
        print(inventory.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    unmapped = (inventory["maps_to"] == "<UNMAPPED>").sum()
    if unmapped:
        print(f"\n{unmapped} unmapped label(s): add them to {args.label_map}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inference", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(required=True)

    for name, func, help_text in (
        ("embed", embed, "tissue masks, tiles and embeddings only (no head needed)"),
        ("predict", predict, "embed (or reuse cached embeddings) and classify"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "--slides",
            type=Path,
            nargs="+",
            required=True,
            help="slide files and/or directories containing slides",
        )
        p.add_argument("--out", type=Path, required=True, help="output directory")
        p.add_argument(
            "--models",
            nargs="+",
            default=["virchow2"],
            help="foundation models: virchow2, h-optimus-0 (predict needs a head in inference/heads)",
        )
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument(
            "--num-workers", type=int, default=8, help="tile-reading processes"
        )
        p.add_argument("--device", default=None, help="default: cuda if available")
        p.add_argument(
            "--tissue-min",
            type=float,
            default=0.0,
            help="keep tiles with tissue coverage > this (training maps used 0.0)",
        )
        p.add_argument("--thumbnail-size", type=int, default=2048)
        p.add_argument("--mpp", type=float, default=0.5, help=argparse.SUPPRESS)
        p.add_argument("--tile-size", type=int, default=224, help=argparse.SUPPRESS)
        p.add_argument("--stride", type=int, default=112, help=argparse.SUPPRESS)
        p.set_defaults(func=func)

    p = sub.add_parser("evaluate", help="compare predictions with GeoJSON ground truth")
    p.add_argument("--slides", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True, help="same --out as predict")
    p.add_argument(
        "--annotations", type=Path, required=True, help="annotation directory"
    )
    p.add_argument("--models", nargs="+", default=["virchow2"])
    p.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    p.add_argument("--thumbnail-size", type=int, default=1536)
    p.add_argument("--mpp", type=float, default=0.5, help=argparse.SUPPRESS)
    p.add_argument("--tile-size", type=int, default=224, help=argparse.SUPPRESS)
    p.add_argument("--stride", type=int, default=112, help=argparse.SUPPRESS)
    p.set_defaults(func=evaluate_cmd)

    p = sub.add_parser("labels", help="list annotation labels and their mapping")
    p.add_argument(
        "--annotations", type=Path, required=True, help="annotation directory"
    )
    p.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    p.set_defaults(func=labels_cmd)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
