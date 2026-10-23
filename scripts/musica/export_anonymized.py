"""Re-render MUSICA predictions in the geometry of the anonymized slide copies.

The anonymized slides (``SLD-*.svs``) were exported from a lower pyramid level
of the originals (``level`` column of the anonymization report, typically
level 1 at ~1.0 MPP) and renamed. This script takes the tile predictions saved
by ``predict_slides.py`` (level-0 coordinates), rescales them by the exact
pyramid downsample of the exported level and writes ``pred/`` and ``prob/``
TIFFs named and sized like the anonymized slides, so they overlay directly.

No GPU needed; tile size 224 / stride 112 at downsample 4 become 56 / 28.

Example:
    python -m scripts.musica.export_anonymized \
        --predictions-dir $DATA/predictions/tc-predict-demo-1807852 \
        --report /data/fs201316/jp50463/anonymize_report.csv \
        --anonymized-dir /data/fs201316/jp50463/annonymized \
        --output-dir $DATA/predictions/tc-predict-demo-1807852/anonymized
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ratiopath.openslide import OpenSlide

from ml.callbacks import TiffPredictionMapWriter
from scripts.musica.predict_slides import TILE_SIZE, TILE_STRIDE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--anonymized-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = pd.read_csv(args.report)
    report = report[report["status"] == "ok"].set_index("source")
    slides = pd.read_csv(args.predictions_dir / "slides.csv")["path"].map(Path)

    written = []
    for slide_path in slides:
        if slide_path.name not in report.index:
            print(f"[{slide_path.stem}] not in anonymization report, skipping")
            continue
        predictions = pd.read_parquet(
            args.predictions_dir / "predictions" / f"{slide_path.stem}.parquet"
        )
        class_names = [
            c.removeprefix("prob_")
            for c in map(str, predictions.columns)
            if c.startswith("prob_")
        ]
        probs = predictions[[f"prob_{n}" for n in class_names]].to_numpy(np.float32)

        # Some sources were anonymized more than once under different names;
        # export maps for every copy.
        for _, row in report.loc[[slide_path.name]].iterrows():
            anonymized = args.anonymized_dir / str(row["output"])
            export_slide(
                slide_path,
                anonymized,
                row,
                predictions,
                probs,
                class_names,
                args.output_dir,
            )
            written.append({"source": slide_path.name, "anonymized": anonymized.name})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(written).to_csv(args.output_dir / "mapping.csv", index=False)


def export_slide(
    slide_path: Path,
    anonymized: Path,
    row: pd.Series,
    predictions: pd.DataFrame,
    probs: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    with OpenSlide(slide_path) as slide:
        downsample = slide.level_downsamples[int(row["level"])]
    with OpenSlide(anonymized) as slide:
        extent = slide.level_dimensions[0]
        mpp = slide.slide_resolution(0)

    factor = round(downsample)
    if abs(downsample - factor) > 1e-3 or TILE_STRIDE % factor:
        raise ValueError(f"{slide_path.name}: non-integer downsample {downsample}")
    expected = (int(row["width"]), int(row["height"]))
    if extent != expected:
        raise ValueError(f"{anonymized.name}: {extent} != report {expected}")

    TiffPredictionMapWriter(slides_uri="")._write_slide_maps(
        {
            "path": anonymized.name,
            "extent_x": extent[0],
            "extent_y": extent[1],
            "tile_extent_x": TILE_SIZE // factor,
            "tile_extent_y": TILE_SIZE // factor,
            "stride_x": TILE_STRIDE // factor,
            "stride_y": TILE_STRIDE // factor,
            "mpp_x": mpp[0],
            "mpp_y": mpp[1],
        },
        pd.DataFrame(
            {
                "x": predictions["x"] // factor,
                "y": predictions["y"] // factor,
                "probs": list(probs),
            }
        ),
        output_dir,
        class_names=class_names,
    )
    print(f"[{slide_path.stem}] -> {anonymized.stem} extent={extent} /{factor}")


if __name__ == "__main__":
    main()
