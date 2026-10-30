"""xOpat report of perineural invasion (PNI) candidates for the pathologist.

One card per slide of the dataset: the slide, the PNI candidate regions of the
detection run (shown on open) and the tissue class map of the prediction run they
were derived from (off on open, toggled in the viewer's layer panel). The page is
written locally and not published anywhere.

Needs only the MLflow tracking API (in-cluster, or tunnelled): the slides are
taken from the dataset's slide map, so nothing has to be mounted where it runs.

    uv run --group report python -m pni.report --output pni_report.html

Run IDs come from the dataset config, so the report always matches the last
recorded pipeline run.
"""

import argparse
import tempfile

import pandas as pd
from omegaconf import OmegaConf
from report_fast import Mask, MlflowRun, Report, SlideCard, case_matrix
from report_fast.mlflow import Mlflow


# Order and names of the probe's classes (collapse_alterations_to_other) and the
# value each one has in the prediction maps: round(255 * (i + 1) / 7), 0 = no tile.
TISSUE_CLASSES = [
    ("Nerve", "#ffd400"),
    ("Blood", "#d62728"),
    ("Connective-Tissue", "#e377c2"),
    ("Fat", "#c7c7c7"),
    ("Epithelium", "#1f77b4"),
    ("Muscle", "#8c564b"),
    ("Other", "#7f7f7f"),
]
CANDIDATE_COLOR = "#00e5ff"


def tissue_mask(run_id: str) -> Mask:
    n = len(TISSUE_CLASSES)
    return Mask(
        "Tissue prediction",
        MlflowRun(run_id, "prediction_maps_tiff/pred"),
        classes=n + 1,
        palette=["#000000", *(color for _, color in TISSUE_CLASSES)],
        # Midpoints between the encoded values k / n, k = 0..n.
        breaks=[round((k + 0.5) / n, 4) for k in range(n)],
        mask=[0] + [1] * n,
        opacity=0.5,
    )


def candidate_mask(run_id: str) -> Mask:
    return Mask(
        "PNI candidates",
        MlflowRun(run_id, "masks"),
        classes=2,
        palette=["#000000", CANDIDATE_COLOR],
        breaks=[0.5],
        mask=[0, 1],
        opacity=0.6,
        visible=True,
    )


def card_note(stem: str, regions: pd.DataFrame, metrics: dict[str, float]) -> str:
    """Region count, and on annotated slides how many annotations they touch."""
    note = f"{len(regions[regions['slide'].str.startswith(stem)])} candidate regions"
    if f"pni_polygons_{stem}" in metrics:
        note += (
            f" · annotated PNI touched {metrics[f'pni_hit_{stem}']:.0f}"
            f"/{metrics[f'pni_polygons_{stem}']:.0f}"
            f" · annotated nerves touched {metrics[f'nerve_hit_{stem}']:.0f}"
            f"/{metrics[f'nerve_polygons_{stem}']:.0f}"
        )
    else:
        note += " · not annotated"
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="configs/data/pni_prostate.yaml")
    parser.add_argument("--output", default="pni_report.html")
    args = parser.parse_args()

    dataset = OmegaConf.load(args.dataset).dataset
    artifacts = dataset.mlflow_artifacts
    detection = OmegaConf.select(artifacts, "pni_detection_run_id")
    if not detection:
        raise SystemExit(f"{args.dataset}: pni_detection_run_id is not set yet")

    flow = Mlflow.from_env()
    with tempfile.TemporaryDirectory() as tmp:
        slide_map = pd.read_csv(
            flow.download(artifacts.mapping_run_id, artifacts.mapping_filename, tmp)
        )
        regions = pd.read_csv(flow.download(detection, "regions.csv", tmp))
    metrics = flow.metrics(detection)

    matrix = case_matrix(
        sorted(slide_map["wsi_path"]),
        [candidate_mask(detection), tissue_mask(artifacts.prediction_run_id)],
        min_layers=2,
        flow=flow,
    )

    legend = ", ".join(f"{name} {color}" for name, color in TISSUE_CLASSES)
    report = Report(
        title="Perineural invasion candidates · prostate case 2025_01403",
        subtitle=(
            f"{len(matrix.sessions)} slides · detection run {detection[:8]}"
            f" · predictions {artifacts.prediction_run_id[:8]}"
        ),
        preamble=(
            "Cyan regions are where the tissue model predicts nerve and epithelium"
            " within two tiles (about 110 µm) of each other, joined into regions of"
            " at least three tiles: candidates for perineural invasion, not"
            " diagnoses. Larger regions were more often true invasion on the"
            " annotated slides. The 'Tissue prediction' layer (off on open) shows"
            f" the underlying tile classes: {legend}. Slides -17 and -21 carry your"
            " QuPath annotations, so their counts say how many annotated PNI and"
            " nerve outlines a candidate touches; -23 and -25 are unannotated."
        ),
    )
    for session in matrix.sessions:
        stem = session.names[0]
        report.add(SlideCard(session, note=card_note(stem, regions, metrics)))

    path = report.write(args.output)
    print(f"wrote {path}")
    print("coverage:", matrix.coverage, "dropped:", matrix.dropped)


if __name__ == "__main__":
    main()
