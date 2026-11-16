"""xOpat report of perineural invasion (PNI) candidates for the pathologist.

One card per slide of the dataset, with Czech text for the pathologist: the
slide, the nerve side of the PNI candidate regions (shown on open), the nerve
and epithelium detections and probabilities of the prediction run they were
derived from and the pathologist's outlines (off on open, toggled in the
viewer's layer panel). The page is
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
from report_fast import Mask, MlflowRun, Report, SlideGrid, case_matrix
from report_fast.mlflow import Mlflow


# Order of the probe's classes (collapse_alterations_to_other), their Czech
# label and colour, and the value each one has in the prediction maps:
# round(255 * (i + 1) / 7), 0 = no tile.
TISSUE_CLASSES = [
    ("Nerv", "#ffd400", "žlutě"),
    ("Krev", "#d62728", "červeně"),
    ("Vazivo", "#e377c2", "růžově"),
    ("Tuk", "#c7c7c7", "šedě"),
    ("Epitel", "#1f77b4", "modře"),
    ("Sval", "#8c564b", "hnědě"),
    ("Ostatní", "#7f7f7f", "tmavě šedě"),
]
CANDIDATE_COLOR = "#00e5ff"
# Annotation mask values: 0 background, 128 Nerve, 255 PNI (pni.detect).
ANNOTATION_COLORS = {"Nerve": "#00c853", "PNI": "#ff00ff"}


def detector_mask(run_id: str, class_name: str) -> Mask:
    """One class of the fused tissue map: the cells the detection used as it."""
    labels = [name for name, _, _ in TISSUE_CLASSES]
    n = len(TISSUE_CLASSES)
    index = labels.index(class_name)
    return Mask(
        f"Detektor: {class_name.lower()}",
        MlflowRun(run_id, "prediction_maps_tiff/pred"),
        classes=n + 1,
        palette=["#000000", *(color for _, color, _ in TISSUE_CLASSES)],
        breaks=[round((k + 0.5) / n, 4) for k in range(n)],
        mask=[0] + [int(i == index) for i in range(n)],
        opacity=0.5,
    )


def probability_mask(run_id: str, class_name: str, folder: str) -> Mask:
    """Fused probability of one class, drawn as a heatmap of its colour."""
    color = next(c for name, c, _ in TISSUE_CLASSES if name == class_name)
    return Mask(
        f"Pravděpodobnost: {class_name.lower()}",
        MlflowRun(run_id, f"prediction_maps_tiff/prob/{folder}"),
        color=color,
        opacity=0.7,
    )


def candidate_mask(run_id: str) -> Mask:
    """Nerve of the candidate regions: the invaded nerve to look at."""
    return Mask(
        "Kandidáti PNI (nerv)",
        MlflowRun(run_id, "nerve_masks"),
        classes=2,
        palette=["#000000", CANDIDATE_COLOR],
        breaks=[0.5],
        mask=[0, 1],
        opacity=0.6,
        visible=True,
    )


def annotation_mask(run_id: str) -> Mask:
    return Mask(
        "Anotace patologa",
        MlflowRun(run_id, "annotations"),
        classes=3,
        palette=["#000000", ANNOTATION_COLORS["Nerve"], ANNOTATION_COLORS["PNI"]],
        breaks=[0.25, 0.75],
        mask=[0, 1, 1],
        opacity=0.5,
    )


def card_note(stem: str, metrics: dict[str, float]) -> str:
    """Candidate count, and on annotated slides how many annotations they touch.

    Counts are of the nerve shapes, the layer shown on open.
    """
    note = f"{metrics[f'nerve_regions_{stem}']:.0f} kandidátů"
    m = "nerve_shape_"
    if f"{m}pni_polygons_{stem}" in metrics:
        note += (
            f" · zasaženo anotací PNI {metrics[f'{m}pni_hit_{stem}']:.0f}"
            f"/{metrics[f'{m}pni_polygons_{stem}']:.0f}"
            f" · zasaženo anotací nervů {metrics[f'{m}nerve_hit_{stem}']:.0f}"
            f"/{metrics[f'{m}nerve_polygons_{stem}']:.0f}"
        )
    else:
        note += " · bez anotací"
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
    metrics = flow.metrics(detection)
    predictions = artifacts.prediction_run_id

    matrix = case_matrix(
        sorted(slide_map["wsi_path"]),
        [
            candidate_mask(detection),
            detector_mask(predictions, "Nerv"),
            detector_mask(predictions, "Epitel"),
            probability_mask(predictions, "Nerv", "Nerve"),
            probability_mask(predictions, "Epitel", "Epithelium"),
            annotation_mask(detection),
        ],
        min_layers=2,
        flow=flow,
    )

    report = Report(
        title="Kandidáti na perineurální invazi · prostata, případ 2025_01403",
        subtitle=(
            f"{len(matrix.sessions)} skeny · detekce {detection[:8]}"
            f" · predikce tkáně {artifacts.prediction_run_id[:8]}"
        ),
        preamble=(
            "Tyrkysově vyznačení kandidáti na perineurální invazi jsou místa, která"
            " model klasifikuje jako nerv a hned vedle nich (v sousední dlaždici,"
            " cca 110 µm) jako epitel. Nejde o diagnózy. Zobrazuje se jen nervová"
            " část kontaktu. Další vrstvy jsou při otevření vypnuté a zapínají se"
            " v panelu vrstev. Pro rozbor chyb: 'Detektor: nerv' (žlutě) a"
            " 'Detektor: epitel' (modře) ukazují, kde model danou tkáň určil a"
            " detekce s ní počítala; 'Pravděpodobnost: nerv / epitel' ukazuje, jak"
            " si tím model byl jistý (sytější barva = vyšší pravděpodobnost), takže"
            " je vidět i tkáň, kterou model téměř určil jinak. U skenů -17 a -21"
            " jsou také Vaše anotace z QuPathu (nervy zeleně, PNI purpurově). Skeny -23 a -25"
            " anotované nejsou."
        ),
    )
    # Narrow columns keep all four slides in one row on a laptop screen.
    grid = SlideGrid(min_width=180, card={"thumbnail_size": 256})
    for session in matrix.sessions:
        stem = session.names[0]
        grid.add(session, note=card_note(stem, metrics))
    report.add(grid)

    path = report.write(args.output)
    print(f"wrote {path}")
    print("coverage:", matrix.coverage, "dropped:", matrix.dropped)


if __name__ == "__main__":
    main()
