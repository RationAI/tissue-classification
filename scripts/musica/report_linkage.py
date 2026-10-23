# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "report-fast[mlflow] @ git+https://github.com/RationAI/ReportFast@810c09f",
#     "mlflow>=2.8,<3",
# ]
# ///
"""ReportFast report of the MUSICA tissue-class predictions on the linkage slides.

Everything comes from MLflow run ``musica_linkage_predictions``:

- backgrounds (``backgrounds/``): pyramidal TIFF copies of the anonymized
  slides ``/mnt/data/Public/MUG/linkage/SLD-*.svs``, which are single-level and
  load blurred and slowly in the viewer (``upload_musica_backgrounds.py``);
- overlays, logged by ``upload_musica_predictions.py``:

  - ``Tissue classes``: the class map (``pred/``) as a colormap, shown on open;
  - one heatmap per class (``prob/<class>/``), hidden until picked.

The builder never opens a slide, so it runs anywhere the MLflow tracking API
answers. Run from the repo root:

    MLFLOW_TRACKING_URI=https://mlflow.rationai.cloud.trusted.e-infra.cz \
        uv run --script scripts/musica/report_linkage.py musica_linkage.html
"""

import sys

from report_fast import Mask, MlflowRun, Report, SlideGrid, XopatEndpoint, case_matrix


RUN_ID = "30f1f669d79b458aba5c210a41a23897"  # TissueClassification/musica_linkage_predictions
SLIDES = MlflowRun(RUN_ID, "backgrounds")
# Read everything through WSI-Service, the protocol registered on the cluster.
ENDPOINT = XopatEndpoint(image_protocol="wsi_service")

# One card per slide. SLD-640ca20c is a second anonymized copy of SLD-92b035e0
# (same source scan), so it is left out.
CASES = [
    "SLD-ea951570",
    "SLD-62fda854",
    "SLD-bad3116b",
    "SLD-60257199",
    "SLD-cd6665d8",
    "SLD-92b035e0",
    "SLD-06b42b6f",
    "SLD-b0c952af",
    "SLD-0b6d9611",
    "SLD-f25a3c68",
]

# Class order and colours as in predict_slides.py (checkpoint class_indices).
CLASSES = {
    "Nerve": "#ffd700",
    "Blood": "#dc143c",
    "Connective-Tissue": "#4169e1",
    "Fat": "#fffacd",
    "Epithelium": "#9400d3",
    "Muscle": "#ff7f50",
    "Other": "#808080",
}

# pred/ stores background 0 and class i as round(255 * (i + 1) / 7), i.e.
# 0, 1/7, ..., 7/7 after normalisation: 8 bins with boundaries halfway between
# neighbouring values, background bin not painted.
N = len(CLASSES)
CLASS_MAP = Mask(
    "Tissue classes",
    MlflowRun(RUN_ID, "prediction_maps_tiff/pred"),
    classes=N + 1,
    palette=["#000000", *CLASSES.values()],
    breaks=[round((2 * k + 1) / (2 * N), 4) for k in range(N)],
    mask=[0] + [1] * N,
    opacity=0.5,
    visible=True,
)
PROBABILITIES = [
    Mask(
        f"P({name})",
        MlflowRun(RUN_ID, f"prediction_maps_tiff/prob/{name}"),
        color=color,
    )
    for name, color in CLASSES.items()
]


def main(output: str) -> None:
    matrix = case_matrix(
        SLIDES,
        [CLASS_MAP, *PROBABILITIES],
        only=CASES,
        min_layers=1 + len(PROBABILITIES),
        endpoint=ENDPOINT,
    )
    # Sample the class map with gl.NEAREST: linear filtering between two class
    # values would paint a third class along every boundary.
    for session in matrix.sessions:
        for entry in session.data:
            if (
                isinstance(entry, dict)
                and "/prediction_maps_tiff/pred/" in entry["dataID"]
            ):
                entry["imageSmoothingEnabled"] = False
    legend = ", ".join(f"{name} ({color})" for name, color in CLASSES.items())
    Report(
        title="Tissue classification on MUG prostate slides",
        subtitle=f"{len(matrix.sessions)} slides, linear Virchow2 classifier",
        preamble=(
            "Model predictions, not pathologist annotations. Inference ran on the "
            "original scans at ~0.25 µm/px (224 px tiles, 112 px stride); the maps "
            "were scaled 4x down to these ~1.0 µm/px anonymized copies. "
            f"Tissue classes: {legend}. Per-class probability layers are off by "
            "default and can be switched on in the viewer. Prostate was not among "
            "the training organs, and tiles at the tissue edge tend to be called Fat."
        ),
    ).add(SlideGrid(sessions=matrix.sessions)).write(output)

    print(f"wrote {output}")
    print("coverage", matrix.coverage)
    print("dropped", matrix.dropped)
    for session in matrix.sessions:
        print(session.url(ENDPOINT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "musica_linkage.html")
