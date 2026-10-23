"""Upload MUSICA linkage predictions to a new MLflow run (stdlib only).

Layout in the run:
    prediction_maps_tiff/pred/SLD-*.tiff
    prediction_maps_tiff/prob/<class>/SLD-*.tiff
    prediction_maps_tiff/mapping.csv   (MUSICA source -> SLD name, both runs)
"""

import json
import time
import urllib.request
from pathlib import Path


TRACKING = "http://mlflow.rationai-mlflow:5000"
EXPERIMENT_ID = "104"  # TissueClassification
ROOT = Path.home() / "musica"
RUNS = ["tc-predict-demo-1807852", "tc-predict-anon-1848669"]
ARTIFACT_DIR = "prediction_maps_tiff"


def api(method, path, body=None, data=None, ctype="application/json"):
    payload = json.dumps(body).encode() if body is not None else data
    req = urllib.request.Request(
        TRACKING + path, data=payload, method=method, headers={"Content-Type": ctype}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = resp.read()
    return json.loads(text) if text else {}


files = {}
mapping = ["source,anonymized"]
for run in RUNS:
    base = ROOT / run / "anonymized"
    for f in sorted(base.rglob("*.tiff")):
        files[f.relative_to(base).as_posix()] = f
    mapping += base.joinpath("mapping.csv").read_text().splitlines()[1:]
print(f"{len(files)} tiffs, {len(mapping) - 1} mapping rows")
assert len(files) == 88, len(files)

now = int(time.time() * 1000)
run = api(
    "POST",
    "/api/2.0/mlflow/runs/create",
    {
        "experiment_id": EXPERIMENT_ID,
        "run_name": "musica_linkage_predictions",
        "start_time": now,
        "tags": [
            {
                "key": "mlflow.note.content",
                "value": "Linear Virchow2 L-BFGS classifier (checkpoint run 0e2230c7) predictions for 10 "
                "prostate MUG slides. Inference on MUSICA on the original slides (level 0, ~0.25 mpp, "
                "tile 224 / stride 112, tissue coverage > 0); maps re-rendered 4x down to the geometry "
                "of the anonymized copies in /mnt/data/Public/MUG/linkage (SLD-*.svs, ~1.0 mpp). "
                "pred: uint8 round(255*(i+1)/7) for Nerve, Blood, Connective-Tissue, Fat, Epithelium, "
                "Muscle, Other; 0 = background. prob/<class>: probability * 255.",
            },
            {
                "key": "source",
                "value": "scripts/musica (predict_slides.py + export_anonymized.py)",
            },
        ],
    },
)["run"]["info"]
run_id = run["run_id"]
print("run_id", run_id, flush=True)

prefix = f"/api/2.0/mlflow-artifacts/artifacts/{EXPERIMENT_ID}/{run_id}/artifacts/{ARTIFACT_DIR}"
api(
    "PUT",
    f"{prefix}/mapping.csv",
    data=("\n".join(mapping) + "\n").encode(),
    ctype="text/csv",
)
for i, (rel, f) in enumerate(files.items(), 1):
    api("PUT", f"{prefix}/{rel}", data=f.read_bytes(), ctype="image/tiff")
    if i % 11 == 0:
        print(f"{i}/{len(files)}", flush=True)

api(
    "POST",
    "/api/2.0/mlflow/runs/update",
    {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)},
)
print("DONE", run_id)
