"""Upload pyramidal TIFF copies of the linkage slides to the predictions run.

Layout in the run: backgrounds/SLD-*.tiff (vips tiffsave --pyramid of
/mnt/data/Public/MUG/linkage/SLD-*.svs, JPEG Q90, 512 px tiles). Stdlib only.
"""

import urllib.request
from pathlib import Path


TRACKING = "http://mlflow.rationai-mlflow:5000"
EXPERIMENT_ID = "104"
RUN_ID = "30f1f669d79b458aba5c210a41a23897"
SOURCE = Path.home() / "musica" / "backgrounds"

files = sorted(SOURCE.glob("SLD-*.tiff"))
assert len(files) == 10, [f.name for f in files]
prefix = f"{TRACKING}/api/2.0/mlflow-artifacts/artifacts/{EXPERIMENT_ID}/{RUN_ID}/artifacts/backgrounds"
for i, f in enumerate(files, 1):
    with f.open("rb") as body:
        req = urllib.request.Request(
            f"{prefix}/{f.name}",
            data=body,
            method="PUT",
            headers={
                "Content-Type": "image/tiff",
                "Content-Length": str(f.stat().st_size),
            },
        )
        urllib.request.urlopen(req, timeout=1800).read()
    print(f"{i}/{len(files)} {f.name} {f.stat().st_size / 1e9:.2f} GB", flush=True)
print("DONE")
