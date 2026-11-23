from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-remap-annotation-masks",
    cpu=8,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run python -m preprocessing.remap_annotation_masks +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
