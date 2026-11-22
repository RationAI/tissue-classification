from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-annotation-masks",
    username=...,
    cpu=8,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",  # pin MLflow v2 (cluster default is now v3)
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.annotation_masks +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
