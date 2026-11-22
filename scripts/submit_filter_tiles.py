from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-filter-tiles",
    username=...,
    cpu=8,
    memory="16Gi",
    public=False,
    script=[
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",  # pin MLflow v2 (cluster default is now v3)
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.filter_tiles +experiment=...",
    ],
    storage=[storage.secure.PROJECTS],
)
