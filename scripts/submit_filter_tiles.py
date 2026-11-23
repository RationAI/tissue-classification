from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-filter-tiles",
    username=...,
    cpu=8,
    memory="16Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run python -m preprocessing.filter_tiles +experiment=...",
    ],
    storage=[storage.secure.PROJECTS],
)
