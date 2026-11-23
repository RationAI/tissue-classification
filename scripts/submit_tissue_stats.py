from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tissue-stats",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run python -m preprocessing.coverage_stats +experiment=...",
    ],
    storage=[storage.secure.PROJECTS],
)
