from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-threshold-stats",
    cpu=8,
    memory="32Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run python -m preprocessing.threshold_stats +experiment=...",
    ],
    storage=[storage.secure.PROJECTS],
)
