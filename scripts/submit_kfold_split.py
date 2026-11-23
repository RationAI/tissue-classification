from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-kfold-split",
    username=...,
    cpu=4,
    memory="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run python -m split.kfold_split +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
