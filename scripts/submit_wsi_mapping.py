from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-wsi-mapping",
    cpu=4,
    memory="8Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run python -m preprocessing.wsi_mapping",
    ],
    storage=[storage.secure.DATA],
)
