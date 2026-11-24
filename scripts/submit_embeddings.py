from kube_jobs import storage, submit_job


# For `backend: local` (model run on the job's GPU, e.g. H-Optimus-0) also set
# gpu=..., and export HF_HOME=... pointing to a cache with the HF token.
submit_job(
    job_name="tissue-classification-embeddings",
    public=False,
    cpu=8,
    memory="64Gi",
    shm="24Gi",
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run -m preprocessing.embeddings +experiment=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
