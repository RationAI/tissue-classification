from kube_jobs import storage, submit_job


submit_job(
    job_name="tc-embeddings-local",
    public=False,
    cpu=16,
    memory="64Gi",
    shm="48Gi",
    gpu="A40",
    script=[
        "git clone -b feature/h-optimus-0 https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "export HF_HOME=/mnt/projects/tissue_classification/hf_cache",
        "nvidia-smi",
        "uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings_hoptimus0_0_5mpp",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
