from kube_jobs import storage, submit_job


# Local-GPU embedding (backend: local), e.g. for models not on the model service.
# Experiments: preprocessing/embeddings_hoptimus0_0_5mpp,
#              preprocessing/embeddings_virchow2_local_parity
# Gated HF models: the read-only HF token must already exist at
# $HF_HOME/token on the shared project volume (never put it in this script).
submit_job(
    job_name="tc-embeddings-local",
    public=False,
    cpu=32,
    memory="128Gi",
    shm="48Gi",
    gpu="A40",
    script=[
        "git clone -b feature/h-optimus-0 https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "export HF_HOME=/mnt/projects/tissue_classification/hf_cache",
        "nvidia-smi",
        "uv run -m preprocessing.embeddings +experiment=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
