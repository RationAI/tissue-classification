from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-embeddings",
    username=...,
    public=False,
    cpu=8,
    memory="32Gi",
    shm="16Gi",
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.embeddings +experiment=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
