from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-embedding-dataset",
    username=...,
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.embedding_dataset +experiment=...",
    ],
    storage=[storage.secure.PROJECTS],
)
