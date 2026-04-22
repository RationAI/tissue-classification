from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tiling",
    username=...,
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.tiling +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
