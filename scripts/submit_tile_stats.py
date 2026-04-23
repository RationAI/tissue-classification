from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tile-stats",
    username=...,
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.tile_stats +experiment=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
