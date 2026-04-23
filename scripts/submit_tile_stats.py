from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tile-stats",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone --branch feature/tile-statistics https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.tile_stats +experiment=preprocessing/tile_stats_05mpp",
    ],
    storage=[storage.secure.PROJECTS],
)
