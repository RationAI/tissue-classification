from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-threshold-stats",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone --branch https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.threshold_stats +experiment=...",
    ],
    storage=[storage.secure.PROJECTS],
)
