from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tile-masks",
    username=...,
    cpu=8,
    memory="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.tile_masks +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
