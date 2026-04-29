from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tile-masks",
    username="vcifka",
    cpu=8,
    memory="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/implement-tile-masks https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.tile_masks +experiment=preprocessing/tile_masks_05mpp",
    ],
    storage=[storage.secure.DATA],
)
