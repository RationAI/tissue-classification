from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tiling",
    username="vcifka",
    cpu=8,
    memory="32Gi",  # approximately 4GiB per process
    public=False,
    script=[
        "git clone --branch feature/implement-tiling-script https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.tiling +experiment=preprocessing/tiling_05mpp",
    ],
    storage=[storage.secure.DATA],
)
