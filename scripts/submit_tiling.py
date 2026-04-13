from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tiling",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/tissue-classification.git workdir",
        "cd workdir",
        "git checkout fix/speed-up-the-tiling-process",
        "uv sync",
        "uv run -m preprocessing.tiling +experiment=preprocessing/tiling_05mpp",
    ],
    storage=[storage.secure.DATA],
)
