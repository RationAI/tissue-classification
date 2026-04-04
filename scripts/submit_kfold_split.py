from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-kfold-split",
    username=...,
    cpu=4,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m split.kfold_split +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
