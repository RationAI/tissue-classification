from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-filter-tiles",
    username=...,
    cpu=8,
    memory="16Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.filter_tiles +experiment=preprocessing/filter_tiles",
    ],
    storage=[storage.secure.PROJECTS],
)
