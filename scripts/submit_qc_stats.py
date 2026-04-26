from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-qc-stats",
    username=...,
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.qc_stats +experiment=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
