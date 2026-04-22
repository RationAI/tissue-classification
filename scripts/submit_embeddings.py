from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-embeddings",
    username=...,
    cpu=8,
    memory="32Gi",
    gpu=1,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings_05mpp tiling_run_id=...",
    ],
    storage=[storage.secure.DATA],
)
