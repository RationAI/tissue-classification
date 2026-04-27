from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-wsi-mapping",
    username=...,
    cpu=4,
    memory="8Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.wsi_mapping",
    ],
    storage=[storage.secure.DATA],
)
