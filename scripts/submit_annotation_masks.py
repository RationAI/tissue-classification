from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-annotation-masks",
    username=...,
    cpu=8,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.annotation_masks +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
