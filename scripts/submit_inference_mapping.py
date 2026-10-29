from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-inference-mapping",
    username=...,
    cpu=4,
    memory="8Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.inference_mapping +experiment=...",
    ],
    storage=[storage.secure.DATA],
)
