from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-train-linear",
    username=...,
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m ml +experiment=... val_fold=0,1,2,3,4 --multirun",
    ],
    storage=[storage.secure.PROJECTS],
)
