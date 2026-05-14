from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-train-logistic-regression",
    username=...,
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        (
            "uv run python -m ml "
            "+experiment=ml/..."
            "val_fold=0,1,2,3,4 "
            "model.C=0.001,0.01,0.1,1,10,100 "
            "--multirun"
        ),
    ],
    storage=[storage.secure.PROJECTS],
)
