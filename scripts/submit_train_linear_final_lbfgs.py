from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-train-linear-final-lbfgs",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/ml-test-mode https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m ml +experiment=ml/linear_classifier_final_lbfgs",
    ],
    storage=[storage.secure.PROJECTS],
)
