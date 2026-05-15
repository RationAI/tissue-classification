from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-linear-probe",
    username="vcifka",
    cpu=4,
    memory="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/linear-probe https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m ml.train +experiment=ml/linear_probe_collapse_alterations_to_other",
    ],
    storage=[storage.secure.PROJECTS],
)
