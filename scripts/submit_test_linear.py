from kube_jobs import storage, submit_job


checkpoint = "mlflow-artifacts:/104/<run_id>/artifacts/checkpoints/last/checkpoint.ckpt"


submit_job(
    job_name="tissue-classification-test-linear-final-...",
    username=...,
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        f'uv run python -m ml +experiment=... checkpoint=\\"{checkpoint}\\"',
    ],
    storage=[storage.secure.PROJECTS],
)
