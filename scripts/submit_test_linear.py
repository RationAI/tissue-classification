from kube_jobs import storage, submit_job


checkpoint = "mlflow-artifacts:/104/a23e478b00b04da79cfbf4d91cada8cd/artifacts/checkpoints/last/checkpoint.ckpt"


submit_job(
    job_name="tissue-classification-test-linear-final-adamw",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/ml-test-mode https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        f'PYTHONUNBUFFERED=1 uv run python -m ml +experiment=ml/linear_classifier_test_adamw mode=test checkpoint=\\"{checkpoint}\\"',
    ],
    storage=[storage.secure.PROJECTS],
)
