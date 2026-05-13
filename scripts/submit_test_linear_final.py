from kube_jobs import storage, submit_job


checkpoint = (
    "mlflow-artifacts:/104/<final_fit_run_id>/artifacts/checkpoints/last/checkpoint.ckpt"
)


submit_job(
    job_name="tissue-classification-test-linear-final",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/ml-test-mode https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        f'uv run python -m ml +experiment=ml/linear_classifier_final mode=test checkpoint=\\"{checkpoint}\\"',
    ],
    storage=[storage.secure.PROJECTS],
)
