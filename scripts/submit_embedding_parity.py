from kube_jobs import storage, submit_job


# Compare a local-backend parity run against the model-service Virchow2 run.
# Fill in the local run id; server run id = dataset.mlflow_artifacts.embedding_run_ids.virchow2
submit_job(
    job_name="tc-embedding-parity",
    public=False,
    cpu=4,
    memory="64Gi",
    script=[
        "git clone -b feature/h-optimus-0 https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000",
        "uv run -m preprocessing.embedding_parity --local-run-id ... --server-run-id 6dbdc6e57fab472b8d8c33b44b82aa6a",
    ],
    storage=[storage.secure.PROJECTS],
)
