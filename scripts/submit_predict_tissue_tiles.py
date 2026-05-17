from kube_jobs import storage, submit_job


# Final probe checkpoint to predict with (same convention as submit_test_linear).
checkpoint = "mlflow-artifacts:/104/0e2230c722134ce0985e09a18ccadf75/artifacts/checkpoints/last/checkpoint.ckpt"

# MLflow run of embeddings_virchow2_tissue_tiles_05mpp (all test-split tiles
# intersecting the tissue mask). Fill after that preprocessing run completes.
tissue_embedding_run_id = "FILL_ME"


# Predicts over every test tile intersecting the tissue mask (no labels, no
# metrics). Loads all tissue-tile embeddings into one in-memory array, so this
# needs more memory than the annotated-only test job.
submit_job(
    job_name="tissue-classification-predict-tissue-tiles",
    username="vcifka",
    cpu=8,
    memory="128Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/ml-test-mode https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        f'PYTHONUNBUFFERED=1 uv run python -m ml +experiment=ml/linear_classifier_predict_tissue_tiles mode=predict checkpoint=\\"{checkpoint}\\" tissue_embedding_run_id={tissue_embedding_run_id}',
    ],
    storage=[storage.secure.PROJECTS],
)
