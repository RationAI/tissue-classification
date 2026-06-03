from kube_jobs import storage, submit_job


RUN_ID = "0e2230c722134ce0985e09a18ccadf75"
CHECKPOINT_URI = (
    f"mlflow-artifacts:/104/{RUN_ID}/artifacts/checkpoints/last/checkpoint.ckpt"
)

submit_job(
    job_name="tissue-classification-export-onnx",
    username=...,
    cpu=4,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv add onnx onnxscript",
        (
            "uv run python scripts/export_onnx.py "
            f"--checkpoint {CHECKPOINT_URI} "
            "--output linear_head.onnx "
            f"--log-to-run-id {RUN_ID} "
            "--mlflow-artifact-path onnx"
        ),
    ],
    storage=[storage.secure.PROJECTS],
)
