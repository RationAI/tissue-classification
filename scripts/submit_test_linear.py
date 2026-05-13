from kube_jobs import storage, submit_job


fold_checkpoints = {
    0: "mlflow-artifacts:/104/26a6f9c741d543c9a09b54048be527a1/artifacts/checkpoints/epoch=3-val_loss=0.1973/checkpoint.ckpt",
    1: "mlflow-artifacts:/104/cc2be862324a446baffd9a8d90be604d/artifacts/checkpoints/epoch=1-val_loss=0.1218/checkpoint.ckpt",
    2: "mlflow-artifacts:/104/8454857b11984419bb7eae02a520ec71/artifacts/checkpoints/epoch=0-val_loss=0.2980/checkpoint.ckpt",
    3: "mlflow-artifacts:/104/bfa52277ea2744b9ab523c56a905dcda/artifacts/checkpoints/epoch=0-val_loss=1.0547/checkpoint.ckpt",
    4: "mlflow-artifacts:/104/358cd6ee286b4d67b7c12cf9bce0c3b4/artifacts/checkpoints/epoch=0-val_loss=0.1462/checkpoint.ckpt",
}


submit_job(
    job_name="tissue-classification-test-linear",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/ml-test-mode https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        *[
            "uv run python -m ml +experiment=ml/linear_classifier "
            f'mode=test val_fold={fold} checkpoint=\\"{checkpoint}\\"'
            for fold, checkpoint in fold_checkpoints.items()
        ],
    ],
    storage=[storage.secure.PROJECTS],
)
