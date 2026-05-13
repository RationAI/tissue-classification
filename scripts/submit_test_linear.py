from kube_jobs import storage, submit_job


fold_checkpoints = {
    0: "runs:/<fold_0_fit_run_id>/checkpoints/<best>.ckpt",
    1: "runs:/<fold_1_fit_run_id>/checkpoints/<best>.ckpt",
    2: "runs:/<fold_2_fit_run_id>/checkpoints/<best>.ckpt",
    3: "runs:/<fold_3_fit_run_id>/checkpoints/<best>.ckpt",
    4: "runs:/<fold_4_fit_run_id>/checkpoints/<best>.ckpt",
}


submit_job(
    job_name="tissue-classification-test-linear",
    username=...,
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        *[
            "uv run python -m ml +experiment=ml/linear_classifier "
            f"mode=test val_fold={fold} checkpoint={checkpoint}"
            for fold, checkpoint in fold_checkpoints.items()
        ],
    ],
    storage=[storage.secure.PROJECTS],
)
