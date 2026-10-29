from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-tissue-masks",
    username="vcifka",
    cpu=16,
    memory="32Gi",
    gpu=None,
    public=False,
    script=[
        "git clone -b feat/pni-prostate https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.tissue_masks +experiment=preprocessing/pni_prostate_tissue_masks_2mpp",
    ],
    storage=[storage.secure.DATA],
)
