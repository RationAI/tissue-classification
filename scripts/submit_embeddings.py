from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-embeddings",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=24,
    memory="64Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/tissue-classification.git workdir",
        "cd workdir",
        "export HF_TOKEN=<YOUR_HF_TOKEN>",
        "uv sync --frozen",
        "uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings_05mpp tiling_run_id=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
