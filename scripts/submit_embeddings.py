from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-embeddings",
    username=...,
    cpu=24,
    memory="64Gi",
    gpu="A40",
    public=False,
    script=[
        "git clone https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "export HF_TOKEN=<YOUR_HF_TOKEN>",
        "uv sync",
        "uv run -m preprocessing.embeddings +experiment=preprocessing/embeddings_05mpp",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
