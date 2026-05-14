from kube_jobs import storage, submit_job


submit_job(
    job_name="tissue-classification-train-knn",
    username="vcifka",
    cpu=8,
    memory="64Gi",
    gpu=None,
    public=False,
    script=[
        "git clone --branch feature/ml-linear-classifier https://github.com/RationAI/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m ml +experiment=ml/knn_stratified_group_kfold val_fold=0,1,2,3,4 model.n_neighbors=1,3,5,11,25,51,101 model.weights=uniform,distance model.metric=cosine,euclidean --multirun",
    ],
    storage=[storage.secure.PROJECTS],
)
