from kube_jobs import storage, submit_job

config_dir = "/mnt/projects/tissue_classification/conf/reporter"
config_name = "tissue_classification_lbfgs_mug"

submit_job(
    job_name=f"tissue-classification-report-{config_name.replace('_', '-')}",
    username="vcifka",
    cpu=8,
    memory="16Gi",
    gpu=None,
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/tissue-classification.git workdir",
        "cd workdir",
        "uv sync",
        "uv pip install git+ssh://git@gitlab.ics.muni.cz/rationai/digital-pathology/pipeline/report.git@feature/force-wsi-service-protocol",
        f"uv run python -m report --config-dir {config_dir} --config-name={config_name} user=vcifka"
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)