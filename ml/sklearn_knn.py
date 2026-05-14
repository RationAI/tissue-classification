from random import randint
from typing import Any

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import DataModule
from ml.sklearn_linear import _log_model, _log_split_metrics


if not OmegaConf.has_resolver("random_seed"):
    OmegaConf.register_new_resolver(
        "random_seed", lambda: randint(0, 2**31), use_cache=True
    )
if not OmegaConf.has_resolver("len"):
    OmegaConf.register_new_resolver("len", lambda x: len(x))


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    run(config)
    mlflow.end_run()


def run(config: DictConfig) -> None:
    if config.mode != "fit":
        raise ValueError("sklearn_knn currently supports only mode='fit'")

    np.random.seed(config.seed)
    data = hydra.utils.instantiate(config.data, _recursive_=False, _target_=DataModule)
    data.setup("fit")
    if "test" in data.datasets:
        data.setup("test")

    x_train = data.train.embeddings
    y_train = data.train.labels
    x_val = data.val.embeddings
    y_val = data.val.labels

    class_names = [
        n for n, _ in sorted(config.class_indices.items(), key=lambda kv: kv[1])
    ]
    model = _build_model(config)
    model.fit(x_train, y_train)

    _log_split_metrics(
        model, x_val, y_val, data.val.slide_ids, class_names, "validation"
    )
    if "test" in data.datasets:
        _log_split_metrics(
            model,
            data.test.embeddings,
            data.test.labels,
            data.test.slide_ids,
            class_names,
            "test",
        )
    if config.model.get("log_model", False):
        _log_model(model)

    params = {
        "model_type": "sklearn_knn",
        "n_neighbors": config.model.n_neighbors,
        "weights": config.model.weights,
        "metric": config.model.metric,
        "algorithm": config.model.algorithm,
        "n_jobs": config.model.n_jobs,
        "standardize": config.model.standardize,
        "log_model": config.model.get("log_model", False),
        "val_fold": config.val_fold,
        "kfold_strategy": config.kfold_strategy,
        "embedding_run_id": config.embedding_run_id,
        "kfold_run_id": config.kfold_run_id,
        "filter_tiles_run_id": config.filter_tiles_run_id,
        "train_tiles": len(y_train),
        "validation_tiles": len(y_val),
    }
    if "test" in data.datasets:
        params["test_tiles"] = len(data.test.labels)
    mlflow.log_params(params)


def _build_model(config: DictConfig) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if config.model.standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=config.model.n_neighbors,
                weights=config.model.weights,
                metric=config.model.metric,
                algorithm=config.model.algorithm,
                n_jobs=config.model.n_jobs,
            ),
        )
    )
    return Pipeline(steps)


if __name__ == "__main__":
    main()
