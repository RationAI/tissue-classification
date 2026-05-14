from pathlib import Path
from random import randint
from typing import Any

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import DataModule
from ml.meta_arch import _confmat_figure


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
        raise ValueError("sklearn_linear currently supports only mode='fit'")

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
    _log_convergence(model, config)

    _log_split_metrics(
        model, x_val, y_val, data.val.slide_ids, class_names, "validation"
    )
    if hasattr(data, "test"):
        _log_split_metrics(
            model,
            data.test.embeddings,
            data.test.labels,
            data.test.slide_ids,
            class_names,
            "test",
        )
    _log_model(model)

    params = {
        "model_type": "sklearn_logistic_regression",
        "solver": config.model.solver,
        "penalty": config.model.penalty,
        "C": config.model.C,
        "max_iter": config.model.max_iter,
        "tol": config.model.tol,
        "class_weight": config.model.class_weight,
        "standardize": config.model.standardize,
        "val_fold": config.val_fold,
        "kfold_strategy": config.kfold_strategy,
        "embedding_run_id": config.embedding_run_id,
        "kfold_run_id": config.kfold_run_id,
        "filter_tiles_run_id": config.filter_tiles_run_id,
        "train_tiles": len(y_train),
        "validation_tiles": len(y_val),
    }
    if hasattr(data, "test"):
        params["test_tiles"] = len(data.test.labels)
    mlflow.log_params(params)


def _build_model(config: DictConfig) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if config.model.standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=config.model.C,
                class_weight=config.model.class_weight,
                max_iter=config.model.max_iter,
                n_jobs=config.model.n_jobs,
                penalty=config.model.penalty,
                random_state=config.seed,
                solver=config.model.solver,
                tol=config.model.tol,
                verbose=config.model.verbose,
            ),
        )
    )
    return Pipeline(steps)


def _log_split_metrics(
    model: Pipeline,
    inputs: np.ndarray,
    targets: np.ndarray,
    slide_ids: np.ndarray,
    class_names: list[str],
    split: str,
) -> None:
    labels = np.arange(len(class_names))
    preds = model.predict(inputs)
    probs = _predict_proba_for_all_classes(model, inputs, labels)

    mlflow.log_metric(
        f"{split}/acc_macro",
        recall_score(targets, preds, labels=labels, average="macro", zero_division=0),
    )
    mlflow.log_metric(
        f"{split}/f1_macro",
        f1_score(targets, preds, average="macro", zero_division=0),
    )

    per_class_acc = recall_score(
        targets, preds, labels=labels, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        targets, preds, labels=labels, average=None, zero_division=0
    )
    for cls_name, acc, f1 in zip(
        class_names, per_class_acc.tolist(), per_class_f1.tolist(), strict=True
    ):
        mlflow.log_metric(f"{split}/acc_per_class/{cls_name}", acc)
        mlflow.log_metric(f"{split}/f1_per_class/{cls_name}", f1)

    matrix = confusion_matrix(targets, preds, labels=labels)
    fig = _confmat_figure(matrix, class_names, title=f"{split} confmat")
    try:
        mlflow.log_figure(fig, artifact_file=f"confusion_matrix/{split}.png")
    finally:
        plt.close(fig)

    prob_columns = [f"prob_{c}" for c in class_names]
    predictions = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "target": targets,
            "pred": preds,
        }
    )
    predictions = pd.concat(
        [predictions, pd.DataFrame(probs, columns=prob_columns)], axis=1
    )
    out_path = Path(f"{split}_predictions.parquet")
    predictions.to_parquet(out_path, index=False)
    mlflow.log_artifact(str(out_path), artifact_path="predictions")
    _log_per_slide_accuracy(predictions, split)


def _log_per_slide_accuracy(predictions: pd.DataFrame, split: str) -> None:
    rows = []
    for slide_id, slide_df in predictions.groupby("slide_id"):
        rows.append(
            {
                "slide_id": slide_id,
                "tile_accuracy": float((slide_df["pred"] == slide_df["target"]).mean()),
                "n_tiles": len(slide_df),
            }
        )
    if not rows:
        return

    per_slide = pd.DataFrame(rows)
    mlflow.log_metric(f"{split}/slide_acc_mean", per_slide["tile_accuracy"].mean())
    mlflow.log_metric(f"{split}/slide_acc_median", per_slide["tile_accuracy"].median())
    mlflow.log_metric(f"{split}/slide_acc_min", per_slide["tile_accuracy"].min())
    mlflow.log_table(per_slide, artifact_file=f"per_slide/{split}_tile_accuracy.json")


def _predict_proba_for_all_classes(
    model: Pipeline, inputs: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    raw_probs = model.predict_proba(inputs)
    probs = np.zeros((len(inputs), len(labels)), dtype=raw_probs.dtype)
    for source_idx, class_idx in enumerate(model.classes_):
        matching = np.flatnonzero(labels == class_idx)
        if len(matching) == 1:
            probs[:, matching[0]] = raw_probs[:, source_idx]
    return probs


def _log_convergence(model: Pipeline, config: DictConfig) -> None:
    classifier = model.named_steps["classifier"]
    n_iter = int(classifier.n_iter_.max())
    mlflow.log_metric("n_iter", n_iter)
    mlflow.log_param("converged", n_iter < config.model.max_iter)


def _log_model(model: Pipeline) -> None:
    mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
