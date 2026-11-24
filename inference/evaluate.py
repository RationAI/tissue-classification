"""Evaluation of cached predictions against GeoJSON ground truth.

Ground-truth tile labels follow the training rule exactly:

- per-class coverage of the tile's central ROI (half-size square, as in
  ``preprocessing/tiling.py``) by the union of that class's annotations
- ``preprocessing._labels.assign_tile_labels`` with the training thresholds and
  ``tissue_prop_min`` (from ``configs/ml/task/final_linear_classifier.yaml``)

Outputs (``<out>/evaluation/``):

- ``metrics.json``          everything below, machine readable
- ``per_class.csv``         precision / recall / F1 / support per model, variant, class
- ``per_slide.csv``         macro F1 and tile counts per model, variant, slide
- ``label_predictions.csv`` original GT label -> predicted class distribution
- ``label_inventory.csv``   labels found in the annotations and how they were mapped
- ``confusion_<model>_<variant>.png``, ``slides/<slide>.png`` (thumbnail | GT | models)
- ``report.html``           all of it on one page
"""

from __future__ import annotations

import base64
import html
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely
from omegaconf import OmegaConf
from shapely import STRtree
from shapely.geometry import shape
from shapely.ops import unary_union

from preprocessing._labels import REASON_OK, ROI_PREFIX, assign_tile_labels
from preprocessing.annotations import (
    QUPATH_LABEL_KEY,
    geojson_feature_label,
    get_validated_polygons,
    open_annotations,
)


if TYPE_CHECKING:
    from inference.predict import Panel
    from inference.slides import SlideGeometry


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_TASK_CONFIG = REPO_ROOT / "configs/ml/task/final_linear_classifier.yaml"
RAW_LABEL_MAJORITY = 0.5  # ROI share for the "GT label -> prediction" table

# Mapping targets that are not one of our classes.
IGNORED = "<ignored>"
UNMAPPED = "<UNMAPPED>"
NO_LABEL = "<no classification>"


# --------------------------------------------------------------------------- config


@dataclass
class LabelMap:
    path: Path
    label_key: str
    annotation_patterns: list[str]
    classes: dict[str, list[str]]
    ignore: list[str]
    merged_variants: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> LabelMap:
        raw: Any = OmegaConf.to_container(OmegaConf.load(path))
        return cls(
            path=path,
            label_key=raw.get("label_key", QUPATH_LABEL_KEY),
            annotation_patterns=raw.get(
                "annotation_patterns", ["{stem}.geojson", "{stem}.geo.json"]
            ),
            classes={k: list(v) for k, v in raw["classes"].items()},
            ignore=list(raw.get("ignore", [])),
            merged_variants=raw.get("merged_variants") or {},
        )

    def target_of(self, label: str | None) -> str | None:
        """Our class for an original label, ``IGNORED``, or None if unmapped."""
        if label is None:
            return IGNORED
        low = label.lower()
        for target, names in self.classes.items():
            if low in (n.lower() for n in names):
                return target
        if low in (n.lower() for n in self.ignore):
            return IGNORED
        return None


def training_label_rule() -> tuple[dict[str, float], float]:
    """Per-class thresholds and tissue_prop_min used to label training tiles."""
    cfg: Any = OmegaConf.to_container(OmegaConf.load(TRAINING_TASK_CONFIG))
    return {k: float(v) for k, v in cfg["thresholds"].items()}, float(
        cfg["tissue_prop_min"]
    )


def find_annotation(slide: Path, directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        candidate = directory / pattern.format(stem=slide.stem)
        if candidate.exists():
            return candidate
    return None


# ------------------------------------------------------------------ label inventory


def label_inventory(files: list[Path], label_map: LabelMap) -> pd.DataFrame:
    """Every label in the annotation files: count, area, files, mapping target."""
    stats: dict[str | None, dict[str, Any]] = defaultdict(
        lambda: {"features": 0, "area_px": 0.0, "files": set()}
    )
    for path in files:
        for feature in json.loads(path.read_text()).get("features", []):
            if not feature.get("geometry"):
                continue
            label = geojson_feature_label(
                feature.get("properties"), label_map.label_key
            )
            entry = stats[label]
            entry["features"] += 1
            entry["area_px"] += float(shape(feature["geometry"]).area)
            entry["files"].add(path.name)
    rows = []
    for label, entry in stats.items():
        target = label_map.target_of(label)
        rows.append(
            {
                "label": label if label is not None else NO_LABEL,
                "maps_to": target if target is not None else UNMAPPED,
                "features": entry["features"],
                "files": len(entry["files"]),
                "area_px": entry["area_px"],
            }
        )
    return pd.DataFrame(rows).sort_values("area_px", ascending=False, ignore_index=True)


# ------------------------------------------------------------------ ground truth


def _roi_boxes(xs: np.ndarray, ys: np.ndarray, tile: int, downsample: float) -> Any:
    """Central half-size ROI of each tile, in level-0 coordinates (as tiling.py)."""
    x0 = (xs + tile / 4) * downsample
    y0 = (ys + tile / 4) * downsample
    size = tile / 2 * downsample
    return shapely.box(x0, y0, x0 + size, y0 + size)


def _coverage(boxes: Any, polygons: list[Any], box_area: float) -> np.ndarray:
    """Fraction of each box covered by the union of ``polygons``."""
    out = np.zeros(len(boxes), dtype=np.float64)
    if not polygons:
        return out
    hits = np.unique(STRtree(polygons).query(boxes, predicate="intersects")[0])
    if len(hits) == 0:
        return out
    union = unary_union(polygons)
    shapely.prepare(union)
    out[hits] = shapely.area(shapely.intersection(boxes[hits], union)) / box_area
    return out


def ground_truth_tiles(
    annotation: Path,
    tiles: pd.DataFrame,
    tile_size: int,
    downsample: float,
    label_map: LabelMap,
    thresholds: dict[str, float],
    tissue_prop_min: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GT label per tile (training rule) and ROI coverage per original label."""
    parser = open_annotations(annotation)
    if parser is None:
        raise ValueError(f"unsupported annotation file: {annotation}")
    xs, ys = tiles["x"].to_numpy(), tiles["y"].to_numpy()
    boxes = _roi_boxes(xs, ys, tile_size, downsample)
    box_area = (tile_size / 2 * downsample) ** 2

    roi = pd.DataFrame(index=tiles.index)
    for target, names in label_map.classes.items():
        polygons = get_validated_polygons(parser, names, label_map.label_key)
        roi[f"{ROI_PREFIX}{target}"] = _coverage(boxes, polygons, box_area)
    labelled = assign_tile_labels(roi, thresholds, tissue_prop_min)
    gt = pd.concat([tiles[["x", "y"]], roi, labelled], axis=1)

    raw_labels = sorted(
        {
            lbl
            for f in json.loads(annotation.read_text()).get("features", [])
            if (lbl := geojson_feature_label(f.get("properties"), label_map.label_key))
            is not None
        }
    )
    raw = pd.DataFrame(index=tiles.index)
    for label in raw_labels:
        polygons = get_validated_polygons(parser, [label], label_map.label_key)
        raw[label] = _coverage(boxes, polygons, box_area)
    return gt, raw


# ------------------------------------------------------------------ metrics


def confusion(gt: np.ndarray, pred: np.ndarray, class_names: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for g, p in zip(gt, pred, strict=True):
        cm[index[g], index[p]] += 1
    return cm


def scores(cm: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(
            precision + np.nan_to_num(recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    present = support > 0
    return {
        "tiles": int(support.sum()),
        "accuracy": float(tp.sum() / support.sum()) if support.sum() else float("nan"),
        "macro_f1": float(np.mean(f1[present])) if present.any() else float("nan"),
        "balanced_accuracy": float(np.mean(recall[present]))
        if present.any()
        else float("nan"),
        "classes_present": [c for c, p in zip(class_names, present, strict=True) if p],
        "per_class": {
            c: {
                "precision": float(precision[i]),
                "recall": float(recall[i]) if present[i] else None,
                "f1": float(f1[i]) if present[i] else None,
                "support": int(support[i]),
                "predicted": int(predicted[i]),
            }
            for i, c in enumerate(class_names)
        },
    }


def bootstrap(
    per_slide_cms: dict[str, list[np.ndarray]],
    class_names: list[str],
    n: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Slide-level bootstrap of pooled macro F1 (per model and paired differences)."""
    models = list(per_slide_cms)
    n_slides = len(next(iter(per_slide_cms.values())))
    if n_slides < 2:
        return {}
    rng = np.random.default_rng(seed)
    stacks = {m: np.stack(per_slide_cms[m]) for m in models}
    samples: dict[str, list[float]] = {m: [] for m in models}
    for _ in range(n):
        idx = rng.integers(0, n_slides, n_slides)
        for m in models:
            samples[m].append(
                scores(stacks[m][idx].sum(axis=0), class_names)["macro_f1"]
            )
    out: dict[str, Any] = {"n_bootstrap": n, "n_slides": n_slides, "models": {}}
    for m in models:
        arr = np.asarray(samples[m])
        out["models"][m] = {
            "macro_f1_ci95": [
                float(np.nanpercentile(arr, 2.5)),
                float(np.nanpercentile(arr, 97.5)),
            ]
        }
    diffs = {}
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            d = np.asarray(samples[a]) - np.asarray(samples[b])
            diffs[f"{a} - {b}"] = {
                "mean": float(np.nanmean(d)),
                "ci95": [
                    float(np.nanpercentile(d, 2.5)),
                    float(np.nanpercentile(d, 97.5)),
                ],
                "p_a_better": float(np.nanmean(d > 0)),
            }
    out["differences"] = diffs
    return out


def merge_labels(labels: np.ndarray, merge: dict[str, list[str]]) -> np.ndarray:
    lookup = {src: dst for dst, srcs in merge.items() for src in srcs}
    return np.array([lookup.get(x, x) for x in labels], dtype=object)


# ------------------------------------------------------------------ plots / report


def _confusion_png(
    cm: np.ndarray, class_names: list[str], title: str, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    support = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(support > 0, cm / support, 0.0)
    fig, ax = plt.subplots(
        figsize=(1.0 * len(class_names) + 2.5, 0.9 * len(class_names) + 2)
    )
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    ax.set_xticks(
        range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%">'


def _table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    return df.to_html(
        index=False,
        border=0,
        classes="t",
        float_format=lambda v: floatfmt.format(v),
        na_rep="-",
    )


def write_report(
    eval_dir: Path, metrics: dict[str, Any], tables: dict[str, pd.DataFrame]
) -> Path:
    e = html.escape
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Tissue classification evaluation</title>",
        "<style>body{font:14px system-ui,sans-serif;margin:24px;max-width:1300px}"
        "table.t{border-collapse:collapse;margin:8px 0 20px}"
        "table.t td,table.t th{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right}"
        "table.t th{background:#f4f4f4}td:first-child,th:first-child{text-align:left}"
        ".warn{background:#fff3cd;padding:8px 12px;border-radius:4px}"
        "h2{margin-top:32px}</style>",
        "<h1>Tissue classification — evaluation vs. ground truth</h1>",
        f"<p>Slides evaluated: {metrics['n_slides']} · models: {e(', '.join(metrics['models']))}"
        f" · label map: <code>{e(metrics['label_map'])}</code></p>",
        f"<p>GT tile labels use the training rule: ROI coverage ≥ thresholds "
        f"<code>{e(json.dumps(metrics['thresholds']))}</code>, tissue_prop_min "
        f"{metrics['tissue_prop_min']}, single-class ROI only. Accuracy is dominated by the "
        f"largest class; compare models on macro F1 and per-class F1.</p>",
    ]
    if metrics["unmapped_labels"]:
        parts.append(
            "<p class='warn'>Unmapped labels (not evaluated, add them to the label map): "
            + e(", ".join(metrics["unmapped_labels"]))
            + "</p>"
        )
    parts += ["<h2>Summary</h2>", _table(tables["summary"])]
    if not tables["bootstrap"].empty:
        parts.append("<h2>Model differences (slide bootstrap, pooled macro F1)</h2>")
        parts.append(_table(tables["bootstrap"]))
    parts += ["<h2>Per class</h2>", _table(tables["per_class"])]
    parts.append("<h2>Confusion matrices (row-normalised)</h2>")
    for png in sorted(eval_dir.glob("confusion_*.png")):
        parts.append(f"<div style='display:inline-block;width:49%'>{_img(png)}</div>")
    parts += [
        "<h2>Original GT label → predicted class</h2>",
        f"<p>Tiles whose ROI is ≥ {RAW_LABEL_MAJORITY:.0%} covered by the label; includes "
        "coarse and ignored labels.</p>",
        _table(tables["label_predictions"], "{:.2f}"),
        "<h2>Per slide</h2>",
        _table(tables["per_slide"]),
        "<h2>Label inventory</h2>",
        _table(tables["inventory"], "{:.3g}"),
        "<h2>Slides</h2>",
    ]
    for png in sorted((eval_dir / "slides").glob("*.png")):
        parts.append(
            f"<h3>{e(png.stem)}</h3><a href='slides/{e(png.name)}'>{_img(png)}</a>"
        )
    path = eval_dir / "report.html"
    path.write_text("\n".join(parts))
    return path


# ------------------------------------------------------------------ driver


@dataclass(frozen=True)
class EvalItem:
    """A slide to evaluate: its annotation, tiling and ``predict`` output directory."""

    slide: Path
    annotation: Path
    geometry: SlideGeometry
    slide_dir: Path


@dataclass
class SlideEval:
    name: str
    gt: pd.DataFrame
    raw: pd.DataFrame
    predictions: dict[str, pd.DataFrame]

    @property
    def labelled(self) -> np.ndarray:
        """Mask of tiles with a ground-truth label (the evaluated tiles)."""
        return (self.gt["reason"] == REASON_OK).to_numpy()


@dataclass
class _Rows:
    """Rows of the report tables, accumulated over variants and models."""

    summary: list[dict[str, Any]] = field(default_factory=list)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    per_slide: list[dict[str, Any]] = field(default_factory=list)
    bootstrap: list[dict[str, Any]] = field(default_factory=list)


def evaluate(
    items: list[EvalItem],
    models: list[str],
    label_map: LabelMap,
    out: Path,
    thumbnail_size: int,
) -> dict[str, Any]:
    from inference.heads import load_head

    class_names = load_head(models[0]).class_names
    thresholds, tissue_prop_min = training_label_rule()
    eval_dir = out / "evaluation"
    (eval_dir / "slides").mkdir(parents=True, exist_ok=True)

    inventory = label_inventory([item.annotation for item in items], label_map)
    inventory.to_csv(eval_dir / "label_inventory.csv", index=False)
    unmapped = inventory.loc[inventory["maps_to"] == UNMAPPED, "label"].tolist()
    if unmapped:
        print(
            f"[warn] unmapped labels (not evaluated): {', '.join(unmapped)}", flush=True
        )

    evals: list[SlideEval] = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item.slide.name}: ground truth", flush=True)
        ev = _load_slide(item, models, label_map, thresholds, tissue_prop_min)
        evals.append(ev)
        _write_slide_panels(
            ev,
            item.geometry,
            class_names,
            thumbnail_size,
            eval_dir / "slides" / f"{item.slide_dir.name}.png",
        )

    metrics: dict[str, Any] = {
        "models": models,
        "n_slides": len(evals),
        "label_map": str(label_map.path),
        "thresholds": thresholds,
        "tissue_prop_min": tissue_prop_min,
        "unmapped_labels": unmapped,
        "variants": {},
    }
    variants: dict[str, dict[str, list[str]]] = {"all classes": {}}
    variants.update(label_map.merged_variants)
    rows = _Rows()
    for variant, merge in variants.items():
        metrics["variants"][variant] = _evaluate_variant(
            variant, merge, evals, models, class_names, eval_dir, rows
        )

    label_rows = _label_prediction_rows(evals, models, label_map, class_names)
    tables = {
        "summary": pd.DataFrame(rows.summary),
        "per_class": pd.DataFrame(rows.per_class),
        "per_slide": pd.DataFrame(rows.per_slide),
        "bootstrap": pd.DataFrame(rows.bootstrap),
        "label_predictions": pd.DataFrame(label_rows).sort_values(["gt_label", "model"])
        if label_rows
        else pd.DataFrame(),
        "inventory": inventory,
    }
    tables["per_class"].to_csv(eval_dir / "per_class.csv", index=False)
    tables["per_slide"].to_csv(eval_dir / "per_slide.csv", index=False)
    tables["label_predictions"].to_csv(eval_dir / "label_predictions.csv", index=False)
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = write_report(eval_dir, metrics, tables)
    print(f"report: {report}")
    return metrics


def _load_slide(
    item: EvalItem,
    models: list[str],
    label_map: LabelMap,
    thresholds: dict[str, float],
    tissue_prop_min: float,
) -> SlideEval:
    """Ground truth (also saved as ``gt_tiles.parquet``) and cached predictions."""
    tiles = pd.read_parquet(item.slide_dir / "tiles.parquet")
    gt, raw = ground_truth_tiles(
        item.annotation,
        tiles,
        item.geometry.tile_size,
        item.geometry.downsample,
        label_map,
        thresholds,
        tissue_prop_min,
    )
    gt.to_parquet(item.slide_dir / "gt_tiles.parquet", index=False)
    reasons = gt["reason"].value_counts().to_dict()
    print(
        f"  {int((gt['reason'] == REASON_OK).sum())} labelled tiles {reasons}",
        flush=True,
    )

    predictions = {}
    for model in models:
        p = pd.read_parquet(item.slide_dir / f"predictions_{model}.parquet")
        same_tiles = (
            len(p) == len(tiles)
            and (p[["x", "y"]].to_numpy() == tiles[["x", "y"]].to_numpy()).all()
        )
        if not same_tiles:
            raise RuntimeError(
                f"{item.slide.name}: predictions_{model} do not match tiles.parquet"
            )
        predictions[model] = p
    return SlideEval(item.slide.name, gt, raw, predictions)


def _write_slide_panels(
    ev: SlideEval,
    geometry: SlideGeometry,
    class_names: list[str],
    thumbnail_size: int,
    path: Path,
) -> None:
    """Thumbnail | ground truth (evaluated tiles) | each model."""
    from inference.predict import assemble_class_map, write_panels

    panels: list[Panel] = [(ev.name, None, None)]
    ok = ev.labelled
    if ok.any():
        labelled = ev.gt.loc[ok]
        onehot = np.zeros((len(labelled), len(class_names)), dtype=np.float32)
        onehot[
            np.arange(len(labelled)), [class_names.index(c) for c in labelled["label"]]
        ] = 1
        class_map = assemble_class_map(
            onehot, labelled["x"].to_numpy(), labelled["y"].to_numpy(), geometry
        )
        panels.append(("ground truth (evaluated tiles)", *class_map.class_indices()))
    for model, p in ev.predictions.items():
        probs = p[[f"prob_{c}" for c in class_names]].to_numpy()
        class_map = assemble_class_map(
            probs, p["x"].to_numpy(), p["y"].to_numpy(), geometry
        )
        panels.append((model, *class_map.class_indices()))
    write_panels(geometry, panels, class_names, thumbnail_size, path)


def _slide_confusion(
    ev: SlideEval, model: str, merge: dict[str, list[str]], names: list[str]
) -> np.ndarray:
    ok = ev.labelled
    gt = merge_labels(ev.gt.loc[ok, "label"].to_numpy(), merge)
    pred = merge_labels(ev.predictions[model].loc[ok, "pred_name"].to_numpy(), merge)
    return confusion(gt, pred, names)


def _evaluate_variant(
    variant: str,
    merge: dict[str, list[str]],
    evals: list[SlideEval],
    models: list[str],
    class_names: list[str],
    eval_dir: Path,
    rows: _Rows,
) -> dict[str, Any]:
    """Metrics of every model for one class-merge variant; appends to ``rows``."""
    names = list(
        dict.fromkeys(merge_labels(np.array(class_names, dtype=object), merge))
    )
    result: dict[str, Any] = {"models": {}}
    per_slide_cms: dict[str, list[np.ndarray]] = {}
    for model in models:
        cms = [_slide_confusion(ev, model, merge, names) for ev in evals]
        per_slide_cms[model] = cms
        for ev, cm in zip(evals, cms, strict=True):
            s = scores(cm, names)
            rows.per_slide.append(
                {
                    "variant": variant,
                    "model": model,
                    "slide": ev.name,
                    "tiles": s["tiles"],
                    "macro_f1": s["macro_f1"],
                    "balanced_accuracy": s["balanced_accuracy"],
                }
            )

        pooled = np.sum(cms, axis=0)
        s = scores(pooled, names)
        s["confusion"] = pooled.tolist()
        s["class_names"] = names
        result["models"][model] = s
        rows.summary.append(
            {
                "variant": variant,
                "model": model,
                "tiles": s["tiles"],
                "macro_f1": s["macro_f1"],
                "balanced_accuracy": s["balanced_accuracy"],
                "accuracy": s["accuracy"],
            }
        )
        rows.per_class.extend(
            {"variant": variant, "model": model, "class": c, **v}
            for c, v in s["per_class"].items()
        )
        safe = variant.replace(" ", "_").replace("+", "_")
        _confusion_png(
            pooled,
            names,
            f"{model} — {variant}",
            eval_dir / f"confusion_{model}_{safe}.png",
        )

    boot = bootstrap(per_slide_cms, names)
    result["bootstrap"] = boot
    if boot:
        for m, v in boot["models"].items():
            rows.bootstrap.append(
                {
                    "variant": variant,
                    "comparison": m,
                    "mean": result["models"][m]["macro_f1"],
                    "ci95_low": v["macro_f1_ci95"][0],
                    "ci95_high": v["macro_f1_ci95"][1],
                    "p_a_better": np.nan,
                }
            )
        for k, v in boot["differences"].items():
            rows.bootstrap.append(
                {
                    "variant": variant,
                    "comparison": k,
                    "mean": v["mean"],
                    "ci95_low": v["ci95"][0],
                    "ci95_high": v["ci95"][1],
                    "p_a_better": v["p_a_better"],
                }
            )
    return result


def _label_prediction_rows(
    evals: list[SlideEval],
    models: list[str],
    label_map: LabelMap,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Original GT label -> predicted class distribution (all labels, incl. ignored)."""
    rows = []
    for model in models:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for ev in evals:
            for label in ev.raw.columns:
                mask = (ev.raw[label] >= RAW_LABEL_MAJORITY).to_numpy()
                counts[label].update(
                    ev.predictions[model].loc[mask, "pred_name"].tolist()
                )
        for label, ctr in counts.items():
            total = sum(ctr.values())
            if not total:
                continue
            rows.append(
                {
                    "model": model,
                    "gt_label": label,
                    "maps_to": label_map.target_of(label) or UNMAPPED,
                    "tiles": total,
                    **{c: ctr.get(c, 0) / total for c in class_names},
                }
            )
    return rows
