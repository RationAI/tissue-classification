import glob
import tempfile
from pathlib import Path
from typing import cast

import hydra
import openslide
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def scan_directory(root_pattern: str) -> list[dict[str, str]]:
    """Scan files based on config pattern using rglob."""
    found_files = []
    for path_str in glob.glob(root_pattern, recursive=True):
        file_path = Path(path_str)
        if not file_path.is_file():
            continue
        found_files.append({"id_stem": get_stem(file_path), "path": str(file_path)})

    return found_files


def find_files(sources: DictConfig) -> pd.DataFrame:
    """Iterates over sources config and returns a DataFrame."""
    all_records = []
    for source_id, pattern in sources.items():
        records = scan_directory(str(pattern))
        for r in records:
            r["source"] = str(source_id)
        all_records.extend(records)

    return pd.DataFrame(all_records)


def get_stem(path: Path) -> str:
    """Extracts the base name of a file."""
    name = path.name
    if name.lower().endswith(".geo.json"):
        return name[:-9]
    return path.stem


def assign_organs(df: pd.DataFrame, mug_config: DictConfig) -> pd.Series:
    """Extracts organ type based on source and path."""
    mug_map = cast("dict[str, str]", OmegaConf.to_container(mug_config, resolve=True))
    src = df["source"].str.upper()

    organs = pd.Series("Unknown", index=df.index)
    organs[src.eq("MUG")] = (
        df.loc[src.eq("MUG"), "id_stem"].map(mug_map).fillna("MUG_Unknown")
    )
    parents_dir = df["wsi_path"].str.rsplit("/", n=2).str[-2].str.lower()

    organs[src.isin(["MOU", "TCGA"])] = parents_dir[src.isin(["MOU", "TCGA"])]
    organs[src.eq("FNBRNO")] = "pancreas"

    return organs


def extract_wsi_metadata(wsi_path: str) -> dict[str, float | int | str]:
    """Extracts key metadata (MPP, Magnification, Vendor) from a WSI file."""
    with openslide.OpenSlide(wsi_path) as slide:
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0)
        mag = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 0)
        vendor = slide.properties.get(openslide.PROPERTY_NAME_VENDOR, "N/A")

    return {
        "mpp_x": round(float(mpp_x), 4),
        "mpp_y": round(float(mpp_y), 4),
        "magnification": int(float(mag)),
        "vendor": str(vendor),
    }


@with_cli_args(["+preprocessing=wsi_mapping"])
@hydra.main(
    config_path="../configs",
    config_name="preprocessing",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    df_wsi = find_files(config.dataset.wsi_sources)
    df_wsi = df_wsi.rename(columns={"path": "wsi_path"})

    df_ann = find_files(config.dataset.annotation_sources)
    df_ann = df_ann.rename(columns={"path": "annotation_path"})
    df_ann = df_ann[["id_stem", "annotation_path"]]

    df_merged = pd.merge(df_wsi, df_ann, on="id_stem", how="inner")
    df_merged["organ"] = assign_organs(df_merged, config.dataset.mug_organ_map)

    classes = sorted(df_merged["organ"].unique())
    class_map = {name: idx for idx, name in enumerate(classes)}
    df_merged["label"] = df_merged["organ"].map(class_map)

    df_meta = df_merged["wsi_path"].apply(lambda p: pd.Series(extract_wsi_metadata(p)))

    df_final = pd.concat([df_merged, df_meta], axis=1)
    df_final = df_final.sort_values(by=["source", "id_stem"])
    df_final = df_final[
        [
            "id_stem",
            "source",
            "organ",
            "label",
            "wsi_path",
            "annotation_path",
            "mpp_x",
            "mpp_y",
            "magnification",
            "vendor",
        ]
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        csv_file = tmp_path / str(config.output_csv_name)
        df_final.to_csv(csv_file, index=False, encoding="utf-8")
        logger.log_artifact(str(csv_file))

        map_file = tmp_path / str(config.output_json_name)
        pd.Series(class_map).to_json(map_file, indent=4)
        logger.log_artifact(str(map_file))


if __name__ == "__main__":
    main()
