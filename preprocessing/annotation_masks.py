from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import pyvips
import ray
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from rationai.masks import write_big_tiff
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide
from ratiopath.parsers import Darwin7JSONParser, GeoJSONParser
from shapely import Polygon, make_valid
from shapely.geometry.base import BaseGeometry


def get_validated_polygons(
    parser: GeoJSONParser | Darwin7JSONParser, original_names: list[str]
) -> list[BaseGeometry]:
    """Extraction and topological validation of geometries from a specified parser.

    Applies a regular expression to target class names, filters empty geometries,
    and fixes potential self-intersections using `make_valid`.
    Complex geometries (MultiPolygon/GeometryCollection) are linearly
    flattened into a one-dimensional list of valid atomic polygons.

    Args:
        parser: Initialized parser instance (GeoJSON/Darwin).
        original_names: List of original class names in the dataset.

    Returns:
        List of validated and topologically correct Polygon objects.
    """
    if not original_names:
        return []

    regex_pattern = f"(?i)^({'|'.join(original_names)})$"

    if isinstance(parser, GeoJSONParser):
        raw_polygons = parser.get_polygons(meta_category_value=regex_pattern)
    else:
        raw_polygons = parser.get_polygons(name=regex_pattern)

    valid_polygons = []
    for poly in raw_polygons:
        if poly is None or poly.is_empty:
            continue

        validated = make_valid(poly) if not poly.is_valid else poly

        if isinstance(validated, Polygon):
            valid_polygons.append(validated)
        elif hasattr(validated, "geoms"):
            for geom in validated.geoms:
                if isinstance(geom, Polygon):
                    valid_polygons.append(geom)

    return valid_polygons


def extract_mapped_regions(
    annotation_path: Path, class_mapping: dict[str, list[str]]
) -> dict[str, list[BaseGeometry]]:
    """Identification of annotation format, file loading, and selection of target classes.

    Utilizes a mapping dictionary to extract and aggregate original tissue names
    into unified semantic classes required by the training configuration.

    Args:
        annotation_path: Physical path to the annotation file.
        class_mapping: Configuration mapping of target classes to original strings.

    Returns:
        Dictionary mapping target class names to lists of valid geometries.
    """
    path_str = str(annotation_path)

    if path_str.endswith((".geo.json", ".geojson")):
        parser = GeoJSONParser(annotation_path)
    elif path_str.endswith(".json"):
        parser = Darwin7JSONParser(annotation_path)
    else:
        return {}

    extracted_regions = {}
    for target_class, original_names in class_mapping.items():
        valid_polygons = get_validated_polygons(parser, original_names)
        extracted_regions[target_class] = valid_polygons

    return extracted_regions


@ray.remote(num_cpus=1, memory=(3 * 1024**3))
def process_slide(
    item: tuple[str, str],
    target_mpp: float,
    output_dir: str,
    class_mapping: dict[str, list[str]],
    class_indices: dict[str, int],
    bad_slides: list[str],
) -> None:
    slide_path = Path(item[0])
    annotation_path = Path(item[1])

    extracted_regions = extract_mapped_regions(annotation_path, class_mapping)
    total_geoms = sum(len(geoms) for geoms in extracted_regions.values())

    if total_geoms == 0:
        missing_marker = Path(output_dir, f"{slide_path.name}.missing")
        with open(missing_marker, "w") as f:
            f.write(str(slide_path))
        return

    with OpenSlide(slide_path) as slide:
        if slide_path.name in bad_slides:
            target_mpp *= 2
        level = slide.closest_level(target_mpp)

        mpp_x, mpp_y = slide.slide_resolution(level)
        base_width, base_height = slide.level_dimensions[0]
        target_width, target_height = slide.level_dimensions[level]

        scale_x = target_width / base_width
        scale_y = target_height / base_height

        mask_size = (target_width, target_height)

    mask = Image.new("L", size=mask_size, color=255)
    mask_drawer = ImageDraw.Draw(mask)

    for target_class, geometries in extracted_regions.items():
        class_index = class_indices.get(target_class)
        if class_index is None:
            continue

        for polygon in geometries:
            if polygon.is_empty:
                continue

            exterior_coords = [
                (x * scale_x, y * scale_y) for x, y in polygon.exterior.coords
            ]
            mask_drawer.polygon(xy=exterior_coords, fill=class_index)

            for interior in polygon.interiors:
                interior_coords = [
                    (x * scale_x, y * scale_y) for x, y in interior.coords
                ]
                mask_drawer.polygon(xy=interior_coords, fill=255)

    del mask_drawer

    width, height = mask.size
    mask_bytes = mask.tobytes()

    del mask

    output_path = Path(output_dir, slide_path.with_suffix(".tiff").name)

    write_big_tiff(
        image=pyvips.Image.new_from_memory(
            data=mask_bytes, width=width, height=height, bands=1, format="uchar"
        ),
        path=output_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )


@with_cli_args(["+preprocessing=annotation_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    artifact_uri = f"runs:/{config.dataset.mlflow_artifacts.mapping_run_id}/{config.dataset.mlflow_artifacts.mapping_filename}"
    slides = pd.read_csv(download_artifacts(artifact_uri))

    items_to_process = list(
        zip(slides["wsi_path"], slides["annotation_path"], strict=True)
    )

    with TemporaryDirectory(dir=Path.cwd()) as output_dir:
        process_items(
            items_to_process,
            process_item=process_slide,
            fn_kwargs={
                "target_mpp": config.mpp,
                "output_dir": output_dir,
                "class_mapping": OmegaConf.to_container(
                    config.class_mapping, resolve=True
                ),
                "class_indices": OmegaConf.to_container(
                    config.class_indices, resolve=True
                ),
                "bad_slides": config.dataset.exclusions.get("bad_slides", []),
            },
            max_concurrent=config.max_concurrent,
        )

        missing_slides = []
        for missing_marker in Path(output_dir).glob("*.missing"):
            with open(missing_marker) as f:
                missing_slides.append(f.read().strip())
            missing_marker.unlink()

        csv_path = Path(output_dir, "missing_annotations.csv")
        pd.DataFrame({"wsi_path": missing_slides}).to_csv(csv_path, index=False)

        logger.log_artifacts(
            local_dir=output_dir,
            artifact_path=config.mlflow_artifact_path,
        )


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
