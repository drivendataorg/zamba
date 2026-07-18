from enum import StrEnum
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Iterable, Optional, Tuple, Union
from zipfile import BadZipFile, ZipFile

from loguru import logger
import pandas as pd
from PIL import Image

from zamba.settings import IMAGE_SUFFIXES


class BboxInputFormat(StrEnum):
    COCO = "coco"
    MEGADETECTOR = "megadetector"
    CAMTRAP_DP = "camtrap_dp"


class BboxLayout(StrEnum):
    XYXY = "xyxy"
    XYWH = "xywh"


def bbox_json_to_df(
    bbox_json: dict, bbox_format: BboxInputFormat = BboxInputFormat.COCO
) -> pd.DataFrame:
    if bbox_format == BboxInputFormat.COCO:
        logger.info("Processing bounding box labels from coco format")

        images = pd.DataFrame(bbox_json["images"])[["id", "file_name"]]
        images["filepath"] = images["file_name"]
        images = images.drop("file_name", axis=1)

        annotations = pd.DataFrame(bbox_json["annotations"])[
            ["id", "image_id", "category_id", "bbox"]
        ]

        annotations = annotations[~annotations["bbox"].isna()]

        annotations[["x", "y", "width", "height"]] = pd.DataFrame(annotations["bbox"].tolist())

        categories = pd.DataFrame(bbox_json["categories"])[["id", "name"]]

        categories.rename(columns={"name": "label"}, inplace=True)

        result = pd.merge(
            images,
            annotations,
            left_on="id",
            right_on="image_id",
            how="right",
            suffixes=("_image", "_annotation"),
        )

        result = pd.merge(
            result,
            categories,
            left_on="category_id",
            right_on="id",
            how="left",
            suffixes=("_image", "_category"),
        )

        if all(column in result.columns for column in ["x", "y", "width", "height"]):
            result.rename({"x": "x1", "y": "y1"}, axis=1, inplace=True)
            result["x2"] = result["x1"] + result["width"]
            result["y2"] = result["y1"] + result["height"]

        return result
    elif bbox_format == BboxInputFormat.MEGADETECTOR:
        logger.info("Processing bounding box labels from megadetector format")

        detection_categories = bbox_json["detection_categories"]
        classification_categories = bbox_json["classification_categories"]

        out_rows = []
        for img_ix, image in enumerate(bbox_json["images"]):
            if image.get("detections") is None:
                continue
            for d_ix, detection in enumerate(image.get("detections", [])):
                # skip if bbox is 0 height or width
                if detection["bbox"][2] <= 0 or detection["bbox"][3] <= 0:
                    continue

                detection_classification = detection.get("classifications")
                if detection_classification is None:
                    continue

                out_rows.append(
                    {
                        "image_id": img_ix,
                        "detection_id": d_ix,
                        "id": f"{img_ix}_{d_ix}",
                        "filepath": image["file"],
                        "label": classification_categories[detection_classification[0][0]],
                        "label_id": detection_classification[0][0],
                        "label_confidence": detection_classification[0][1],
                        "detection_label": detection_categories[detection["category"]],
                        "detection_label_id": detection["category"],
                        "detection_confidence": detection["conf"],
                        "x1": detection["bbox"][0],
                        "y1": detection["bbox"][1],
                        "x2": detection["bbox"][0] + detection["bbox"][2],
                        "y2": detection["bbox"][1] + detection["bbox"][3],
                    }
                )

        return pd.DataFrame(out_rows).set_index("id")
    elif bbox_format == BboxInputFormat.CAMTRAP_DP:
        raise ValueError(
            "Camtrap DP labels must be loaded from a data package path via camtrap_dp_to_df(), "
            "not from an in-memory JSON object."
        )
    else:
        raise ValueError(
            f"Invalid bbox_format: {bbox_format}; expected one of {BboxInputFormat.__members__.keys()}"
        )


def _is_remote_path(path: str) -> bool:
    return re.match(r"[a-zA-Z][a-zA-Z0-9+.\-]*://", str(path)) is not None


def _camtrap_media_filepath(row: pd.Series) -> Optional[str]:
    """Prefer a local filePath; fall back to fileName when filePath is remote or missing."""
    file_path = row.get("filePath")
    if pd.notna(file_path) and str(file_path).strip() and not _is_remote_path(file_path):
        return str(file_path)

    file_name = row.get("fileName")
    if pd.notna(file_name) and str(file_name).strip():
        return str(file_name)

    return None


def _camtrap_label(row: pd.Series) -> Optional[str]:
    scientific_name = row.get("scientificName")
    if pd.notna(scientific_name) and str(scientific_name).strip():
        return str(scientific_name).strip()

    observation_type = row.get("observationType")
    if pd.notna(observation_type) and str(observation_type).strip():
        return str(observation_type).strip()

    return None


def _read_camtrap_table(
    resource_name: str,
    resources_by_name: dict,
    package_dir: Optional[Path],
    zip_file: Optional[ZipFile] = None,
    zip_prefix: str = "",
) -> pd.DataFrame:
    if resource_name not in resources_by_name:
        raise ValueError(f"Camtrap DP package is missing required resource '{resource_name}'.")

    resource = resources_by_name[resource_name]
    path = resource.get("path")
    if not path:
        raise ValueError(
            f"Camtrap DP resource '{resource_name}' must include a 'path' to a tabular file."
        )
    if isinstance(path, list):
        if len(path) != 1:
            raise ValueError(
                f"Camtrap DP resource '{resource_name}' has multiple paths; "
                "only a single path is supported."
            )
        path = path[0]

    # the Data Package spec allows URL resource paths (e.g. GBIF-published packages)
    if _is_remote_path(path):
        return pd.read_csv(path)

    if zip_file is not None:
        member = posixpath.normpath(posixpath.join(zip_prefix, path))
        if member not in zip_file.namelist():
            raise ValueError(
                f"Camtrap DP resource '{resource_name}' points to missing archive member: {member}"
            )
        with zip_file.open(member) as f:
            return pd.read_csv(f)

    table_path = package_dir / path
    if not table_path.exists():
        raise ValueError(
            f"Camtrap DP resource '{resource_name}' points to missing file: {table_path}"
        )
    return pd.read_csv(table_path)


def _camtrap_resources_by_name(datapackage: dict) -> dict:
    return {r["name"]: r for r in datapackage.get("resources", []) if "name" in r}


def _read_camtrap_tables_from_zip(package_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        zip_file = ZipFile(package_path)
    except BadZipFile:
        raise ValueError(f"Camtrap DP labels are not a valid zip archive: {package_path}")

    with zip_file:
        candidates = [
            name for name in zip_file.namelist() if PurePosixPath(name).name == "datapackage.json"
        ]
        if not candidates:
            raise ValueError(f"Zip archive does not contain a datapackage.json: {package_path}")
        # tolerate archives where the package sits under a root folder (e.g. `zip -r pkg.zip pkg/`)
        datapackage_name = min(candidates, key=lambda name: len(PurePosixPath(name).parts))
        prefix = str(PurePosixPath(datapackage_name).parent)
        zip_prefix = "" if prefix == "." else prefix
        with zip_file.open(datapackage_name) as f:
            resources_by_name = _camtrap_resources_by_name(json.load(f))
        return (
            _read_camtrap_table("media", resources_by_name, None, zip_file, zip_prefix),
            _read_camtrap_table("observations", resources_by_name, None, zip_file, zip_prefix),
        )


def camtrap_dp_to_df(package_path: Union[str, Path]) -> pd.DataFrame:
    """Load Camtrap DP observations into a labels dataframe.

    Accepts a path to:
    - a Camtrap DP directory containing ``datapackage.json``
    - a ``datapackage.json`` file
    - a ``.zip`` archive containing a Camtrap DP package

    Returns rows with ``filepath``, ``label``, and ``site`` (from ``deploymentID``)
    when available. If any observations carry usable bounding boxes, only those rows
    are returned with relative ``x1/y1/x2/y2`` columns for crop-based training. If the
    package has no usable boxes, all labeled image observations are returned without
    box columns, so training runs on whole images (downstream can still run
    MegaDetector to generate crops when ``crop_images`` is set).
    """
    package_path = Path(package_path)
    logger.info(f"Processing bounding box labels from Camtrap DP package: {package_path}")

    datapackage_path: Optional[Path] = None
    if package_path.is_dir():
        datapackage_path = package_path / "datapackage.json"
        if not datapackage_path.exists():
            raise ValueError(f"Camtrap DP directory is missing datapackage.json: {package_path}")
    elif package_path.name == "datapackage.json":
        datapackage_path = package_path
    elif package_path.suffix.lower() != ".zip":
        raise ValueError(
            "Camtrap DP labels must be a directory containing datapackage.json, "
            "a datapackage.json file, or a .zip archive."
        )

    if datapackage_path is None:
        media, observations = _read_camtrap_tables_from_zip(package_path)
    else:
        package_dir = datapackage_path.parent
        with open(datapackage_path, "r") as f:
            resources_by_name = _camtrap_resources_by_name(json.load(f))
        media = _read_camtrap_table("media", resources_by_name, package_dir)
        observations = _read_camtrap_table("observations", resources_by_name, package_dir)

    if "mediaID" not in observations.columns:
        raise ValueError("Camtrap DP observations table is missing required column 'mediaID'.")

    if "mediaID" not in media.columns:
        raise ValueError("Camtrap DP media table is missing required column 'mediaID'.")

    if "fileMediatype" in media.columns:
        media = media[media["fileMediatype"].fillna("").astype(str).str.startswith("image/")]
    elif "fileName" in media.columns or "filePath" in media.columns:
        # no mediatype column: fall back to file extensions so videos don't reach image loading
        names = media["fileName"] if "fileName" in media.columns else media["filePath"]
        suffixes = (
            names.fillna("").astype(str).str.lower().str.extract(r"(\.[^./\\]+)$", expand=False)
        )
        media = media[suffixes.isin([s.lower() for s in IMAGE_SUFFIXES])]

    observations = observations.dropna(subset=["mediaID"])

    # Bounding boxes are optional in Camtrap DP: the bbox columns exist in the schema but are
    # empty for most packages. Branch on whether any row carries a *usable* box, not on column
    # presence, so a schema-complete-but-boxless export falls back to whole-image labels.
    bbox_cols = ["bboxX", "bboxY", "bboxWidth", "bboxHeight"]
    if all(col in observations.columns for col in bbox_cols):
        # coerce malformed bbox cells (e.g. locale decimals) to NaN so invalid boxes drop
        # instead of making the column object dtype and crashing the comparisons below
        observations[bbox_cols] = observations[bbox_cols].apply(pd.to_numeric, errors="coerce")
        valid_box = (
            observations[bbox_cols].notna().all(axis=1)
            & (observations["bboxWidth"] > 0)
            & (observations["bboxHeight"] > 0)
        )
    else:
        valid_box = pd.Series(False, index=observations.index)

    has_boxes = bool(valid_box.any())
    if has_boxes:
        # some observations carry usable boxes: train on crops and drop boxless rows, since
        # downstream cropping ignores boxless rows whenever any boxed rows are present
        observations = observations[valid_box]

    result = observations.merge(media, on="mediaID", how="inner", suffixes=("", "_media"))
    if result.empty:
        raise ValueError(
            "No Camtrap DP observations matched image media. "
            "Ensure observations reference image mediaIDs present in the media table."
        )

    result["filepath"] = result.apply(_camtrap_media_filepath, axis=1)
    result["label"] = result.apply(_camtrap_label, axis=1)
    result = result.dropna(subset=["filepath", "label"])

    if result.empty:
        raise ValueError(
            "No usable Camtrap DP rows after resolving filepaths and labels. "
            "Provide local filePath/fileName values and scientificName or observationType."
        )

    keep_cols = ["id", "filepath", "label"]

    if has_boxes:
        # Camtrap DP boxes are relative; clamp to [0, 1] because downstream absolute_bbox
        # treats any coordinate > 1 as absolute pixels, silently corrupting the crop
        result["x1"] = result["bboxX"].astype(float).clip(0.0, 1.0)
        result["y1"] = result["bboxY"].astype(float).clip(0.0, 1.0)
        result["x2"] = (result["x1"] + result["bboxWidth"].astype(float)).clip(0.0, 1.0)
        result["y2"] = (result["y1"] + result["bboxHeight"].astype(float)).clip(0.0, 1.0)
        keep_cols += ["x1", "y1", "x2", "y2"]

    if "deploymentID" in result.columns:
        result["site"] = result["deploymentID"]

    if "observationID" in result.columns:
        result["id"] = result["observationID"]
    else:
        result["id"] = result["mediaID"].astype(str)

    if "site" in result.columns:
        keep_cols.append("site")

    mode = "with bounding boxes" if has_boxes else "without bounding boxes (whole-image labels)"
    logger.info(f"Loaded {len(result)} Camtrap DP observations {mode}")
    return result[keep_cols].reset_index(drop=True)


def absolute_bbox(
    img: Image,
    bbox: Union[Iterable[float], Iterable[int]],
    bbox_layout: BboxLayout = BboxLayout.XYXY,
) -> Tuple[int, int, int, int]:
    """Ensures a bbox is in absolute pixel units. Turns relative to absolute.
    This assumes the origin is top-left.

    Returns:
        tuple[int, int, int, int]: x1 (left), y1 (top), x2 (right), y2 (bottom)
    """
    if all([n <= 1 for n in bbox]):
        shape = img.size
        if bbox_layout == BboxLayout.XYWH:
            x1, y1, width, height = [
                int(bbox[0] * shape[0]),
                int(bbox[1] * shape[1]),
                int(bbox[2] * shape[0]),
                int(bbox[3] * shape[1]),
            ]
            x2, y2 = x1 + width, y1 + height
        elif bbox_layout == BboxLayout.XYXY:
            x1, y1, x2, y2 = [
                int(bbox[0] * shape[0]),
                int(bbox[1] * shape[1]),
                int(bbox[2] * shape[0]),
                int(bbox[3] * shape[1]),
            ]
        else:
            raise ValueError(
                f"Invalid bbox_format: {bbox_layout}; expected one of {BboxInputFormat.__members__.keys()}"
            )
        return x1, y1, x2, y2
    else:
        return bbox


def crop_to_bounding_box(row, cache_dir, image_dir: Path | None = None) -> Image:
    _, row = row  # Unpack the index and the row
    filepath = row["filepath"] if image_dir is None else image_dir / row["filepath"]
    image = load_image(filepath)
    bbox = absolute_bbox(
        image, [row["x1"], row["y1"], row["x2"], row["y2"]], bbox_layout=BboxLayout.XYXY
    )
    row["x1"], row["y1"], row["x2"], row["y2"] = bbox
    cache_path = cache_dir / get_cache_filename(row["filepath"], bbox)

    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cropped_image = image.crop(bbox)
        with open(cache_path, "wb") as f:
            cropped_image.save(f)

    row["cached_bbox"] = cache_path.resolve().absolute()
    return row


def load_image(img_path) -> Image:
    with open(img_path, "rb") as fp:
        image = Image.open(fp)
        image = image.convert("RGB")
        return image


def get_cache_filename(filepath: str, bbox) -> Path:
    path = Path(filepath)
    if path.is_absolute():
        path = path.relative_to("/")
    return path.parent / f"{path.stem}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}{path.suffix}"
