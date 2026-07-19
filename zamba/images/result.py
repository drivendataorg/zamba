import json
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any, Union

from loguru import logger
import pandas as pd
from PIL import Image
from pydantic import BaseModel


class ImageDetectionResult(BaseModel):
    category: str
    conf: float
    bbox: list[
        float
    ]  # MegaDetector bbox is relative measures from top left [x1, y1, width, height]
    classifications: list[list]


class ImageResult(BaseModel):
    file: str
    detections: list[ImageDetectionResult]


class ClassificationResultMegadetectorFormat(BaseModel):
    info: dict[str, Any]
    detection_categories: dict[str, str]
    classification_categories: dict[str, str]
    images: list[ImageResult]


def get_top_3_classifications(row, species) -> list:
    row_to_compare = row[species]
    row_to_compare = pd.to_numeric(row_to_compare, errors="coerce")
    return [
        [str(species.index(col)), round(float(value), 4)]
        for col, value in row_to_compare.nlargest(3).items()
    ]


def results_to_megadetector_format(
    df: pd.DataFrame, species: list
) -> ClassificationResultMegadetectorFormat:
    classification_categories = {str(idx): specie for idx, specie in enumerate(species)}
    info = {}
    detection_categories = {"1": "animal", "2": "person", "3": "vehicle"}

    image_results = {}
    for _, row in df.iterrows():
        filepath = row["filepath"]

        with Image.open(filepath) as img:
            width, height = img.size

        detection_category = row["detection_category"]
        if image_results.get(filepath) is None:
            image_results[filepath] = ImageResult(file=filepath, detections=[])

        detection_classifications = []
        if detection_category == "1":
            detection_classifications = get_top_3_classifications(row, species)

        image_results[filepath].detections.append(
            ImageDetectionResult(
                category=detection_category,
                conf=round(float(row["detection_conf"]), 4),
                bbox=[
                    round(row["x1"] / width, 4),
                    round(row["y1"] / height, 4),
                    round((row["x2"] - row["x1"]) / width, 4),
                    round((row["y2"] - row["y1"]) / height, 4),
                ],  # MegaDetector bbox is relative measures from top left [x1, y1, width, height]
                classifications=detection_classifications,
            )
        )

    return ClassificationResultMegadetectorFormat(
        info=info,
        detection_categories=detection_categories,
        classification_categories=classification_categories,
        images=list(image_results.values()),
    )


# Camtrap DP profile targeted by the exporter (same version zamba ingests on the input side)
CAMTRAP_DP_PROFILE = (
    "https://raw.githubusercontent.com/tdwg/camtrap-dp/1.0.2/camtrap-dp-profile.json"
)

# MegaDetector detection categories -> Camtrap DP observationType controlled vocabulary
_DETECTION_TYPE = {"1": "animal", "2": "human", "3": "vehicle"}

_CAMTRAP_TABULAR_RESOURCE = {
    "profile": "tabular-data-resource",
    "format": "csv",
    "mediatype": "text/csv",
    "encoding": "utf-8",
}


def results_to_camtrap_dp(
    df: pd.DataFrame,
    species: list,
    output_dir: Union[str, Path],
    package_name: str = "zamba-image-predictions",
    deployment_id: str = "zamba-deployment",
) -> Path:
    """Write zamba image predictions as a (partial) Camtrap DP package.

    Emits a package directory with ``datapackage.json``, ``deployments.csv``,
    ``media.csv``, and ``observations.csv``. Only fields derivable from inference are
    populated: media file references, per-detection ``observationType`` /
    ``scientificName`` / ``classificationProbability``, and relative bounding boxes.

    Deployment, timestamp, and location fields are required by the Camtrap DP spec but
    are not knowable from images alone, so they are written as a single placeholder
    deployment with empty values. The package is meant for downstream ingestion (and
    round-trips back through ``camtrap_dp_to_df``), not strict spec validation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # one media row per unique image; deterministic ids by first appearance so a given
    # predictions dataframe always yields the same package
    media_ids: dict = {}
    dims: dict = {}
    media_rows = []
    for filepath in df["filepath"]:
        if filepath in media_ids:
            continue
        media_id = f"m{len(media_ids):06d}"
        media_ids[filepath] = media_id
        with Image.open(filepath) as img:
            dims[filepath] = img.size  # (width, height)
        file_name = PurePosixPath(str(filepath).replace("\\", "/")).name
        media_rows.append(
            {
                "mediaID": media_id,
                "deploymentID": deployment_id,
                "timestamp": "",
                "filePath": filepath,
                "filePublic": "false",
                "fileName": file_name,
                "fileMediatype": mimetypes.guess_type(file_name)[0] or "",
            }
        )

    obs_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        filepath = row["filepath"]
        width, height = dims[filepath]
        observation_type = _DETECTION_TYPE.get(str(row["detection_category"]), "unknown")

        scientific_name = ""
        probability = ""
        if observation_type == "animal" and species:
            probs = pd.to_numeric(row[species], errors="coerce")
            if probs.notna().any():
                top = probs.idxmax()
                scientific_name = top
                probability = float(probs[top])

        bbox = {"bboxX": "", "bboxY": "", "bboxWidth": "", "bboxHeight": ""}
        if width and height and all(pd.notna(row.get(c)) for c in ("x1", "y1", "x2", "y2")):
            bbox = {
                "bboxX": row["x1"] / width,
                "bboxY": row["y1"] / height,
                "bboxWidth": (row["x2"] - row["x1"]) / width,
                "bboxHeight": (row["y2"] - row["y1"]) / height,
            }

        media_id = media_ids[filepath]
        obs_rows.append(
            {
                "observationID": f"o{i:06d}",
                "deploymentID": deployment_id,
                "mediaID": media_id,
                "eventID": media_id,
                "observationLevel": "media",
                "observationType": observation_type,
                "scientificName": scientific_name,
                "count": 1,
                "classificationMethod": "machine",
                "classifiedBy": "zamba",
                "classificationProbability": probability,
                **bbox,
            }
        )

    deployments_df = pd.DataFrame(
        [
            {
                "deploymentID": deployment_id,
                "deploymentStart": "",
                "deploymentEnd": "",
                "latitude": "",
                "longitude": "",
            }
        ]
    )
    deployments_df.to_csv(output_dir / "deployments.csv", index=False)
    pd.DataFrame(media_rows).to_csv(output_dir / "media.csv", index=False)
    pd.DataFrame(obs_rows).to_csv(output_dir / "observations.csv", index=False)

    datapackage = {
        "name": package_name,
        "profile": CAMTRAP_DP_PROFILE,
        "resources": [
            {"name": "deployments", "path": "deployments.csv", **_CAMTRAP_TABULAR_RESOURCE},
            {"name": "media", "path": "media.csv", **_CAMTRAP_TABULAR_RESOURCE},
            {"name": "observations", "path": "observations.csv", **_CAMTRAP_TABULAR_RESOURCE},
        ],
    }
    with open(output_dir / "datapackage.json", "w") as f:
        json.dump(datapackage, f, indent=2)

    logger.warning(
        "Wrote a partial Camtrap DP package: deployment, timestamp, and location fields are "
        "placeholders because they cannot be inferred from images alone."
    )
    return output_dir
