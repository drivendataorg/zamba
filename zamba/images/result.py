from typing import Any

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
    return [[species.index(col), value] for col, value in row_to_compare.nlargest(3).items()]


def results_to_megadetector_format(
    df: pd.DataFrame, species: list
) -> ClassificationResultMegadetectorFormat:
    classification_categories = {idx: specie for idx, specie in enumerate(species)}
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
                conf=row["detection_conf"],
                bbox=[
                    row["x1"] / width,
                    row["y1"] / height,
                    (row["x2"] - row["x1"]) / width,
                    (row["y2"] - row["y1"]) / height,
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
