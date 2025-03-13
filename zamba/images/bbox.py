from enum import StrEnum
from pathlib import Path
from typing import Iterable, Tuple, Union

from loguru import logger
import pandas as pd
from PIL import Image


class BboxInputFormat(StrEnum):
    COCO = "coco"
    MEGADETECTOR = "megadetector"


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
    else:
        raise ValueError(
            f"Invalid bbox_format: {bbox_format}; expected one of {BboxInputFormat.__members__.keys()}"
        )


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
