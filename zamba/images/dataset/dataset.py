import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import typer
import yaml
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger
from PIL import Image
from tqdm.contrib.concurrent import process_map

app = typer.Typer()


def _remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def load_image_from_bucket(source: str) -> Image.Image:
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    source = _remove_prefix(source, "s3://")
    bucket_name, object_key = source.split("/", 1)
    bucket = s3.Bucket(bucket_name)
    obj = bucket.Object(object_key)
    response = obj.get()
    file_stream = response["Body"]
    img = Image.open(file_stream)
    return img


def split(value, splits) -> str:
    if str(value) in splits.get("train", []):
        return "train"
    elif str(value) in splits.get("test", []):
        return "test"
    elif str(value) in splits.get("val", []):
        return "val"
    else:
        return "other"


def prepare_dataset(
    annotations: Dict[str, Any],
    splits: Dict[str, Any],
    storage_path: str,
    categories_name_mapper: Dict[str, str],
    name: str,
):
    if not storage_path.endswith("/"):
        storage_path += "/"

    if "splits" in splits:
        splits = splits["splits"]

    df_annotations = pd.DataFrame(annotations["annotations"])
    df_images = pd.DataFrame(annotations["images"])

    df = pd.merge(
        df_annotations,
        df_images,
        left_on="image_id",
        right_on="id",
        how="inner",
        suffixes=("_annotations", "_images"),
    )
    # remove rows with non-existing files
    if annotations.get("info", {}).get("contributor", "").lower() == "wcs":
        df = df[df["file_name"].apply(lambda x: x[:7] != "humans/")]

    df = df[df["bbox"].notnull()]

    df["split"] = df["location"].apply(lambda x: split(x, splits))
    df["source"] = storage_path + df["file_name"]  # storage path to file

    df["id"] = name + df["id_annotations"]  # unique ID in merged dataset

    # map categories and drop categories non-included in config
    categories_map = {x["id"]: x["name"] for x in annotations["categories"]}
    df["category_name"] = df["category_id"].apply(lambda x: categories_map[x])
    df["category"] = df["category_name"].apply(lambda x: categories_name_mapper.get(x, None))
    df = df[df["category"].notnull()]

    return df


def merge_datasets(
    dataset_config: Path,
) -> pd.DataFrame:
    with open(dataset_config, "r") as f:
        config = yaml.safe_load(f)

    categories_name_mapper = config["categories"]["map"]
    annotations_list = []
    splits_list = []
    storages = []
    names = []

    for dataset in config["datasets"]:
        with open(dataset["annotations"], "r") as f:
            annotations_list.append(json.load(f))
        with open(dataset["splits"], "r") as f:
            splits_list.append(json.load(f))
        storages.append(dataset["storage"])
        names.append(dataset["name"])

    data_frames = []
    for annotations, splits, storage_path, name in zip(
        annotations_list, splits_list, storages, names
    ):
        data_frames.append(
            prepare_dataset(annotations, splits, storage_path, categories_name_mapper, name)
        )

    df = pd.concat(data_frames, ignore_index=True)

    df["filepath"] = df["id"] + "." + df["source"].str.split(".").str[-1]
    df["label"] = df["category_name"]

    return df


def crop_image(img: Image, bbox: List) -> Image.Image:
    """
    Crop image using annotation bbox.

    Args:
        img (Image): Original image
        bbox (list): A list containing four elements representing the bounding box: [x_min, y_min, width, height].
    Returns:
        Cropped Image.

    Notes:
        x_min (float): The x-coordinate of the top-left corner of the bounding box.
        y_min (float): The y-coordinate of the top-left corner of the bounding box.
        width (float): The width of the bounding box.
        height (float): The height of the bounding box.
    """

    bbox = [int(coord) for coord in bbox]
    return img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))


def process_files(data_dir: Path, annot: Dict, download: bool = True) -> Optional[Dict]:
    image_path = data_dir / annot["filepath"]

    if os.path.exists(image_path):
        return annot

    if not download:
        return None

    try:
        img = load_image_from_bucket(annot["source"])
        img = crop_image(img, annot["bbox"])

        with image_path.open("w") as output_file:
            img.save(output_file)

    except ClientError:
        logger.warning(f"No such key {annot['source']}!")
        return None
    except (Exception,):
        logger.warning(f"Exception for {annot['source']}!")
        return None

    return annot


@app.command(help="Load and preprocess datasets from config file")
def load_data(
    data_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, help="Path to the data directory."
    ),
    dataset_config: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, help="Path to the config file."
    ),
    result_path: Path = typer.Option(
        ..., exists=False, file_okay=True, dir_okay=False, help="Path to the result file."
    ),
    download_data: bool = typer.Option(
        False, "--download", help="Download and process dataset images."
    ),
) -> None:
    """
    Preprocesses datasets from the specified images directory.

    Args:
        data_dir (Path): Path to the data directory.
        dataset_config (Path): Path to the dataset config .yaml file.
        result_path (Path): Path to the result file (.csv) with 'filepath', 'label' and 'split' columns.
        download_data (bool): Download and process dataset images. Boolean flag, default is False
    """
    annotations = merge_datasets(dataset_config)

    if download_data:
        data = process_map(
            partial(process_files, data_dir, download=False),
            annotations.to_dict(orient="records"),
            num_workers=os.cpu_count(),
            chunksize=int(math.sqrt(len(annotations))),
        )
        data = [x for x in data if x is not None]
        annotations = pd.DataFrame(data)
    annotations[["filepath", "label", "split"]].to_csv(result_path)


if __name__ == "__main__":
    app()
