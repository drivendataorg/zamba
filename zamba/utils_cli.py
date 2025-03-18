from itertools import chain
from pathlib import Path

import json
from loguru import logger
from tqdm import tqdm
import typer
import yaml
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from zamba import MODELS_DIRECTORY
from zamba.images.bbox import bbox_json_to_df, BboxInputFormat
from zamba.images.data import ImageClassificationDataModule
from zamba.images.config import ImageModelEnum
from zamba.models.config import ModelEnum
from zamba.models.densepose.densepose_manager import MODELS as DENSEPOSE_MODELS
from zamba.models.utils import download_weights
from zamba.settings import get_model_cache_dir

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("dl-weights")
def dl_weights(
    cache_dir: Path = typer.Option(
        None,
        exists=False,
        help="Path to directory for downloading model weights. Alternatively, specify with environment variable `MODEL_CACHE_DIR`. If not specified, user's cache directory is used.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-o",
        help="Overwrite existing weights in the cache directory.",
    ),
):
    cache_dir = cache_dir or get_model_cache_dir()

    weights_filenames = []

    # get weights filenames for model files
    model_names = list(chain(ModelEnum.__members__.keys(), ImageModelEnum.__members__.values()))
    for model_name in model_names:
        with (MODELS_DIRECTORY / model_name / "config.yaml").open() as f:
            config_dict = yaml.safe_load(f)

        weights_filenames.append(config_dict["public_checkpoint"])

    # get weights filenames for densepose models
    for model_name, model_config in DENSEPOSE_MODELS.items():
        weights_filenames.append(model_config["weights"])

    # download weights
    for weights_filename in tqdm(weights_filenames, desc="Downloading weights for all models..."):
        cache_path = cache_dir / weights_filename

        if not overwrite and cache_path.exists():
            logger.info(
                f"Weights {weights_filename} already exist in {cache_path}, skipping. Use --overwrite to download again."
            )
            continue

        logger.info(f"Downloading weights {weights_filename} to {cache_path}")
        download_weights(weights_filename, cache_path)


@app.command("crop-bbox", help="Crop to bounding boxes from images and write results.")
def crop_bbox(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Path to data directory.",
    ),
    input_file: Path = typer.Option(
        ...,
        "--input-file",
        "-i",
        help="""
            Path to input image file; must be one of:
             - csv with `filepath` column (we will run MegaDetector to get the bounding boxes)
             - csv with columns `filepath`, `x1`, `y1`, `x2`, `y2` (bounding box format from zamba output)
             - COCO format json (set --bbox-format to `coco`)
             - Megadetector format json (set --bbox-format to `megadetector`)
        """,
    ),
    bbox_format: BboxInputFormat = typer.Option(
        BboxInputFormat.COCO,
        "--bbox-format",
        "-b",
        help="OPTIONAL: Bounding box format. Needed if passing a json file as input.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Path to output directory.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-w",
        help="Overwrite existing files in the output directory; defaults to not overwriting if a file exists.",
    ),
    detection_threshold: float = typer.Option(
        0.2,
        "--detection-threshold",
        "-t",
        help="Detection threshold for MegaDetector; 0.2 is default and commonly used.",
    ),
):
    # turn json in DF
    try:
        df = pd.read_csv(input_file)
        if "filepath" not in df.columns:
            raise typer.Abort("`filepath` column is required in input CSV")

    except (EmptyDataError, ParserError):
        # Handle CSV parsing errors
        logger.debug("Could not parse file as CSV, attempting to read as JSON")
        with input_file.open("r") as f:
            bbox_json = json.load(f)
        df = bbox_json_to_df(bbox_json, bbox_format)

    # fake ImageClassificationDataModule
    data_module = ImageClassificationDataModule(
        data_dir=data_dir,
        annotations=df,
        cache_dir=output_dir,
        crop_images=True,
        detection_threshold=detection_threshold,
    )

    out_df = data_module.preprocess_annotations(df, overwrite_cache=overwrite)

    # write out_df to csv
    out_df.to_csv(output_dir / "cropped_annotations.csv", index=False)
