from itertools import chain
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import yaml

from zamba import MODELS_DIRECTORY
from zamba.images.config import ImageModelEnum
from zamba.models.config import ModelEnum
from zamba.models.densepose.densepose_manager import MODELS as DENSEPOSE_MODELS
from zamba.models.utils import download_weights
from zamba.settings import get_model_cache_dir

app = typer.Typer(pretty_exceptions_show_locals=False)


def download_all_weights(
    cache_dir: Path,
    overwrite: bool = False,
    include_video: bool = True,
    include_image: bool = True,
    include_depth: bool = True,
    include_densepose: bool = True,
    weight_download_region: str = "us",
):
    """Download all required model weights.
    
    Args:
        cache_dir: Directory to store downloaded weights
        overwrite: Whether to overwrite existing weights
        include_video: Whether to download video model weights
        include_image: Whether to download image model weights
        include_depth: Whether to download depth model weights
        include_densepose: Whether to download densepose model weights
        weight_download_region: Region for downloading weights
    """
    weights_filenames = []

    # Video models
    if include_video:
        model_names = list(ModelEnum.__members__.keys())
        for model_name in model_names:
            config_path = MODELS_DIRECTORY / model_name / "config.yaml"
            if config_path.exists():
                with config_path.open() as f:
                    config_dict = yaml.safe_load(f)
                weights_filenames.append(config_dict["public_checkpoint"])

    # Image models
    if include_image:
        image_model_names = list(ImageModelEnum.__members__.values())
        for model_name in image_model_names:
            config_path = MODELS_DIRECTORY / model_name / "config.yaml"
            if config_path.exists():
                with config_path.open() as f:
                    config_dict = yaml.safe_load(f)
                weights_filenames.append(config_dict["public_checkpoint"])

    # Depth models
    if include_depth:
        for model_name, model_config in DENSEPOSE_MODELS.items():
            weights_filenames.append(model_config["weights"])

    # Densepose models
    if include_densepose:
        for model_name, model_config in DENSEPOSE_MODELS.items():
            weights_filenames.append(model_config["weights"])

    # Download weights
    for weights_filename in tqdm(weights_filenames, desc="Downloading weights for all models..."):
        cache_path = cache_dir / weights_filename

        if not overwrite and cache_path.exists():
            logger.info(
                f"Weights {weights_filename} already exist in {cache_path}, skipping. Use --overwrite to download again."
            )
            continue

        logger.info(f"Downloading weights {weights_filename} to {cache_path}")
        download_weights(weights_filename, cache_path, weight_download_region)


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
    download_all_weights(cache_dir, overwrite=overwrite)
