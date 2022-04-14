import os

from loguru import logger
import pandas as pd
from pathlib import Path
from pydantic.class_validators import root_validator, validator
from typing import Optional, Union, List

from zamba.models.config import (
    ZambaBaseModel,
    check_files_exist_and_load,
    validate_model_cache_dir,
)
from zamba.models.depth_estimation.depth_manager import DepthEstimationManager


class DepthEstimationConfig(ZambaBaseModel):
    """Configuration for running depth estimation on images. At a minimum, must provide either
    a list of full filepaths, or a list of relative filepaths along with the img_dir

    Args:
        filepaths (Path or List[Path]): Either a path to a CSV file with a list of filepaths to
            process, or a list of filepaths to process
        img_dir (Path, Optional): Where to find the files listed in filepaths, as well as the other
            images before and after each frame to create stacked image arrays. If no img_dir is
            provided, it will be set to the parent of the first filepath and the depth module will
            assume that all filepaths are full paths, rather than relative to the img_dir.
        save_to (Path, optional): Either a path or a directory to save the predicted depths. If a
            directory is provided, predictions will be saved to depth_predictions.csv in that
            directory. Defaults to os.getcwd()
        cache_dir (Path, optional): Path for downloading and saving model weights. Defaults to
            .zamba_cache
        batch_size (int, optional): Batch size for running the depth model. Defaults to 64
    """

    filepaths: Union[Path, List[Union[Path, str]]]
    img_dir: Optional[Path]
    save_to: Optional[Path] = None
    cache_dir: Path = Path(".zamba_cache")
    batch_size: int = 64

    _validate_cache_dir = validator("cache_dir", allow_reuse=True, always=True)(
        validate_model_cache_dir
    )

    def run_model(self):
        # get path to save predictions
        if self.save_to is None:
            save_path = Path(os.getcwd()) / "depth_predictions.csv"
        elif self.save_to.suffix:
            save_path = self.save_to
        else:
            save_path = self.save_to / "depth_predictions.csv"

        if save_path.exists():
            logger.warning(f"Predictions will NOT be saved out because {save_path} already exists")

        dm = DepthEstimationManager(
            img_dir=self.img_dir,
            model_cache_dir=self.cache_dir,
            batch_size=self.batch_size,
        )
        predictions = dm.predict(self.filepaths)

        if not save_path.exists():
            predictions.to_csv(save_path, index=False)
            logger.info(f"Depth predictions saved to {save_path}")

    @root_validator(pre=False, skip_on_failure=True)
    def get_filepaths(cls, values):
        # load the filepaths if a path to a csv is provided
        if isinstance(values["filepaths"], Path):
            files_df = pd.read_csv(values["filepaths"])
            if "filepath" not in files_df.columns:
                raise ValueError(f"{values['filepaths']} must contain a `filepath` column.")
            values["filepaths"] = files_df["filepath"].values.tolist()

        return values

    @root_validator(skip_on_failure=True)
    def validate_files(cls, values):
        # check for duplicates
        if len(values["filepaths"]) != len(set(values["filepaths"])):
            logger.warning(
                f"Found {len(values['filepaths']) - len(set(values['filepaths']))} duplicate filepath(s). Dropping duplicates."
            )
            values["filepaths"] = list(set(values["filepaths"]))

        files_df = pd.DataFrame({"filepath": values["filepaths"]})
        values["filepaths"] = check_files_exist_and_load(
            df=files_df, skip_load_validation=True, data_dir=None
        ).filepath.values.tolist()

        return values
