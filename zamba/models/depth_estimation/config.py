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

# Union[
#     Path, List[Union[Path, str]]
# ]  # either a path to a df with one column for filepath or a list of filepaths


class DepthEstimationConfig(ZambaBaseModel):
    """_summary_

    Args:
        ZambaBaseModel (_type_): _description_
        save_to: # Directory for where to save the output files; defaults to os.getcwd(). If a path will save to that path, if a directory will save to depth_predictions.csv in that directory. Defaults to os.getcwd() directory.

        cache_dir: # Path for downloading and saving model weights. Defaults to env var `MODEL_CACHE_DIR` or the OS app cache dir. <-- TODO update

        batch_size # 256 in winning code, may want to change this

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    filepaths: Union[Path, List[Union[Path, str]]]
    img_dir: Optional[Path]
    save_to: Optional[Path] = None
    cache_dir: Optional[Path] = Path(".zamba_cache")
    batch_size: Optional[int] = 64

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
        else:
            logger.info(f"Predictions will be saved to {save_path}")

        logger.info("Instantiating depth manager")
        dm = DepthEstimationManager(
            img_dir=self.img_dir,
            model_cache_dir=self.cache_dir,
            batch_size=self.batch_size,
        )

        logger.info("Generating depth predictions")
        predictions = dm.predict(self.filepaths)

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

    # @root_validator(skip_on_failure=True)
    # def validate_files(cls, values):
    #     # check for duplicates
    #     if len(self.filepaths)
