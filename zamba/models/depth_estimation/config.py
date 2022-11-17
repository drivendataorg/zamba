import os

from loguru import logger
import pandas as pd
from pathlib import Path
from pydantic.class_validators import root_validator, validator
from typing import Optional, Union, List

from zamba.models.config import (
    ZambaBaseModel,
    check_files_exist_and_load,
    get_filepaths,
    validate_model_cache_dir,
)
from zamba.models.depth_estimation.depth_manager import DepthEstimationManager
from zamba.models.utils import RegionEnum
from zamba.settings import VIDEO_SUFFIXES


class DepthEstimationConfig(ZambaBaseModel):
    """Configuration for running depth estimation on images. At a minimum, must provide either
    a list of full filepaths, or a list of relative filepaths along with the data_dir

    Args:
        filepaths (FilePath): Path to a CSV containing videos for inference, with one row per
            video in the data_dir. There must be a column called 'filepath' (absolute or
            relative to the data_dir). If None, uses all files in data_dir. Defaults to None.
        data_dir (DirectoryPath): Path to a directory containing videos for inference.
            Defaults to the working directory.
        save_to (Path, optional): Either a path or a directory to save the predicted depths. If a
            directory is provided, predictions will be saved to depth_predictions.csv in that
            directory. Defaults to os.getcwd()
        model_cache_dir (Path, optional): Path for downloading and saving model weights.
            Defaults to env var `MODEL_CACHE_DIR` or the OS app cache dir.
        weight_download_region (str): s3 region to download pretrained weights from. Options are
            "us" (United States), "eu" (Europe), or "asia" (Asia Pacific). Defaults to "us".
        batch_size (int): Batch size to use for inference. Defaults to 64. Note: a batch is a set
            of frames, not videos, for the depth model.
    """

    filepaths: Union[Path, List[Union[Path, str]]]
    data_dir: Optional[Path]
    save_to: Optional[Path] = None
    model_cache_dir: Optional[Path] = None
    weight_download_region: RegionEnum = RegionEnum("us")
    batch_size: int = 64

    _validate_cache_dir = validator("model_cache_dir", allow_reuse=True, always=True)(
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

        # TODO; should this error and not warn
        if save_path.exists():
            logger.warning(f"Predictions will NOT be saved out because {save_path} already exists")

        dm = DepthEstimationManager(
            model_cache_dir=self.model_cache_dir,
            batch_size=self.batch_size,
            weight_download_region=self.weight_download_region,
        )

        predictions = dm.predict(self.filepaths)

        if not save_path.exists():
            predictions.to_csv(save_path, index=False)
            logger.info(f"Depth predictions saved to {save_path}")

    _get_filepaths = root_validator(allow_reuse=True, pre=False, skip_on_failure=True)(
        get_filepaths
    )

    @root_validator(skip_on_failure=True)
    def validate_files(cls, values):
        # if globbing from data directory, already have valid dataframe
        if isinstance(values["filepaths"], pd.DataFrame):
            files_df = values["filepaths"]
        else:
            # make into dataframe even if only one column for clearer indexing
            files_df = pd.DataFrame(pd.read_csv(values["filepaths"]))

        if "filepath" not in files_df.columns:
            raise ValueError(f"{values['filepaths']} must contain a `filepath` column.")

        # can only contain one row per filepath
        num_duplicates = len(files_df) - files_df.filepath.nunique()
        if num_duplicates > 0:
            logger.warning(
                f"Found {num_duplicates} duplicate row(s) in filepaths csv. Dropping duplicates."
            )
            files_df = files_df[["filepath"]].drop_duplicates()

        values["filepaths"] = check_files_exist_and_load(
            df=files_df,
            data_dir=values["data_dir"],
            skip_load_validation=True,  # just check files exist
        ).filepath.values.tolist()

        return values
