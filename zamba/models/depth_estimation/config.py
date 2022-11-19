import os
from pathlib import Path

from loguru import logger
import pandas as pd
from pydantic import DirectoryPath, FilePath, validator, root_validator
from typing import Optional, Union

from zamba.models.config import (
    ZambaBaseModel,
    check_files_exist_and_load,
    get_filepaths,
    GPUS_AVAILABLE,
    validate_gpus,
    validate_model_cache_dir,
)
from zamba.models.depth_estimation.depth_manager import DepthEstimationManager
from zamba.models.utils import RegionEnum


class DepthEstimationConfig(ZambaBaseModel):
    """Configuration for running depth estimation on videos.

    Args:
        filepaths (FilePath, optional): Path to a CSV containing videos for inference, with one row per
            video in the data_dir. There must be a column called 'filepath' (absolute or
            relative to the data_dir). If None, uses all files in data_dir. Defaults to None.
        data_dir (DirectoryPath): Path to a directory containing videos for inference.
            Defaults to the working directory.
        save_to (Path or DirectoryPath, optional): Either a filename or a directory in which to
            save the output csv. If a directory is provided, predictions will be saved to
            depth_predictions.csv in that directory. Defaults to the working directory.
        overwrite (bool): If True, overwrite output csv path if it exists. Defaults to False.
        batch_size (int): Batch size to use for inference. Defaults to 64. Note: a batch is a set
            of frames, not videos, for the depth model.
        model_cache_dir (Path, optional): Path for downloading and saving model weights.
            Defaults to env var `MODEL_CACHE_DIR` or the OS app cache dir.
        weight_download_region (str): s3 region to download pretrained weights from. Options are
            "us" (United States), "eu" (Europe), or "asia" (Asia Pacific). Defaults to "us".
        num_workers (int): Number of subprocesses to use for data loading. The maximum value is
           the number of CPUs in the system. Defaults to 8.
        gpus (int): Number of GPUs to use for inference. Defaults to all of the available GPUs
            found on the machine.
    """

    filepaths: Optional[Union[FilePath, pd.DataFrame]] = None
    data_dir: DirectoryPath = ""
    save_to: Optional[Path] = None
    overwrite: bool = False
    batch_size: int = 64
    model_cache_dir: Optional[Path] = None
    weight_download_region: RegionEnum = RegionEnum("us")
    num_workers: int = 8
    gpus: int = GPUS_AVAILABLE

    class Config:
        # support pandas dataframe
        arbitrary_types_allowed = True

    def run_model(self):
        dm = DepthEstimationManager(
            model_cache_dir=self.model_cache_dir,
            batch_size=self.batch_size,
            weight_download_region=self.weight_download_region,
            num_workers=self.num_workers,
            gpus=self.gpus,
        )

        predictions = dm.predict(self.filepaths)
        self.save_to.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(self.save_to, index=False)
        logger.info(f"Depth predictions saved to {self.save_to}")

    _get_filepaths = root_validator(allow_reuse=True, pre=False, skip_on_failure=True)(
        get_filepaths
    )

    _validate_cache_dir = validator("model_cache_dir", allow_reuse=True, always=True)(
        validate_model_cache_dir
    )

    _validate_gpus = validator("gpus", allow_reuse=True, pre=True)(validate_gpus)

    @root_validator(skip_on_failure=True)
    def validate_save_to(cls, values):
        save_to = values["save_to"]
        if save_to is None:
            save_path = Path(os.getcwd()) / "depth_predictions.csv"
        elif save_to.suffix:
            save_path = save_to
        else:
            save_path = save_to / "depth_predictions.csv"

        if save_path.exists() and not values["overwrite"]:
            raise ValueError(
                f"{save_path} already exists. If you would like to overwrite, set overwrite=True."
            )

        values["save_to"] = save_path
        return values

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
        duplicated = files_df.filepath.duplicated()
        if duplicated.sum() > 0:
            logger.warning(
                f"Found {duplicated.sum():,} duplicate row(s) in filepaths csv. Dropping duplicates."
            )
            files_df = files_df[["filepath"]].drop_duplicates()

        values["filepaths"] = check_files_exist_and_load(
            df=files_df,
            data_dir=values["data_dir"],
            skip_load_validation=True,  # just check files exist
        ).filepath.values.tolist()

        return values
