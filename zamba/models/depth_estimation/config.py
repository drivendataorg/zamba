from pathlib import Path
from pydantic.class_validators import root_validator, validator
from typing import Optional

from zamba.models.config import (
    ZambaBaseModel,
    check_files_exist_and_load,
    validate_model_cache_dir,
)


class DepthEstimationConfig(ZambaBaseModel):
    filepaths: Optional[
        Path
    ] = None  # Path to a CSV file with a list of filepaths to process. In this case, for each image for prediction there should be five filepaths (so we don't have to save duplicate information in image arrays). columns can be frame_0, frame_1, frame_2, frame_3, and frame_4 - frame 3 will be the target frame. or for now can only take in one, but then would only predict on one at a time - csv is better with all five columns
    save_dir: Optional[
        Path
    ] = None  # Directory for where to save the output files; defaults to os.getcwd().
    cache_dir: Optional[
        Path
    ] = None  # Path for downloading and saving model weights. Defaults to env var `MODEL_CACHE_DIR` or the OS app cache dir.
    tta: int  # assert 1 <= args.tta <= 2, args.tta in test.py, add validation for that

    @root_validator(pre=False, skip_on_failure=True)
    def get_filepaths(cls, values):
        """If no file list is passed, get all files in data directory. Warn if there
        are unsupported suffixes. Filepaths is set to a dataframe, where column `filepath`
        contains files with valid suffixes.
        """
        # likely will want to return a list of lists, where each one is the list of five filepaths for the target image
        pass  # pull from densepose config

    @root_validator(skip_on_failure=True)
    def validate_files(cls, values):
        pass  # pull from densepose config

    def run_model(self):
        # this is what will be called from the CLI

        # model = ...

        # set other args that are inputs to DepthManager

        # dm = DepthManager(...)

        # for fp_list in filepaths_list:
        # load five images and stack into an array
        # predict on stacked images
        # save out labels

        pass
