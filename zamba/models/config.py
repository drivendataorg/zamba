from enum import Enum
from pathlib import Path
from typing import Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, validate_arguments
import yaml


class ModelClassEnum(str, Enum):
    cnnensemble = "cnnensemble"
    custom = "custom"
    sample = "sample"


class ModelLibraryEnum(str, Enum):
    keras = "keras"
    pytorch = "pytorch"


class ModelProfileEnum(str, Enum):
    fast = "fast"
    full = "full"


class RegionEnum(str, Enum):
    us = "us"
    eu = "eu"
    asia = "asia"


class CnnModelKwargs(TypedDict):
    download_region: RegionEnum = "us"
    profile: ModelProfileEnum = "full"


class CnnPredictKwargs(TypedDict):
    resample: Optional[bool] = False
    seperate_blank_model: Optional[bool] = False


# @validate_arguments
class TrainConfig(BaseModel):
    train_data: Path = Path("train_videos")
    val_data: Path = Path("val_videos")
    labels: Path = Path("labels.csv")
    model_path: Optional[Path] = None
    model_library: ModelLibraryEnum = "keras"
    model_class: ModelClassEnum = "custom"
    tempdir: Optional[Path] = None
    n_epochs: Optional[int] = 10
    height: Optional[int] = None
    width: Optional[int] = None
    augmentation: Optional[bool] = False
    early_stopping: Optional[bool] = False
    save_path: Optional[Path] = None


# @validate_arguments
class PredictConfig(BaseModel):
    data_path: Path = Path(".")
    model_path: Path = Path(".")
    model_class: ModelClassEnum = "cnnensemble"
    pred_path: Optional[Path] = None
    proba_threshold: Optional[float] = None
    output_class_names: Optional[bool] = False
    tempdir: Optional[Path] = None
    verbose: Optional[bool] = False
    save: Optional[bool] = False
    model_kwargs: Optional[CnnModelKwargs] = dict()
    predict_kwargs: Optional[CnnPredictKwargs] = dict()


# @validate_arguments
class FineTuneConfig(BaseModel):
    pass


# TODO: get validation to work
# @validate_arguments
class ModelConfig(BaseModel):
    train_config: Optional[TrainConfig] = None
    predict_config: Optional[PredictConfig] = None

    class Config:
        json_loads = yaml.safe_load
