from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import DirectoryPath, FilePath
import yaml


class BaseModel(PydanticBaseModel):
    """Set defaults for all models that inherit from the pydantic base model."""

    class Config:
        extra = "forbid"
        use_enum_values = True


class ModelClassEnum(str, Enum):
    cnnensemble = "cnnensemble"
    custom = "custom"
    sample = "sample"


class FrameworkEnum(str, Enum):
    keras = "keras"
    pytorch = "pytorch"


class ModelProfileEnum(str, Enum):
    fast = "fast"
    full = "full"


class RegionEnum(str, Enum):
    us = "us"
    eu = "eu"
    asia = "asia"


class TrainConfig(BaseModel):
    train_data: DirectoryPath = None
    val_data: DirectoryPath = None
    labels: FilePath = None
    model_path: FilePath = None
    framework: FrameworkEnum = "keras"
    model_class: ModelClassEnum = "custom"
    tempdir: Optional[Path] = None
    n_epochs: Optional[int] = 10
    height: Optional[int] = None
    width: Optional[int] = None
    augmentation: Optional[bool] = False
    early_stopping: Optional[bool] = False
    save_path: Optional[Path] = None


class PredictConfig(BaseModel):
    data_path: Union[DirectoryPath, FilePath] = None
    model_path: Union[DirectoryPath, FilePath] = None
    model_class: ModelClassEnum = "cnnensemble"
    pred_path: Optional[Path] = None
    proba_threshold: Optional[float] = None
    output_class_names: Optional[bool] = False
    tempdir: Optional[Path] = None
    verbose: Optional[bool] = False
    download_region: RegionEnum = "us"
    save: Optional[bool] = False
    model_kwargs: Optional[dict] = dict(resample=False, seperate_blank_model=False, profile="full")


class FineTuneConfig(BaseModel):
    pass


class ModelConfig(BaseModel):
    train_config: Optional[TrainConfig] = None
    predict_config: Optional[PredictConfig] = None

    class Config:
        json_loads = yaml.safe_load
