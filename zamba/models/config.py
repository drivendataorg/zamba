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
    custom_keras = "custom_keras"
    sample = "sample"


class RegionEnum(str, Enum):
    us = "us"
    eu = "eu"
    asia = "asia"


class ModelConfig(BaseModel):
    model_class: ModelClassEnum = "cnnensemble"
    model_load_path: Union[DirectoryPath, FilePath] = None
    model_save_path: Optional[Path] = None
    model_kwargs: Optional[dict] = dict(resample=False, seperate_blank_model=False, profile="full")
    tempdir: Optional[Path] = None


class TrainConfig(BaseModel):
    train_data: DirectoryPath = None
    val_data: DirectoryPath = None
    labels: FilePath = None
    n_epochs: Optional[int] = 10


class PredictConfig(BaseModel):
    data_path: Union[DirectoryPath, FilePath] = None
    pred_path: Optional[Path] = None
    proba_threshold: Optional[float] = None
    output_class_names: Optional[bool] = False
    tempdir: Optional[Path] = None
    verbose: Optional[bool] = False
    download_region: RegionEnum = "us"
    save: Optional[bool] = True


class ManagerConfig(BaseModel):
    train_config: Optional[TrainConfig] = None
    predict_config: Optional[PredictConfig] = None
    model_config: Optional[ModelConfig] = None

    class Config:
        json_loads = yaml.safe_load
