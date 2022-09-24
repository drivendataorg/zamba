# All configuration options

To make it easy to associate a model configuration with and a set of results, zamba accepts a `yaml` file to define all of the relevant parameters for training or prediction. You can then store the configuration you used with the results in order to easily reproduce it in the future.

In general, we've tried to pick defaults that are reasonable, but it is worth it to familiarize yourself with the options available.

The primary configurations you may want to set are:

* `VideoLoaderConfig`: Defines all possible parameters for how videos are loaded
* `PredictConfig`: Defines all possible parameters for model inference
* `TrainConfig`: Defines all possible parameters for model training

Here's a helpful diagram which shows how everything is related.

![](../media/zamba_config_diagram.png)

<a id='video-loading-arguments'></a>

## Video loading arguments

The [`VideoLoaderConfig` class](api-reference/data-video.md#zamba.data.video.VideoLoaderConfig) defines all of the optional parameters that can be specified for how videos are loaded before either inference or training. This includes selecting which frames to use from each video.

All video loading arguments can be specified either in a [YAML file](yaml-config.md) or when instantiating the [`VideoLoaderConfig` class](api-reference/data-video.md#zamba.data.video.VideoLoaderConfig) in Python. Some can also be specified directly in the command line.

Each model comes with a default video loading configuration. If no user-specified video loading configuration is passed - either through a YAML file or the Python `VideoLoaderConfig` class - all video loading arguments will be set based on the defaults for the given model.

=== "YAML file"
    ```yaml
    video_loader_config:
        model_input_height: 240
        model_input_width: 426
        total_frames: 16
        # ... other parameters
    ```
=== "Python"
    ```python
    from zamba.data.video import VideoLoaderConfig
    from zamba.models.config import PredictConfig
    from zamba.models.model_manager import predict_model

    predict_config = PredictConfig(data_dir="example_vids/")
    video_loader_config = VideoLoaderConfig(
        model_input_height=240,
        model_input_width=426,
        total_frames=16
        # ... other parameters
    )
    predict_model(
        predict_config=predict_config, video_loader_config=video_loader_config
    )
    ```

Let's look at the class documentation in Python.

```python
>> from zamba.data.video import VideoLoaderConfig
>> help(VideoLoaderConfig)

class VideoLoaderConfig(pydantic.main.BaseModel)
 |  VideoLoaderConfig(*,
 crop_bottom_pixels: int = None,
 i_frames: bool = False,
 scene_threshold: float = None,
 megadetector_lite_config: zamba.models.megadetector_lite_yolox.MegadetectorLiteYoloXConfig = None,
 frame_selection_height: int = None,
 frame_selection_width: int = None,
 total_frames: int = None,
 ensure_total_frames: bool = True,
 fps: float = None,
 early_bias: bool = False,
 frame_indices: List[int] = None,
 evenly_sample_total_frames: bool = False,
 pix_fmt: str = 'rgb24',
 model_input_height: int = None,
 model_input_width: int = None,
 cache_dir: pathlib.Path = None,
 cleanup_cache: bool = False) -> None

 ...
```

#### `crop_bottom_pixels (int, optional)`

Number of pixels to crop from the bottom of the video (prior to resizing to `frame_selection_height`). This can sometimes be useful if your videos have a persistent timestamp/camera brand logo at the bottom. Defaults to `None`

#### `i_frames (bool, optional)`

Only load the [I-Frames](https://en.wikipedia.org/wiki/Video_compression_picture_types#Intra-coded_(I)_frames/slices_(key_frames)). I-frames are highly dependent on the encoding of the video, so it is not recommended to use them unless you have verified that the i-frames of your videos are useful. Defaults to `False`

#### `scene_threshold (float, optional)`

Only load frames that correspond to [scene changes](http://www.ffmpeg.org/ffmpeg-filters.html#select_002c-aselect), which are detected when `scene_threshold` percent of pixels are different. This can be useful for selecting frames efficiently if in general you have large animals and stable backgrounds. Defaults to `None`

#### `megadetector_lite_config (MegadetectorLiteYoloXConfig, optional)`

The `megadetector_lite_config` is used to specify any parameters that should be passed to the [MegadetectorLite model](models/species-detection.md#megadetectorlite) for frame selection. For all possible options, see the [`MegadetectorLiteYoloXConfig` class](../api-reference/object-detection-megadetector_lite_yolox/#zamba.object_detection.yolox.megadetector_lite_yolox.MegadetectorLiteYoloXConfig). If `megadetector_lite_config` is `None` (the default), the MegadetectorLite model will not be used to select frames.

#### `frame_selection_height (int, optional), frame_selection_width (int, optional)`

Resize the video to this height and width in pixels, prior to frame selection. If None, the full size video will be used for frame selection. Using full size videos (setting to `None`) is recommended for MegadetectorLite, especially if your species of interest are smaller. Defaults to `None`

#### `total_frames (int, optional)`

Number of frames that should ultimately be returned. Defaults to `None`

#### `ensure_total_frames (bool)`

Some frame selection methods may yield varying numbers of frames depending on timestamps of the video frames. If `True`, ensure the requested number of frames is returned by either clipping or duplicating the final frame. If no frames are selected, returns an array of the desired shape with all zeros. Otherwise, return the array unchanged. Defaults to `True`

#### `fps (float, optional)`

Resample the video evenly from the entire duration to a specific number of frames per second. Use values less than 1 for rates lower than a single frame per second (e.g., `fps=0.5` will result in 1 frame every 2 seconds). Defaults to `None`

#### `early_bias (bool, optional)`

Resamples to 24 fps and selects 16 frames biased toward the beginning of the video. This strategy was used by the [Pri-matrix Factorization](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) machine learning
competition winner. Defaults to `False`

#### `frame_indices (list(int), optional)`

Select specific frame numbers. Note: frame selection is done after any resampling. Defaults to `None`

#### `evenly_sample_total_frames (bool, optional)`

Reach the total number of frames specified by evenly sampling from the duration of the video. Defaults to `False`

#### `pix_fmt (str, optional)`

FFmpeg pixel format, defaults to `rgb24` for RGB channels; can be changed to `bgr24` for BGR.

#### `model_input_height (int, optional), model_input_width (int, optional)`

After frame selection, resize the video to this height and width in pixels. This controls the height and width of the video frames returned by `load_video_frames`.  Defaults to `None`

#### `cache_dir (Path, optional)`

Cache directory where preprocessed videos will be saved upon first load. Alternatively, can be set with `VIDEO_CACHE_DIR` environment variable. Provided there is enough space on your machine, it is highly encouraged to cache videos for training as this will speed up all subsequent epochs after the first. If you are predicting on the same videos with the same video loader configuration, this will save time on future runs. Defaults to `None`, which means videos will not be cached.

#### `cleanup_cache (bool, optional)`

Whether to delete the cache directory after training or predicting ends. Defaults to `False`

<a id='prediction-arguments'></a>

## Prediction arguments

All possible model inference parameters are defined by the [`PredictConfig` class](api-reference/models-config.md#zamba.models.config.PredictConfig). Let's see the class documentation in Python:

```python
>> from zamba.models.config import PredictConfig
>> help(PredictConfig)

class PredictConfig(ZambaBaseModel)
 |  PredictConfig(*,
 data_dir: DirectoryPath = Path.cwd(),
 filepaths: FilePath = None,
 checkpoint: FilePath = None,
 model_name: zamba.models.config.ModelEnum = <ModelEnum.time_distributed: 'time_distributed'>,
 gpus: int = 0,
 num_workers: int = 3,
 batch_size: int = 2,
 save: bool = True,
 save_dir: Optional[Path] = None,
 overwrite: bool = False,
 dry_run: bool = False,
 proba_threshold: float = None,
 output_class_names: bool = False,
 weight_download_region: zamba.models.utils.RegionEnum = 'us',
 skip_load_validation: bool = False,
 model_cache_dir: pathlib.Path = None) -> None

 ...
```

**Either `data_dir` or `filepaths` must be specified to instantiate `PredictConfig`.** If neither is specified, the current working directory will be used as the default `data_dir`.

#### `data_dir (DirectoryPath, optional)`

Path to the directory containing videos for inference. Defaults to the current working directory.

#### `filepaths (FilePath, optional)`

Path to a csv containing a `filepath` column with paths to the videos that should be classified.

#### `checkpoint (Path or str, optional)`

Path to a model checkpoint to load and use for inference. If you train your own custom models, this is how you can pass those models to zamba when you want to predict on new videos. The default is `None`, which will load the pretrained checkpoint if the model specified by `model_name`.

#### `model_name (time_distributed|slowfast|european, optional)`

Name of the model to use for inference. The model options that ship with `zamba` are `blank_nonblank`, `time_distributed`, `slowfast`, and `european`. See the [Available Models](models/species-detection.md) page for details. Defaults to `time_distributed`

#### `gpus (int, optional)`

The number of GPUs to use during inference. By default, all of the available GPUs found on the machine will be used. An error will be raised if the number of GPUs specified is more than the number that are available on the machine.

#### `num_workers (int, optional)`

The number of CPUs to use during training. The maximum value for `num_workers` is the number of CPUs available on the machine. If you are using MegadetectorLite for frame selection, it is not recommended to use the total number of CPUs available. Defaults to `3`

#### `batch_size (int, optional)`

The batch size to use for inference. Defaults to `2`

#### `save (bool)`

Whether to save out predictions. If `False`, predictions are not saved. Defaults to `True`.

#### `save_dir (Path, optional)`

An optional directory in which to save the model predictions and configuration yaml.  If
no `save_dir` is specified and `save` is True, outputs will be written to the current working directory. Defaults to `None`

#### `overwrite (bool)`

If True, will overwrite `zamba_predictions.csv` and `predict_configuration.yaml` in `save_dir` if they exist. Defaults to False.

#### `dry_run (bool, optional)`

Specifying `True` is useful for ensuring a model implementation or configuration works properly by running only a single batch of inference. Defaults to `False`

#### `proba_threshold (float between 0 and 1, optional)`

For advanced uses, you may want the algorithm to be more or less sensitive to if a species is present. This parameter is a float, e.g., `0.6` corresponding to the probability threshold beyond which an animal is considered to be present in the video being analyzed.

By default no threshold is passed, `proba_threshold=None`. This will return a probability from 0-1 for each species that could occur in each video. If a threshold is passed, then the final prediction value returned for each class is `probability >= proba_threshold`, so that all class values become `0` (`False`, the species does not appear) or `1` (`True`, the species does appear).

#### `output_class_names (bool, optional)`

Setting this option to `True` yields the most concise output `zamba` is capable of. The highest species probability in a video is taken to be the _only_ species in that video, and the output returned is simply the video name and the name of the species with the highest class probability, or `blank` if the most likely classification is no animal. Defaults to `False`

#### `weight_download_region [us|eu|asia]`

Because `zamba` needs to download pretrained weights for the neural network architecture, we make these weights available in different regions. `us` is the default, but if you are not in the US you should use either `eu` for the European Union or `asia` for Asia Pacific to make sure that these download as quickly as possible for you.

#### `skip_load_validation (bool, optional)`

By default, before kicking off inference `zamba` will iterate through all of the videos in the data and verify that each can be loaded. Setting `skip_load_verification` to `True` skips this step. Validation can be very time intensive depending on the number of videos. It is recommended to run validation once, but not on future inference runs if the videos have not changed. Defaults to `False`

#### `model_cache_dir (Path, optional)`

Cache directory where downloaded model weights will be saved. If None and the `MODEL_CACHE_DIR` environment variable is not set, will use your default cache directory (e.g. `~/.cache`). Defaults to `None`

<a id='training-arguments'></a>

## Training arguments

All possible model training parameters are defined by the [`TrainConfig` class](api-reference/models-config.md#zamba.models.config.TrainConfig). Let's see the class documentation in Python:

```python
>> from zamba.models.config import TrainConfig
>> help(TrainConfig)

class TrainConfig(ZambaBaseModel)
 |  TrainConfig(*,
 labels: Union[FilePath, pandas.DataFrame],
 data_dir: DirectoryPath = # your current working directory ,
 checkpoint: FilePath = None,
 scheduler_config: Union[str, zamba.models.config.SchedulerConfig, NoneType] = 'default',
 model_name: zamba.models.config.ModelEnum = <ModelEnum.time_distributed: 'time_distributed'>,
 dry_run: Union[bool, int] = False,
 batch_size: int = 2,
 auto_lr_find: bool = False,
 backbone_finetune_config: zamba.models.config.BackboneFinetuneConfig =
            BackboneFinetuneConfig(unfreeze_backbone_at_epoch=5,
            backbone_initial_ratio_lr=0.01, multiplier=1,
            pre_train_bn=False, train_bn=False, verbose=True),
 gpus: int = 0,
 num_workers: int = 3,
 max_epochs: int = None,
 early_stopping_config: zamba.models.config.EarlyStoppingConfig =
            EarlyStoppingConfig(monitor='val_macro_f1', patience=5,
            verbose=True, mode='max'),
 weight_download_region: zamba.models.utils.RegionEnum = 'us',
 split_proportions: Dict[str, int] = {'train': 3, 'val': 1, 'holdout': 1},
 save_dir: pathlib.Path = # your current working directory ,
 overwrite: bool = False,
 skip_load_validation: bool = False,
 from_scratch: bool = False,
 use_default_model_labels: bool = True,
 model_cache_dir: pathlib.Path = None) -> None

 ...
```

#### `labels (FilePath or pd.DataFrame, required)`

Either the path to a CSV file with labels for training, or a dataframe of the training labels. There must be columns for `filename` and `label`. **`labels` must be specified to instantiate `TrainConfig`.**

#### `data_dir (DirectoryPath, optional)`

Path to the directory containing training videos. Defaults to the current working directory.

#### `checkpoint (Path or str, optional)`

Path to a model checkpoint to load and resume training from. The default is `None`, which automatically loads the pretrained checkpoint for the model specified by `model_name`. Since the default `model_name` is `time_distributed` the default `checkpoint` is `zamba_time_distributed.ckpt`

#### `scheduler_config (zamba.models.config.SchedulerConfig, optional)`

A [PyTorch learning rate schedule](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) to adjust the learning rate based on the number of epochs. Scheduler can either be `default` (the default), `None`, or a [`torch.optim.lr_scheduler`](https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py).

#### `model_name (time_distributed|slowfast|european, optional)`

Name of the model to use for inference. The model options that ship with `zamba` are `blank_nonblank`, `time_distributed`, `slowfast`, and `european`. See the [Available Models](models/species-detection.md) page for details. Defaults to `time_distributed`

#### `dry_run (bool, optional)`

Specifying `True` is useful for trying out model implementations more quickly by running only a single batch of train and validation. Defaults to `False`

#### `batch_size (int, optional)`

The batch size to use for training. Defaults to `2`

#### `auto_lr_find (bool, optional)`

Whether to run a [learning rate finder algorithm](https://arxiv.org/abs/1506.01186) when calling `pytorch_lightning.trainer.tune()` to try to find an optimal initial learning rate. The learning rate finder is not guaranteed to find a good learning rate; depending on the dataset, it can select a learning rate that leads to poor model training. Use with caution. See the PyTorch Lightning [docs](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#auto-lr-find) for more details. Defaults to `False`

#### `backbone_finetune_config (zamba.models.config.BackboneFinetuneConfig, optional)`

Set parameters to finetune a backbone model to align with the current learning rate. Derived from Pytorch Lightning's built-in [`BackboneFinetuning`](https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/finetuning.html). The default values are specified in the [`BackboneFinetuneConfig` class](api-reference/models-config.md#zamba.models.config.BackboneFinetuneConfig): `BackboneFinetuneConfig(unfreeze_backbone_at_epoch=5, backbone_initial_ratio_lr=0.01, multiplier=1, pre_train_bn=False, train_bn=False, verbose=True)`

#### `gpus (int, optional)`

The number of GPUs to use during training. By default, all of the available GPUs found on the machine will be used. An error will be raised if the number of GPUs specified is more than the number that are available on the machine.

#### `num_workers (int, optional)`

The number of CPUs to use during training. The maximum value for `num_workers` is the number of CPUs available in the system. If you are using the Megadetector, it is not recommended to use the total number of CPUs available. Defaults to `3`

#### `max_epochs (int, optional)`

The maximum number of epochs to run during training. Defaults to `None`

#### `early_stopping_config (zamba.models.config.EarlyStoppingConfig, optional)`

Parameters to pass to Pytorch lightning's [`EarlyStopping`](https://github.com/PyTorchLightning/pytorch-lightning/blob/c7451b3ccf742b0e8971332caf2e041ceabd9fe8/pytorch_lightning/callbacks/early_stopping.py#L35) to monitor a metric during model training and stop training when the metric stops improving. The default values are specified in the [`EarlyStoppingConfig` class](api-reference/models-config.md#zamba.models.config.EarlyStoppingConfig): `EarlyStoppingConfig(monitor='val_macro_f1', patience=5, verbose=True, mode='max')`

#### `weight_download_region [us|eu|asia]`

Because `zamba` needs to download pretrained weights for the neural network architecture, we make these weights available in different regions. `us` is the default, but if you are not in the US you should use either `eu` for the European Union or `asia` for Asia Pacific to make sure that these download as quickly as possible for you.

#### `split_proportions (dict(str, int), optional)`

The proportion of data to use during training, validation, and as a holdout set. Defaults to `{"train": 3, "val": 1, "holdout": 1}`

#### `save_dir (Path, optional)`

Directory in which to save model checkpoint and configuration file. If not specified, will save to a `version_n` folder in your current working directory.

#### `overwrite (bool, optional)`

 If `True`, will save outputs in `save_dir` and overwrite the directory if it exists. If False, will create an auto-incremented `version_n` folder within `save_dir` with model outputs. Defaults to `False`

#### `skip_load_validation (bool, optional)`

By default, before kicking off training `zamba` will iterate through all of the videos in the training data and verify that each can be loaded. Setting `skip_load_verification` to `True` skips this step. Validation can be very time intensive depending on the number of videos. It is recommended to run validation once, but not on future training runs if the videos have not changed. Defaults to `False`

#### `from_scratch (bool, optional)`

Whether to instantiate the model with base weights. This means starting from the imagenet weights for image based models and the Kinetics weights for video models. Only used if labels is not None. Defaults to `False`

#### `use_default_model_labels (bool, optional)`

Whether the species outputted by the model should be the default model classes (e.g. all 32 species classes for the `time_distributed` model). If you want the model classes to only be the species in your labels file (e.g. just gorillas and elephants), set to `False`. If either `use_default_model_labels` is `False` or the labels contain species that are not in the model, the model head will be replaced for finetuning. Defaults to `True`

#### `model_cache_dir (Path, optional)`

Cache directory where downloaded model weights will be saved. If None and the `MODEL_CACHE_DIR` environment variable is not set, will use your default cache directory, which is often an automatic temp directory at `~/.cache/zamba`. Defaults to `None`
