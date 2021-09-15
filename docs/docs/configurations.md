# Model Configurations

In both the command line and the Python module, options for video loading, training, and prediction can be set by passing a YAML file.
The basic structure of the configuration is:

```
video_loader_config:
  # video loading parameters, eg. video_height: 224

train_config:
  model_name: slowfast
  data_directory: vids_to_classify/
  # other training parameters, eg. labels, batch_size

predict_config:
  model_name: slowfast
  data_directoty: vids_to_classify/
  # other training parameters, eg. batch_size
```

For example, the configuration below will predict labels for the videos in `vids_to_classify` using the `time_distributed` model, and will resize all videos to 2224x224 pixels:

```
video_loader_config:
  video_height: 224
  video_width: 224

predict_config:
  model_name: slowfast
  data_directoty: vids_to_classify/
```

There are three basic classes used in model configuration: VideoLoaderConfig, TrainConfig, and PredictConfig. Each inherits from the pydantic [BaseModel](https://pydantic-docs.helpmanual.io/usage/models/) as a parent class.

## Video loading

Each video is loaded as a series of frames. For example, to evenly sample 20 frames from each video when loading, add to your YAML file:

```
video_loder_config:
  total_frames: 20
  evenly_sample_total_frames: True
```

All possible video loading and frame selection parameters are defined by the `VideoLoaderConfig` class<!-- TODO: add link to class definition><!-->. 

```python
>> from zamba.data.video import VideoLoaderConfig
>> default_video_loader = VideoLoaderConfig()
>> default_video_loader

VideoLoaderConfig(crop_bottom_pixels=None, i_frames=False, scene_threshold=None, megadetector_lite_config=None, video_height=None, video_width=None, total_frames=None, ensure_total_frames=True, fps=None, early_bias=False, frame_indices=None, evenly_sample_total_frames=False, pix_fmt='rgb24')
```

Let's go through each of those arguments.

#### `crop_bottom_pixels (int, optional)`

Number of pixels to crop from the bottom of the video (prior to resizing to `video_height`). Defaults to `None`

#### `i_frames (bool, optional)`

Only load the I-Frames. See https://en.wikipedia.org/wiki/Video_compression_picture_types#Intra-coded_(I)_frames/slices_(key_frames). Defaults to `False`

#### `scene_threshold (float, optional)`

Only load frames that correspond to scene changes. See http://www.ffmpeg.org/ffmpeg-filters.html#select_002c-aselect. Defaults to `None`

#### `megadetector_lite_config (MegadetectorLiteYoloXConfig, optional)`

The `megadetector_lite_config` is used to specify any parameters that should be passed to the MegadetectorLiteYoloX model for frame selection. For all possible options, see the MegadetectorLiteYoloXConfig<!-- TODO: add github link><!-->. Defaults to `None`

#### `video_height (int, optional), video_width (int, optional)`

Resize the video to this height and width in pixels. Defaults to `None`

#### `total_frames (int, optional)`

Number of frames that should ultimately be returned. Defaults to `None`

#### `ensure_total_frames (bool)`

Selecting the number of frames by resampling may result in one more or fewer frames due to rounding. If True, ensure the requested number of frames is returned by either clipping or duplicating the final frame. Raises an error if no frames have been selected. Otherwise, return the array unchanged. Defaults to `True`

#### `fps (int, optional)`

Resample the video evenly from the entire duration to a specific number of frames per second. Defaults to `None`

#### `early_bias (bool, optional)`

Resamples to 24 fps and selects 16 frames biased toward the front (strategy used by competition winner). Defaults to `False`

#### `frame_indices (list(int), optional)`

Select specific frame numbers. Note: frame selection is done after any resampling. Defaults to `None`

#### `evenly_sample_total_frames (bool, optional)`

Reach the total number of frames specified by evenly sampling from the duration of the video. Defaults to `False`.

#### `pix_fmt (str, optional)`

ffmpeg pixel format, defaults to `rgb24` for RGB channels; can be changed to `bgr24` for BGR.

## Training

All possible model training parameters are defined by the `TrainConfig` class<!-- TODO: add link to class definition><!-->. 

```python
>> from zamba.models.config import TrainConfig
>> default_train_config = TrainConfig(labels='example_labels.csv')
>> default_train_config
```
<!-- TODO: add output of default train config above when it's working><!-->

#### `labels (FilePath, required)`

Path to a CSV file with labels for training. The labels file must have columns for `filename` and `label`

####

```python
class TrainConfig(ZambaBaseModel):
    labels: FilePath
    data_directory: Optional[DirectoryPath] = os.getcwd()
    model_class: ModelEnum = ModelEnum.time_distributed
    model_name: str = None
    model_params: Optional[ModelParams] = None
    resume_from_checkpoint: Optional[Union[Path, str]] = None
    dry_run: Union[bool, int] = False
    batch_size: Optional[int] = 8
    auto_lr_find: bool = True
    backbone_finetune: bool = False
    backbone_finetune_params: BackboneFinetuneConfig = None
    gpus: Optional[Union[List[int], str, int]] = GPUS_AVAILABLE
    max_epochs: int = None
    early_stopping: bool = True
    early_stopping_params: EarlyStoppingConfig = None
    tensorboard_log_dir: str = "tensorboard_logs"
    split_proportions: Optional[Dict[str, int]] = {"train": 3, "val": 1, "holdout": 1}
```

## Prediction

All possible model inference parameters are defined by the `PredictConfig` class<!-- TODO: add link to class definition><!-->. 

```python
>> from zamba.models.config import PredictConfig
>> default_predict_config = PredictConfig()
>> default_predict_config
```
<!-- TODO: add output of default train config above when it's working><!-->
