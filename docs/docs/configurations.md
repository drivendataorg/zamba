# Model Configurations

In both the command line and the Python module, options for video loading, training, and prediction can be set by passing a YAML file. Some - but not all - of these parameters can also be passed directly as arguments to the command line<!-- TODO: link to cli page><!-->.
The basic structure of a configuration is:

```console
$ cat basic_config.yaml
video_loader_config:
  # video loading parameters, eg. video_height: 224

predict_config:
  model_name: slowfast
  data_directoty: vids_to_classify/
  # other training parameters, eg. batch_size

train_config:
  model_name: slowfast
  data_directory: vids_to_classify/
  # other training parameters, eg. labels, batch_size
  ```

For example, the configuration below will predict labels for the videos in `vids_to_classify` using the `slowfast` model, and will resize all videos to 2224x224 pixels:

```console
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

## Prediction

All possible model inference parameters are defined by the `PredictConfig` class<!-- TODO: add link to class definition><!-->. 

```python
>> from zamba.models.config import PredictConfig
>> default_predict_config = PredictConfig()
>> default_predict_config
```
<!-- TODO: add output of default train config above when it's working><!-->

#### `data_directory (DirectoryPath, optional)`

Path to the directory containing training videos. Defaults to the current working directory.

#### `file_list (FilePath, optional)`

Path to a list of files for classification. Defaults to `None`

#### `checkpoint (Path or str, optional)`

Path to a model checkpoint to load and use for inference. To load a model from a checkpoint, the model name must also be specified. The default is `None`, which automatically loads the pretrained checkpoint for the model specified by `model-name`.

#### `model_class, model_name, model_params` <!-- TODO: what's the final status of these params?><!-->

#### `gpus (int, optional)`

The number of GPUs to use during inference. By default, all of the available GPUs found on the machine will be used. An error will be raised if the number of GPUs specified is more than the number that are available on the machine.

#### `batch-size (int, optional)`

The batch size to use for inference. Defaults to `8`.

#### `dry_run (bool, optional)`

Specifying `True` is useful for trying out model implementations more quickly by running only a single batch of inference. Defaults to `False`.

#### `columns (list(str), optional)`

List of possible species class labels for the data. The default is the 31 species from central and west Africa that are predicted by `time_distributed` and `slowfast` <!-- TODO: add link to list of species><!-->

#### `proba_threshold (float between 0 and 1, optional)`

For advanced uses, you may want the algorithm to be more or less sensitive to if a species is present. This parameter is a `FLOAT` number, e.g., `0.64` corresponding to the probability threshold beyond which an animal is considered to be present in the video being analyzed.

By default no threshold is passed, `proba_threshold=None`. This will return a probability from 0-1 for each species that could occur in each video. If a threshold is passed, then the final prediction value returned for each class is `probability >= proba_threshold`, so that all class values become `0` (`False`, the species does not appear) or `1` (`True`, the species does appear).

#### `output_class_names (bool, optional)`

Setting this option to `True` yields the most concise output `zamba` is capable of. The highest species probability in a video is taken to be the _only_ species in that video, and the output returned is simply the video name and the name of the species with the highest class probability, or `blank` if the most likely classification is no animal. Defaults to `False`

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

#### `data_directory (DirectoryPath, optional)`

Path to the directory containing training videos. Defaults to the current working directory.

#### `model_class, model_name, model_params` <!-- TODO: what's the final status of these params?><!-->

#### `resume_from_checkpoint (Path or str, optional)`

Path to a model checkpoint from which to resume training. Defaults to `None`

#### `dry_run (bool, optional)`

Specifying `True` is useful for trying out model implementations more quickly by running only a single batch of train and validation. Defaults to `False`.

#### `batch-size (int, optional)`

The batch size to use for training. Defaults to `8`.

#### auto_lr_find, backbone_finetune, backbone_finetune_params 
<!-- TODO: add these><!-->

#### `gpus (int, optional)`

The number of GPUs to use during training. By default, all of the available GPUs found on the machine will be used. An error will be raised if the number of GPUs specified is more than the number that are available on the machine.

#### `max_epochs (int, optional)`

The maximum number of epochs to run during training. Defaults to `None`

#### `early_stopping (bool, optional), early_stopping_params (EarlyStoppingConfig)`
#### tensorboard_log_dir

<!-- TODO: add these><!-->

#### `split_proportions (dict(str, int), optional)`

The proportion of data to use during training, validation, and as a holdout set. Defaults to `{"train": 3, "val": 1, "holdout": 1}`
