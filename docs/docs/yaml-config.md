# Using YAML Configuration Files

In both the command line and the Python module, options for video loading, training, and prediction can be set by passing a YAML file instead of passing arguments directly. YAML files (`.yml` or `.yaml`) are commonly used to serialize data in an easily readable way.

The basic structure of a YAML model configuration is:

```yaml
$ cat basic_config.yaml
video_loader_config:
  video_height: 224
  video_width: 224
  total_frames: 16
  # other video loading parameters

predict_config:
  model_name: time_distributed
  data_directoty: example_vids/
  # other training parameters, eg. batch_size, video_height, video_width

train_config:
  model_name: time_distributed
  data_directory: example_vids/
  labels: example_labels.csv
  # other training parameters, eg. batch_size, video_height, video_width
```

For example, the configuration below will predict labels for the videos in `example_vids` using the `time_distributed` model. When videos are loaded, each will be resized to 224x224 pixels and 16 frames will be selected:

```yaml
video_loader_config:
  video_height: 224
  video_width: 224
  total_frames: 16

predict_config:
  model_name: time_distributed
  data_directoty: example_vids/
```

## Required arguments

Either `predict_config` or `train_config` is required, based on whether you will be running inference or training a model. See [All Optional Arguments](configurations.md) for a full list of what can be specified under each class. To run inference, either `data_directory` or `filepaths` must be specified. To train a model, both `data_directory` and `labels` must be specified.

In `video_loader_config`, you must specify at least `video_height`, `video_width`, and `total_frames`. 

* For `time_distributed` or `european`, `total_frames` must be 16
* For `slowfast`, `total_frames` must be 32

See the [Available Models](models.md) page for more details on each model's requirements.

## Command line interface

A YAML configuration file can be passed to the command line interface with the `--config` argument. For example, say the example configuration above is saved as `example_config.yaml`. To run prediction:

```console
$ zamba predict --config example_config.yaml
```

Only some of the possible parameters can be passed directly as arguments to the command line. Those not listed in `zamba predict --help` or `zamba train --help` must be passed in a YAML file (see the [Quickstart guide](quickstart.md#getting-help) for details).

## Python package

The main API for zamba is the `ModelManager` class that can be accessed with:
<!-- TODO: add link to source code><!--> 

```python
from zamba.models.manager import ModelManager
```

The `ModelManager` class is used by `zamba`’s command line interface to handle preprocessing the filenames, loading the videos, serving them to the model, and saving predictions. Therefore any functionality available to the command line interface is accessible via the `ModelManager` class.

To instantiate the `ModelManager` based on a configuration file saved at `test_config.yaml`:
```python
>>> manager = ModelManager.from_yaml('test_config.yaml')
>>> manager.config

ModelConfig(
  video_loader_config=VideoLoaderConfig(crop_bottom_pixels=None, i_frames=False, 
                                        scene_threshold=None, megadetector_lite_config=None, 
                                        video_height=224, video_width=224, 
                                        total_frames=16, ensure_total_frames=True, 
                                        fps=None, early_bias=False, frame_indices=None,
                                        evenly_sample_total_frames=False, pix_fmt='rgb24'
                                      ), 
  train_config=None, 
  predict_config=PredictConfig(data_directory=PosixPath('vids'), 
                               filepaths=                         filepath
                                          0    /home/ubuntu/vids/eleph.MP4
                                          1  /home/ubuntu/vids/leopard.MP4
                                          2    /home/ubuntu/vids/blank.MP4
                                          3    /home/ubuntu/vids/chimp.MP4, 
                                checkpoint='zamba_time_distributed.ckpt', 
                                model_params=ModelParams(scheduler=None, scheduler_params=None),
                                model_name='time_distributed', species=None, 
                                gpus=1, num_workers=7, batch_size=8, 
                                save=True, dry_run=False, proba_threshold=None,
                                output_class_names=False, weight_download_region='us', 
                                cache_dir=None, skip_load_validation=False)
                              )
```


## Default configurations

In the command line, the default configuration for each model is passed in using a specified YAML file that ships with `zamba`<!-- TODO: add link to github><!-->.

For example, the default configuration for the [`time_distributed` model](models.md#time-distributed) is:

```yaml
video_loader_config:
  video_height: 224
  video_width: 224
  crop_bottom_pixels: 50
  ensure_total_frames: True
  megadetector_lite_config:
    confidence: 0.25
    fill_model: "score_sorted"
    n_frames: 16
  total_frames: 16

train_config:
  # data_directory: YOUR_DATA_DIRECTORY_HERE
  # labels: YOUR_LABELS_CSV_HERE
  model_name: time_distributed
  # or
  # checkpoint: YOUR_CKPT_HERE
  batch_size: 8
  num_workers: 3
  scheduler_config:
    scheduler: MultiStepLR
    scheduler_params:
      milestones: [3]
      gamma: 0.5
      verbose: True
  auto_lr_find: True
  backbone_finetune: True
  backbone_finetune_params:
    unfreeze_backbone_at_epoch: 3
    verbose: True
    pre_train_bn: True
    multiplier: 1
  early_stopping: True
  early_stopping_params:
    patience: 5

predict_config:
  # data_directory: YOUR_DATA_DIRECTORY_HERE
  # or
  # filepaths: YOUR_FILEPATH_HERE
  model_name: time_distributed
  # or
  # checkpoint: YOUR_CKPT_HERE
```
